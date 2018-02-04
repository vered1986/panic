# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
ap.add_argument('world_knowledge_model_dir', help='the path to the world knowledge model')
ap.add_argument('word_embeddings', help='word embeddings to be used for world knowledge')
ap.add_argument('--k', help='the number of similar paraphrases (in each direction). default = 15', default=15, type=int)
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')

import tqdm
import json
import codecs

from world_knowledge.model import Model
from common import load_binary_embeddings, most_similar_words_with_scores


def main():
    logger.info('Loading the noun compounds from {}'.format(args.noun_compounds_file))
    with codecs.open(args.noun_compounds_file, 'r', 'utf-8') as f_in:
        noun_compounds = [line.strip().split('\t') for line in f_in]

    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)
    word2index = {w: i for i, w in enumerate(words)}

    logger.info('Loading world knowledge model from {}...'.format(args.world_knowledge_model_dir))
    wk_model = Model.load_model(args.world_knowledge_model_dir + '/best', wv, update_embeddings=False)
    logger.info('Predicting paraphrases...')
    paraphrases = {}

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        curr_paraphrases = []

        for first, second in [(w1, w2), (w2, w1)]:
            first_index, second_index = word2index.get(first, -1), word2index.get(second, -1)
            if first_index > 0 and second_index > 0:
                pred_p = wk_model.predict_predicate(first_index, second_index)
                p_with_scores = most_similar_words_with_scores(wk_model.predicate_matrix, pred_p, k=int(args.k))
                curr_paraphrases.extend([('{} {} {}'.format(first,
                                                   ' '.join([words[i] for i in wk_model.index2pred[pred_index]]),
                                                   second), score)
                                 for (pred_index, score) in p_with_scores])

                # Remove "unrelated" paraphrases if found enough related ones with higher scores
                curr_paraphrases = sorted(curr_paraphrases, key=lambda x : x[1], reverse=True)
                best_paraphrases = curr_paraphrases[:int(args.k)//3]
                if not any(['unrelated' in p for p, score in best_paraphrases]):
                    curr_paraphrases = [(p, score) for (p, score) in curr_paraphrases if 'unrelated' not in p]

        paraphrases[(w1, w2)] = curr_paraphrases

    out_file = args.noun_compounds_file.replace('.tsv', '_predicted_paraphrases.jsonl')
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in paraphrases.items():
            f_out.write(json.dumps({
                'w1': w1,
                'w2': w2,
                'paraphrases': list(curr_paraphrases)
            }) + '\n')


if __name__ == '__main__':
    main()