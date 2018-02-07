# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
ap.add_argument('open_ie_lm_model_dir', help='the path to the world knowledge model')
ap.add_argument('word_embeddings', help='word embeddings to be used for world knowledge')
ap.add_argument('--k', help='the number of similar paraphrases (in each direction). default = 15', default=15, type=int)
ap.add_argument('--unrelated_threshold', help='the minimal score the "is unrelated to" paraphrase has to get to be included', default=0.1)
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../source')

import tqdm
import json
import codecs

from open_ie_lm.model import Model
from common import load_binary_embeddings, most_similar_words_with_scores


def main():
    logger.info('Loading the noun compounds from {}'.format(args.noun_compounds_file))
    with codecs.open(args.noun_compounds_file, 'r', 'utf-8') as f_in:
        noun_compounds = [line.strip().split('\t') for line in f_in]

    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)
    word2index = {w: i for i, w in enumerate(words)}
    UNK = word2index['unk']

    logger.info('Loading world knowledge model from {}...'.format(args.open_ie_lm_model_dir))
    wk_model = Model.load_model(args.open_ie_lm_model_dir, wv, update_embeddings=False)
    logger.info('Predicting paraphrases...')
    paraphrases = {}

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        curr_paraphrases = []

        for first, second in [(w1, w2), (w2, w1)]:
            first_index, second_index = word2index.get(first, UNK), word2index.get(second, UNK)

            # Returns the top k predicted paraphrase vectors for (first, second)
            pred_vectors = wk_model.predict_predicate(first_index, second_index, k=int(args.k))

            p_with_scores = []
            for (pred_index, pred_p, score) in pred_vectors:
                par_indices = wk_model.index2pred[pred_index]
                paraphrase = ' '.join([words[i] for i in par_indices])
                p_with_scores.append((paraphrase, score))

            curr_paraphrases.extend([('{} {} {}'.format(first, paraphrase, second), score)
                                     for (paraphrase, score) in p_with_scores])

            # Remove "unrelated" paraphrases if found enough related ones with higher scores
            curr_paraphrases = sorted(curr_paraphrases, key=lambda x : x[1], reverse=True)

            if curr_paraphrases[0][0] == 'is unrelated to' and curr_paraphrases[0][1] > args.unrelated_threshold:
                curr_paraphrases = [(p, score) for (p, score) in curr_paraphrases if 'unrelated' in p]
            else:
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