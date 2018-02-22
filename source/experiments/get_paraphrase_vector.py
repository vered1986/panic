# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
ap.add_argument('open_ie_lm_model_dir', help='the path to the world knowledge model')
ap.add_argument('word_embeddings', help='word embeddings to be used for world knowledge')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')

import tqdm
import codecs

import numpy as np

from open_ie_lm.model import Model
from common import load_binary_embeddings


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
    logger.info('Predicting predicates...')

    vectors = []

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        for first, second in [(w1, w2), (w2, w1)]:
            first_index, second_index = word2index.get(first, UNK), word2index.get(second, UNK)

            # Returns the top k predicted paraphrase vectors for (first, second)
            predicated_predicates = wk_model.predict_predicate(first_index, second_index, k=int(args.k))

            # Remove "unrelated" paraphrases if found enough related ones with higher scores
            pred_indices, pred_vecs, scores = zip(*predicated_predicates)
            paraphrases = [' '.join([words[i] for i in wk_model.index2pred[pred_index]])
                           for pred_index in pred_indices]
            if 'unrelated to' in paraphrases[0] and scores[0] > scores[1] * 2:
                curr_par_vector = pred_vecs[0]

            else:
                pred_indices, pred_vecs, scores = zip(*predicated_predicates[1:])
                curr_par_vector = np.average(pred_vecs, weights=scores)

            vectors.append(curr_par_vector)

    out_file = args.noun_compounds_file.replace('.tsv', '') + '_paraphrase_matrix'
    logger.info('Saving matrix to {}.npy'.format(out_file))
    np.save(out_file, np.vstack(vectors))

    # Save the noun compound to index mapping
    with codecs.open(args.noun_compounds_file.replace('.tsv', '_nc.tsv'), 'w', 'utf-8') as f_out:
        noun_compounds = [[(w1, w2), (w2, w1)] for (w1, w2) in noun_compounds]
        noun_compounds = [nc for nc_list in noun_compounds for nc in nc_list]
        for nc in noun_compounds:
            f_out.write('\t'.join(nc) + '\n')


if __name__ == '__main__':
    main()