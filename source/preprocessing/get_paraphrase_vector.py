# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
ap.add_argument('world_knowledge_model_dir', help='the path to the world knowledge model')
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

from world_knowledge.model import Model
from common import load_binary_embeddings


def main():
    logger.info('Loading the noun compounds from {}'.format(args.noun_compounds_file))
    with codecs.open(args.noun_compounds_file, 'r', 'utf-8') as f_in:
        noun_compounds = [line.strip().split('\t') for line in f_in]

    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)
    word2index = {w: i for i, w in enumerate(words)}

    logger.info('Loading world knowledge model from {}...'.format(args.world_knowledge_model_dir))
    wk_model = Model.load_model(args.world_knowledge_model_dir + '/best', wv, update_embeddings=False)
    logger.info('Predicting predicates...')

    vectors = []

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        w1_index, w2_index = word2index.get(w1, -1), word2index.get(w2, -1)
        if w1_index > 0 and w2_index > 0:
            vectors.append(wk_model.predict_predicate(w1_index, w2_index))
            vectors.append(wk_model.predict_predicate(w2_index, w1_index))
        else:
            vectors.append(np.zeros(wv.shape[1] * 2))
            vectors.append(np.zeros(wv.shape[1] * 2))

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