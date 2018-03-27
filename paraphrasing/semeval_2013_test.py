# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('test_gold_file', help='a tsv file with gold test paraphrases and their scores')
ap.add_argument('language_model_dir', help='the path to the trained language model')
ap.add_argument('patterns_file', help='the file with the POS patterns')
ap.add_argument('word_embeddings', help='word embeddings to be used for the language model')
ap.add_argument('reranker', help='the pkl file for the trained re-ranker')
ap.add_argument('--k', help='the number of paraphrases to retrieve for re-rankning, default = 1000', default=1000, type=int)
ap.add_argument('--minimum_score', help='the minimum score to keep a paraphrase', type=float, default=0.1)
ap.add_argument('--unrelated_threshold', help='the minimal score the "is unrelated to" paraphrase has to get to be included', default=0.1)
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../source')

import random

random.seed(133)

from sklearn.externals import joblib

from model.model import Model
from semeval_2013_common import *
from common import load_binary_embeddings

prepositions = ['with', 'at', 'from', 'into', 'during', 'including', 'until', 'against',
               'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'of', 'to',
               'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before', 'between',
               'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across',
               'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out',
               'around', 'down', 'off', 'above', 'near']


def main():
    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)

    if words[0] != '[w1]':
        words = ['[w1]', '[w2]', '[par]'] + words

    word2index = {w: i for i, w in enumerate(words)}
    UNK = word2index['unk']

    with codecs.open(args.patterns_file, 'r', 'utf-8') as f_in:
        pos_tags = { pos for line in f_in for pos in line.strip().split() }.difference({'[w1]', '[w2]'})
    pos2index = {p: i for i, p in enumerate(pos_tags)}
    prep2index = {p: i for i, p in enumerate(prepositions)}

    logger.info('Loading language model from {}...'.format(args.language_model_dir))
    model = Model.load_model(args.language_model_dir, wv)

    logger.info('Loading the reranker from {}...'.format(args.reranker))
    ranker = joblib.load(args.reranker)

    logger.info('Predicting test paraphrases...')
    test_gold = load_gold(args.test_gold_file)
    test_predicted_paraphrases = predict_paraphrases(model, test_gold.keys(), words, word2index, UNK,
                                                     args.k, args.unrelated_threshold)

    logger.info('Generating test features...')
    test_features, _ = generate_features(test_predicted_paraphrases, pos2index, prep2index,
                                         model, wv, word2index, UNK)

    logger.info('Reranking test paraphrases')
    test_predicted_paraphrases = rerank(test_predicted_paraphrases, test_features, ranker, args.minimum_score)

    logger.info('Evaluation:')
    isomorphic_score, nonisomorphic_score = evaluate(test_predicted_paraphrases, args.test_gold_file,
                                                     'test_predictions.tsv')
    logger.info('Isomorphic = {:.3f}, non-isomorphic = {:.3f}'.format(isomorphic_score, nonisomorphic_score))


if __name__ == '__main__':
    main()