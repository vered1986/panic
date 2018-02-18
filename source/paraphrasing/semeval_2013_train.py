# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('train_gold_file', help='a tsv file with gold train paraphrases and their scores')
ap.add_argument('language_model_dir', help='the path to the trained language model')
ap.add_argument('patterns_file', help='the file with the POS patterns')
ap.add_argument('word_embeddings', help='word embeddings to be used for the language model')
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

import codecs
import random
import itertools

random.seed(133)

import numpy as np

from sklearn import svm
from sklearn.externals import joblib

from semeval_2013_common import *
from language_model.model import Model
from common import load_binary_embeddings


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

    train_gold = load_gold(args.train_gold_file)
    train_gold_features, train_gold_scores = generate_features(train_gold, pos2index, prep2index,
                                                               model, wv, word2index, UNK)
    logger.info('Loaded train gold paraphrases from {}'.format(args.train_gold_file))

    logger.info('Learning to rank...')
    ranker = learn_to_rank(train_gold_features, train_gold_scores)
    logger.info('Done training!')

    logger.info('Predicting train paraphrases...')
    train_predicted_paraphrases = predict_paraphrases(model, train_gold.keys(), words, word2index, UNK,
                                                      args.k, args.unrelated_threshold)

    logger.info('Generating train features...')
    train_features, _ = generate_features(train_predicted_paraphrases, pos2index, prep2index,
                                          model, wv, word2index, UNK)

    logger.info('Reranking train paraphrases')
    train_predicted_paraphrases = rerank(train_predicted_paraphrases, train_features, ranker, args.minimum_score)

    # Save the reranker
    joblib.dump(ranker, 'ranker.pkl')

    logger.info('Evaluation:')
    isomorphic_score, nonisomorphic_score = evaluate(train_predicted_paraphrases, args.train_gold_file,
                                                     'train_predictions.tsv')
    logger.info('Isomorphic = {:.3f}, non-isomorphic = {:.3f}'.format(isomorphic_score, nonisomorphic_score))


def learn_to_rank(X_train, y_train):
    """
    Learn to rank
    :param X_train: a list of paraphrase matrices
    :param y_train:
    :return:
    """
    Xp, yp = [], []
    for instance_features, instance_ranks in zip(X_train, y_train):

        # Shuffle so that we always have two classes -1 and 1
        indices = list(range(len(instance_features)))
        random.shuffle(indices)

        for i1, i2 in itertools.combinations(indices, 2):
            # Same pair/ranking
            if instance_ranks[i1] == instance_ranks[i2]:
                continue

            Xp.append(instance_features[i1] - instance_features[i2])
            yp.append(np.sign(instance_ranks[i1] - instance_ranks[i2]))

    ranker = svm.SVC(kernel='linear', C=0.1)
    ranker.fit(Xp, yp)
    return ranker


if __name__ == '__main__':
    main()