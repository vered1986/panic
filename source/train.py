# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--dynet_requested_gpus', help='number of gpus to use 0-4, default=0', type=int, default=1)
ap.add_argument('--dynet_mem', help='set dynet memory', default='512')
ap.add_argument('--dynet_seed', help='Dynet random seed, default=3016748844', default=3016748844)
ap.add_argument('--nepochs', help='number of epochs', type=int, default=100)
ap.add_argument('--batch_size', help='number of instance per minibatch', type=int, default=10)
ap.add_argument('--patience', help='how many epochs to wait without improvement', type=int, default=3)
ap.add_argument('dataset', help='path to the training data')
ap.add_argument('model_dir', help='where to store the result')
ap.add_argument('embeddings_file', help='path to word embeddings files (.npy and .vocab)')
args = ap.parse_args()

# Log
import os
logdir = os.path.abspath(args.model_dir)
if not os.path.exists(logdir):
    os.mkdir(logdir)

import logging
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler('{}/log.txt'.format(args.model_dir)),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

# DyNet initialization
import dynet_config
dynet_config.set(requested_gpus=args.dynet_requested_gpus, mem=args.dynet_mem, random_seed=args.dynet_seed, autobatch=1)

import sys
sys.path.append('../')

import codecs

from model import Model
from common import load_binary_embeddings


def main():
    # Load the word embeddings
    logger.info('Reading word embeddings from {}...'.format(args.embeddings_file))
    wv, words = load_binary_embeddings(args.embeddings_file)
    word2index = {w: i for i, w in enumerate(words)}

    # Load the dataset
    logger.info('Loading the dataset from {}'.format(args.dataset))
    with codecs.open(args.dataset, 'r', 'utf-8') as f_in:
        train_set = [line.strip().split('\t') for line in f_in]

    train_set = [(word2index.get(w1, -1),
                  [word2index.get(w, -1) for w in predicate.split()],
                  word2index.get(w2, -1)) for w1, predicate, w2 in train_set]

    train_set = [(w1, predicate, w2) for (w1, predicate, w2) in train_set
                 if w1 > 0 and w2 > 0 and all([w > 0 for w in predicate])]

    # Create the model
    model = Model(wv, model_dir=args.model_dir,
                       n_epochs=args.nepochs, minibatch_size=args.batch_size, patience=args.patience)
    logger.info('Training with the following arguments: {}'.format(args))
    model.fit(train_set)

    # Save a predicate matrix
    predicates = list(set([tuple(pred) for w1, pred, w2 in train_set]))
    # pred_to_index = { pred : i for i, pred in enumerate(predicates) }
    predicate_matrix = model.save_predicate_matrix(predicates, args.model_dir + '/predicate_matrix')

    # Try to predict some stuff (to be replaced with some kind of evaluation)
    logger.info('Evaluation:')
    # oil be extracted from olives, company sell textile,  meeting be hold in paris
    for (w1, pred) in [('oil', 'be extracted from'), ('company', 'sell'), ('meeting', 'be hold in')]:
        w1_index = word2index.get(w1, -1)
        pred_indices = [word2index.get(w, -1) for w in pred.split()]
        w2_indices = model.predict_w2(w1_index, pred_indices, k=10)
        for w2_index in w2_indices:
            logger.info('{} {} {}'.format(w1, pred, words[w2_index]))

    for (w2, pred) in [('olive', 'be extracted from'), ('textile', 'sell'), ('paris', 'be hold in')]:
        w2_index = word2index.get(w2, -1)
        pred_indices = [word2index.get(w, -1) for w in pred.split()]
        w1_indices = model.predict_w1(w2_index, pred_indices, k=10)
        for w1_index in w1_indices:
            logger.info('{} {} {}'.format(words[w1_index], pred, w2))

    for (w1, w2) in [('olive', 'oil'), ('textile', 'company'), ('paris', 'meeting')]:
        w1_index = word2index.get(w1, -1)
        w2_index = word2index.get(w2, -1)
        pred_indices = model.predict_predicate(w1_index, w2_index, predicate_matrix, k=10)
        for pred_index in pred_indices:
            logger.info('{} {} {}'.format(w1, ' '.join([words[i] for i in predicates[pred_index]]), w2))


if __name__ == '__main__':
    main()