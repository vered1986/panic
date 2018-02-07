# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--dynet_requested_gpus', help='number of gpus to use 0-4, default=0', type=int, default=1)
ap.add_argument('--dynet_mem', help='set dynet memory', default='512')
ap.add_argument('--dynet_seed', help='Dynet random seed, default=3016748844', default=3016748844)
ap.add_argument('--nepochs', help='number of epochs', type=int, default=10)
ap.add_argument('--batch_size', help='number of instance per minibatch', type=int, default=10)
ap.add_argument('--patience', help='how many epochs to wait without improvement', type=int, default=3)
ap.add_argument('--update', help='whether to update the embeddings', action='store_true')
ap.add_argument('--negative_sampling_ratio', help='the ratio from the training set of negative samples to add', type=float, default='0.0')
ap.add_argument('--continue_training', help='whether to load and keep training an existing model', action='store_true')
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
    level=logging.INFO,
    handlers=[
        logging.FileHandler('{}/log.txt'.format(args.model_dir)),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

# DyNet initialization
import dynet_config
dynet_config.set(requested_gpus=args.dynet_requested_gpus, mem=args.dynet_mem, random_seed=args.dynet_seed, autobatch=1)

import sys
sys.path.append('../')

import tqdm
import random
import codecs

import numpy as np

from collections import defaultdict

from model import Model
from common import load_binary_embeddings, most_similar_words


def main():
    # Load the word embeddings
    logger.info('Reading word embeddings from {}...'.format(args.embeddings_file))
    wv, words = load_binary_embeddings(args.embeddings_file)
    word2index = {w: i for i, w in enumerate(words)}

    # Load the dataset
    train_set = []
    logger.info('Loading the dataset from {}'.format(args.dataset))
    with codecs.open(args.dataset, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            train_set.append(line.strip().split('\t'))

    # Add negative samples
    if args.negative_sampling_ratio > 0:
        logger.info('Adding negative samples...')
        ratio = min(args.negative_sampling_ratio, 1.0)
        w1_num_negative = w2_num_negative = pred_num_negative = int((len(train_set) * ratio) / 3)

        preds_for_w2 = defaultdict(set)
        [preds_for_w2[w2].add(pred) for (w1, pred, w2) in train_set]
        w1_negatives = [('unk', pred, w2)
                        for (_, pred, _), (_, _, w2)
                        in zip(random.sample(train_set, w1_num_negative),
                               random.sample(train_set, w1_num_negative))
                        if pred not in preds_for_w2[w2]]

        preds_for_w1 = defaultdict(set)
        [preds_for_w1[w1].add(pred) for (w1, pred, w2) in train_set]
        w2_negatives = [(w1, pred, 'unk')
                        for (_, pred, _), (w1, _, _)
                        in zip(random.sample(train_set, w2_num_negative),
                               random.sample(train_set, w2_num_negative))
                        if pred not in preds_for_w1[w1]]

        related_pairs = set([(w1, w2) for (w1, pred, w2) in train_set])
        pred_negatives = [(w1, 'is unrelated to', w2)
                          for (w1, _, _), (_, _, w2)
                          in zip(random.sample(train_set, pred_num_negative),
                                 random.sample(train_set, pred_num_negative))
                          if (w1, w2) not in related_pairs]

        negative_samples = w1_negatives + w2_negatives + pred_negatives
        logger.info('Added {} negative samples...'.format(len(negative_samples)))
        train_set += negative_samples

    UNK = word2index['unk']
    train_set = [(word2index.get(w1, UNK),
                  tuple([word2index.get(w, UNK) for w in predicate.split()]),
                  word2index.get(w2, UNK)) for w1, predicate, w2 in train_set]

    index2pred = list(set([tuple(pred) for w1, pred, w2 in train_set]))

    # Create the model
    if args.continue_training:
        logger.info('Continuing previous training.')
        model = Model.load_model(args.model_dir, wv, update_embeddings=args.update)
        model.model_dir = args.model_dir
        model.n_epochs=args.nepochs
        model.minibatch_size=args.batch_size
        model.patience=args.patience
    else:
        model = Model(wv, index2pred, model_dir=args.model_dir, n_epochs=args.nepochs,
                      update_embeddings=args.update,
                      minibatch_size=args.batch_size, patience=args.patience)

    # Dedicate a random small part from the train set to validation
    random.shuffle(train_set)
    val_set = train_set[-1000:]
    train_set = train_set[:-1000]
    logger.info('Training with the following arguments: {}'.format(args))
    model.fit(train_set, val_set)

    # Try to predict some stuff (to be replaced with some kind of evaluation)
    logger.info('Evaluation:')
    for (w1, pred) in [('oil', 'be extract from'), ('company', 'sell'), ('meeting', 'be hold in'), ('cup', 'sell')]:
        w1_index = word2index.get(w1, -1)
        pred_indices = [word2index.get(w, -1) for w in pred.split()]
        for (w2_index, w2_p, score) in model.predict_w2(w1_index, pred_indices, k=10):
            logger.info('{} {} [{}]\t{:.3f}'.format(w1, pred, words[w2_index], score))
        logger.info('')

    for (w2, pred) in [('olive', 'be extract from'), ('textile', 'sell'), ('paris', 'be hold in')]:
        w2_index = word2index.get(w2, -1)
        pred_indices = [word2index.get(w, -1) for w in pred.split()]
        for (w1_index, w1_p, score) in model.predict_w1(w2_index, pred_indices, k=10):
            logger.info('[{}] {} {}\t{:.3f}'.format(words[w1_index], pred, w2, score))
        logger.info('')

    for (w1, w2) in [('oil', 'olive'), ('company', 'textile'), ('meeting', 'paris')]:
        w1_index = word2index.get(w1, -1)
        w2_index = word2index.get(w2, -1)

        for (pred_index, pred_p, score) in model.predict_predicate(w1_index, w2_index, k=10):
            pred_text = ' '.join([words[i] for i in model.index2pred[pred_index]])
            logger.info('{} [{}] {}\t{:.3f}'.format(w1, pred_text, w2, score))
        logger.info('')

        for (pred_index, pred_p, score) in model.predict_predicate(w2_index, w1_index, k=10):
            pred_text = ' '.join([words[i] for i in model.index2pred[pred_index]])
            logger.info('{} [{}] {}\t{:.3f}'.format(w2, pred_text, w1, score))
        logger.info('')
        logger.info('')


if __name__ == '__main__':
    main()