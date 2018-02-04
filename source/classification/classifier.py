# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--w1_w2_embeddings', help='word embeddings to be used for the constituent words', default=None)
ap.add_argument('--paraphrase_matrix', help='the path to the paraphrase matrix', default=None)
ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
ap.add_argument('model_dir', help='where to store the result')
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

import sys
sys.path.append('../')

import codecs

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from dataset_reader import DatasetReader
from common import load_binary_embeddings
from evaluation_common import evaluate, output_predictions


def main():
    if args.paraphrase_matrix is None and args.w1_w2_embeddings is None:
        raise ValueError('At least one of "paraphras_matrix" or "w1_w2_embeddings" should be set.')

    word2index = { 'unk' : 0 }
    if args.w1_w2_embeddings is not None:
        logger.info('Reading word embeddings from {}...'.format(args.w1_w2_embeddings))
        w_wv, words = load_binary_embeddings(args.w1_w2_embeddings)
        word2index = {w: i for i, w in enumerate(words)}

    # Load the datasets
    logger.info('Loading the datasets from {}'.format(args.dataset_prefix))
    train_set = DatasetReader(args.dataset_prefix + '/train.tsv', word2index)
    val_set = DatasetReader(args.dataset_prefix + '/val.tsv', word2index, label2index=train_set.label2index)
    test_set = DatasetReader(args.dataset_prefix + '/test.tsv', word2index, label2index=train_set.label2index)

    # Generate the feature vectors using the world knowledge model
    logger.info('Generating feature vectors...')
    train_features, val_features, test_features = [], [], []

    if args.paraphrase_matrix is not None:
        noun_compounds_file = args.paraphrase_matrix.replace('_paraphrase_matrix.npy', '_nc.tsv')
        with codecs.open(noun_compounds_file, 'r', 'utf-8') as f_in:
            noun_compounds = [tuple(line.strip().split('\t')) for line in f_in]
            nc2index = { nc : i for i, nc in enumerate(noun_compounds) }

        paraphrase_matrix = np.load(args.paraphrase_matrix)

        train_features.append(np.vstack([paraphrase_matrix[nc2index[(w1, w2)], :] for (w1, w2) in train_set.str_noun_compounds]))
        val_features.append(np.vstack([paraphrase_matrix[nc2index[(w1, w2)], :] for (w1, w2) in val_set.str_noun_compounds]))
        test_features.append(np.vstack([paraphrase_matrix[nc2index[(w1, w2)], :] for (w1, w2) in test_set.str_noun_compounds]))

        train_features.append(np.vstack([paraphrase_matrix[nc2index[(w2, w1)], :] for (w1, w2) in train_set.str_noun_compounds]))
        val_features.append(np.vstack([paraphrase_matrix[nc2index[(w2, w1)], :] for (w1, w2) in val_set.str_noun_compounds]))
        test_features.append(np.vstack([paraphrase_matrix[nc2index[(w2, w1)], :] for (w1, w2) in test_set.str_noun_compounds]))

    if args.w1_w2_embeddings is not None:
        train_features.append(np.vstack([np.concatenate([w_wv[w1, :], w_wv[w2, :]]) for (w1, w2) in train_set.noun_compounds]))
        val_features.append(np.vstack([np.concatenate([w_wv[w1, :], w_wv[w2, :]]) for (w1, w2) in val_set.noun_compounds]))
        test_features.append(np.vstack([np.concatenate([w_wv[w1, :], w_wv[w2, :]]) for (w1, w2) in test_set.noun_compounds]))

    train_instances = [np.concatenate(list(f)) for f in zip(*train_features)]
    val_instances = [np.concatenate(list(f)) for f in zip(*val_features)]
    test_instances = [np.concatenate(list(f)) for f in zip(*test_features)]

    # Tune the hyper-parameters using the validation set
    logger.info('Classifying...')
    reg_values = [0.5, 1, 2, 5, 10]
    penalties = ['l2']
    classifiers = ['logistic', 'svm']
    f1_results = []
    descriptions = []
    models = []

    for cls in classifiers:
        for reg_c in reg_values:
            for penalty in penalties:
                descriptions.append('Classifier: {}, Penalty: {}, C: {:.2f}'.format(cls, penalty, reg_c))

                # Create the classifier
                if cls == 'logistic':
                    classifier = LogisticRegression(penalty=penalty, C=reg_c, multi_class='multinomial', n_jobs=20, solver='sag')
                else:
                    classifier = LinearSVC(penalty=penalty, dual=False, C=reg_c)

                logger.info('Training with classifier: {}, penalty: {}, c: {:.2f}...'.format(cls, penalty, reg_c))
                classifier.fit(train_instances, train_set.labels)
                val_pred = classifier.predict(val_instances)
                p, r, f1, support = evaluate(val_set.labels, val_pred, val_set.index2label, do_full_reoprt=False)
                logger.info('Classifier: {}, penalty: {}, c: {:.2f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.
                            format(cls, penalty, reg_c, p, r, f1))
                f1_results.append(f1)
                models.append(classifier)

    best_index = np.argmax(f1_results)
    description = descriptions[best_index]
    classifier = models[best_index]
    logger.info('Best hyper-parameters: {}'.format(description))

    # Save the best model to a file
    logger.info('Copying the best model...')
    joblib.dump(classifier, '{}/best.pkl'.format(args.model_dir))

    # Evaluate on the test set
    logger.info('Evaluation:')
    test_pred = classifier.predict(test_instances)
    precision, recall, f1, support = evaluate(test_set.labels, test_pred, test_set.index2label, do_full_reoprt=True)
    logger.info('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision, recall, f1))

    # Write the predictions to a file
    output_predictions(args.model_dir + '/predictions.tsv', test_set.index2label, test_pred,
                       test_set.str_noun_compounds, test_set.labels)


if __name__ == '__main__':
    main()