# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('language_model_dir', help='the path to the trained language model')
ap.add_argument('word_embeddings_for_model', help='word embeddings to be used for the language model')
ap.add_argument('word_embeddings_for_dist', help='word embeddings to be used for w1 and w2 embeddings')
ap.add_argument('dataset_prefix', help='path to the train/test/val/rel data')
ap.add_argument('model_dir', help='where to store the result')
ap.add_argument('--use_w1_w2_embeddings', help='use w1 and w2 word embeddings as features', action='store_true')
ap.add_argument('--use_paraphrase_vectors', help='use the paraphrase vectors as features', action='store_true')
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
sys.path.append('../../source')

import tqdm

import numpy as np

from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from model.model import Model
from dataset_reader import DatasetReader
from common import load_binary_embeddings
from evaluation_common import evaluate, output_predictions


def main():
    if not (args.use_w1_w2_embeddings or args.use_paraphrase_vectors):
        raise ValueError('At least one of "use_w1_w2_embeddings" or "use_paraphrase_vectors" should be set.')

    # Load the datasets
    logger.info('Loading the datasets from {}'.format(args.dataset_prefix))
    train_set = DatasetReader(args.dataset_prefix + '/train.tsv')
    val_set = DatasetReader(args.dataset_prefix + '/val.tsv', label2index=train_set.label2index)
    test_set = DatasetReader(args.dataset_prefix + '/test.tsv', label2index=train_set.label2index)

    # Generate the feature vectors using the world knowledge model
    logger.info('Generating feature vectors...')
    train_features, val_features, test_features = [], [], []

    if args.use_paraphrase_vectors:
        logger.info('Reading word embeddings from {}...'.format(args.word_embeddings_for_model))
        wv, model_words = load_binary_embeddings(args.word_embeddings_for_model)

        logger.info('Loading language model from {}...'.format(args.language_model_dir))
        model = Model.load_model(args.language_model_dir, wv)

        model_words = ['[w1]', '[w2]', '[par]'] + model_words
        modelw2index = {w: i for i, w in enumerate(model_words)}
        UNK = modelw2index['unk']

    if args.use_w1_w2_embeddings:
        logger.info('Reading word embeddings from {}...'.format(args.word_embeddings_for_dist))
        wv, words = load_binary_embeddings(args.word_embeddings_for_dist)
        w2index = {w: i for i, w in enumerate(words)}
        UNK = w2index['unk']

        train_features.append(np.vstack([np.concatenate([wv[w2index.get(w1, UNK), :], wv[w2index.get(w2, UNK), :]])
                                         for (w1, w2) in train_set.noun_compounds]))
        val_features.append(np.vstack([np.concatenate([wv[w2index.get(w1, UNK), :], wv[w2index.get(w2, UNK), :]])
                                       for (w1, w2) in val_set.noun_compounds]))
        test_features.append(np.vstack([np.concatenate([wv[w2index.get(w1, UNK), :], wv[w2index.get(w2, UNK), :]])
                                        for (w1, w2) in test_set.noun_compounds]))

    # Tune the hyper-parameters using the validation set
    logger.info('Classifying...')
    reg_values = [0.5, 1, 2, 5, 10]
    penalties = ['l2']
    k_values = [10, 15, 25, 50] if args.use_paraphrase_vectors else [0]
    classifiers = ['logistic', 'svm']
    f1_results = []
    descriptions = []
    models = []
    all_test_instances = []

    for k in k_values:
        curr_train_features, curr_val_features, curr_test_features = train_features, val_features, test_features
        if args.use_paraphrase_vectors:
            curr_train_features += [predict_paraphrases(model, train_set.noun_compounds,
                                                        model_words, modelw2index, UNK, k)]
            curr_val_features += [predict_paraphrases(model, val_set.noun_compounds,
                                                      model_words, modelw2index, UNK, k)]
            curr_test_features += [predict_paraphrases(model, test_set.noun_compounds,
                                                       model_words, modelw2index, UNK, k)]

        train_instances = [np.concatenate(list(f)) for f in zip(*curr_train_features)]
        val_instances = [np.concatenate(list(f)) for f in zip(*curr_val_features)]
        test_instances = [np.concatenate(list(f)) for f in zip(*curr_test_features)]

        for cls in classifiers:
            for reg_c in reg_values:
                for penalty in penalties:
                    descriptions.append('K: {}, Classifier: {}, Penalty: {}, C: {:.2f}'.format(k, cls, penalty, reg_c))

                    # Create the classifier
                    if cls == 'logistic':
                        classifier = LogisticRegression(penalty=penalty, C=reg_c, multi_class='multinomial', n_jobs=20, solver='sag')
                    else:
                        classifier = LinearSVC(penalty=penalty, dual=False, C=reg_c)

                    logger.info('Training with classifier: {}, penalty: {}, c: {:.2f}...'.format(cls, penalty, reg_c))
                    classifier.fit(train_instances, train_set.labels)
                    val_pred = classifier.predict(val_instances)
                    p, r, f1, _ = evaluate(val_set.labels, val_pred, val_set.index2label, do_full_reoprt=False)
                    logger.info('K: {}, Classifier: {}, penalty: {}, c: {:.2f}, precision: {:.3f}, recall: {:.3f}, F1: {:.3f}'.
                                format(k, cls, penalty, reg_c, p, r, f1))
                    f1_results.append(f1)
                    models.append(classifier)
                    all_test_instances.append(test_instances)

    best_index = np.argmax(f1_results)
    description = descriptions[best_index]
    classifier = models[best_index]
    logger.info('Best hyper-parameters: {}'.format(description))

    # Save the best model to a file
    logger.info('Copying the best model...')
    joblib.dump(classifier, '{}/best.pkl'.format(args.model_dir))

    # Evaluate on the test set
    logger.info('Evaluation:')
    test_instances = all_test_instances[best_index]
    test_pred = classifier.predict(test_instances)
    precision, recall, f1, support = evaluate(test_set.labels, test_pred, test_set.index2label, do_full_reoprt=True)
    logger.info('Precision: {:.3f}, Recall: {:.3f}, F1: {:.3f}'.format(precision, recall, f1))

    # Write the predictions to a file
    output_predictions(args.model_dir + '/predictions.tsv', test_set.index2label, test_pred,
                       test_set.noun_compounds, test_set.labels)


def predict_paraphrases(model, noun_compounds, words, word2index, UNK, k):
    """
    Gets the language model and retrieves the best k paraphrases for each noun-compound.
    :param model: the trained language model/
    :param noun_compounds: the list of noun-compounds in the dataset.
    :param words: the model vocabulary.
    :param word2index: the word to index dictionary.
    :param UNK: the unknown vector.
    :param k: the number of paraphrase vectors to average.
    :return: the k best paraphrases for each noun-compound.
    """
    paraphrases = []

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        w1_index, w2_index = word2index.get(w1, UNK), word2index.get(w2, UNK)

        # Returns the top k predicted paraphrase vectors for (first, second)
        par_indices, par_vectors, scores = zip(*model.predict_paraphrase(
            w1_index, w2_index, k=k))

        par_text = [get_paraphrase_text(words, model.index2pred[par_index])
                    for par_index in par_indices]

        unrelated_indices = [i for i, p in enumerate(par_text) if p == '[w2] is unrelated to [w1]']

        # No unrelated - do regular average
        if len(unrelated_indices) == 0:
            averaged_vec = np.sum([par_vector * score for par_vector, score in zip(par_vectors, scores)],
                                  axis=0)
        else:
            averaged_vec = par_vectors[unrelated_indices[0]]

        paraphrases.append(averaged_vec)

    return np.vstack(paraphrases)


def get_paraphrase_text(words, par_indices):
    """
    Gets the paraphrase word indices and returns the text
    :param words: the model vocabulary.
    :param par_indices: the word indices.
    :return: the paraphrase text
    """
    paraphrase_words = [words[i] for i in par_indices]
    paraphrase = ' '.join(paraphrase_words)
    yield paraphrase


if __name__ == '__main__':
    main()