# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--k', help='the number of similar paraphrases, default = 15', default=15, type=int)
ap.add_argument('--unrelated_threshold', help='the minimal score the "is unrelated to" paraphrase has to get to be included', default=0.1)
ap.add_argument('train_gold_file', help='a tsv file with gold paraphrases and their scores')
ap.add_argument('language_model_dir', help='the path to the trained language model')
ap.add_argument('word_embeddings', help='word embeddings to be used for the language model')
ap.add_argument('google_ngrams_dataset_file', help='path to the google ngrams tsv file with w1, paraphrase, w2, score triplets')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../source')

import tqdm
import math
import codecs
import subprocess

import numpy as np

from collections import defaultdict

from language_model.model import Model
from common import load_binary_embeddings, most_similar_words_with_scores

TOP = 0.7


def main():
    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)

    if words[0] != '[w1]':
        words = ['[w1]', '[w2]'] + words

    word2index = {w: i for i, w in enumerate(words)}
    UNK = word2index['unk']

    logger.info('Loading language model from {}...'.format(args.language_model_dir))
    model = Model.load_model(args.language_model_dir, wv)
    logger.info('Predicting paraphrases...')

    train_predicted_paraphrases = get_paraphrase(model, 'train', words, word2index, UNK)
    test_predicted_paraphrases = get_paraphrase(model, 'test', words, word2index, UNK)

    thresholds = [0.000]
    scores = []

    for threshold in thresholds:
        curr_train_predicted_paraphrases = { (w1, w2) : [(p, score) for (p, score) in curr_paraphrases
                                                         if score >= threshold]
                                             for (w1, w2), curr_paraphrases in train_predicted_paraphrases.items() }

        curr_scores = evaluate(curr_train_predicted_paraphrases, threshold)
        score = 0.0 if min(curr_scores) < 0.1 else np.mean(curr_scores) # not to do bad in any evaluation!
        scores.append(score)
        logger.info('Threshold = {}, isomorphic = {:.3f}, non-isomorphic = {:.3f}'.format(
            threshold, curr_scores[0], curr_scores[1]))

    best_index = np.argmax(scores)
    best_threshold = thresholds[best_index]
    logger.info('Best threshold: {}, score: {}'.format(best_threshold, scores[best_index]))

    test_predicted_paraphrases = {(w1, w2): [(p, score) for (p, score) in curr_paraphrases if score >= best_threshold]
                                  for (w1, w2), curr_paraphrases in test_predicted_paraphrases.items()}

    out_file = 'test_predicted.tsv'
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in test_predicted_paraphrases.items():
            curr_paraphrases = sorted(curr_paraphrases, key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((w1, w2, paraphrase, str(score))) + '\n')


def evaluate(predictions, threshold):
    """
    Uses the Java scorer class to evaluate the current predictions against the gold standard
    :param predictions: a list of noun-compounds and their ranked predicted paraphrases
    :return: the evaluation score
    """
    # Save the evaluations to a temporary file
    prediction_file = 'temp/train_{}_predictions.tsv'.format(threshold)
    with codecs.open(prediction_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in predictions.items():
            curr_paraphrases = sorted(curr_paraphrases, key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((w1, w2, paraphrase, str(score))) + '\n')

    # java -classpath bin/ semeval2013.Scorer goldstandard.txt semeval_2013_paraphrases.tsv -verbose -isomorphic=true
    scores = []
    for isomporphic in ['true', 'false']:
        result = subprocess.run(['java', '-classpath', 'bin/', 'semeval2013.Scorer',
                                 args.train_gold_file, prediction_file, '-verbose',
                                 '-isomorphic={}'.format(isomporphic)],
                                stdout=subprocess.PIPE)
        # Take the last line
        scores.append(float(result.stdout.decode('utf-8').split('\n')[-2]))

    return scores


def get_paraphrase(model, filename, words, word2index, UNK):

    logger.info('Loading the noun compounds from {}.tsv'.format(filename))
    with codecs.open(filename + '.tsv', 'r', 'utf-8') as f_in:
        noun_compounds = [tuple(line.strip().split('\t')) for line in f_in]

    logger.info('Loading the Google N-grams dataset from {}'.format(args.google_ngrams_dataset_file))
    paraphrases = {(w1, w2): defaultdict(float) for (w1, w2) in noun_compounds}
    with codecs.open(args.google_ngrams_dataset_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, paraphrase, w2, score = line.strip().split('\t')
            if (w1, w2) in noun_compounds:
                paraphrase = paraphrase.replace('[w1]', w1).replace('[w2]', w2)
                if 'of said' not in paraphrase and paraphrase.split()[-2] != 'in' \
                        and paraphrase.split()[-1] != 'who':
                    paraphrases[(w1, w2)][paraphrase] = float(score)

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        w1_index, w2_index = word2index.get(w1, UNK), word2index.get(w2, UNK)

        # Returns the top k predicted paraphrase vectors for (first, second)
        pred_vectors = model.predict_paraphrase(w1_index, w2_index, k=int(args.k))

        for pred_index, pred_p, score in pred_vectors:
            for paraphrase in get_paraphrase_text(words, model.index2pred[pred_index]):
                paraphrase = paraphrase.replace('[w1]', w1).replace('[w2]', w2)
                if 'of said' not in paraphrase and paraphrase.split()[-2] != 'in' \
                        and paraphrase.split()[-1] != 'who':
                    paraphrases[(w1, w2)][paraphrase] = max(paraphrases[(w1, w2)][paraphrase], score)

        # Remove "unrelated" paraphrases if found enough related ones with higher scores
        curr_paraphrases = sorted(paraphrases[(w1, w2)].items(), key=lambda x: x[1], reverse=True)

        if len(curr_paraphrases) > 0 and 'is unrelated to' in curr_paraphrases[0][0] and curr_paraphrases[0][1] > args.unrelated_threshold:
            curr_paraphrases = [(p, score) for (p, score) in curr_paraphrases if 'unrelated' in p]
        else:
            curr_paraphrases = [(p, score) for (p, score) in curr_paraphrases if 'unrelated' not in p]

        paraphrases[(w1, w2)] = curr_paraphrases

    return paraphrases


def get_paraphrase_text(words, par_indices):
    paraphrase_words = [words[i] for i in par_indices]
    paraphrase = ' '.join(paraphrase_words)
    yield paraphrase


def how_many_to_include(score, w):
    return int(math.ceil((1/score - 1)*w+1))


if __name__ == '__main__':
    main()