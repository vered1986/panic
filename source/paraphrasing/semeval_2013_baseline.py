# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('test_gold_file', help='a tsv file with gold test paraphrases and their scores')
ap.add_argument('google_ngrams_dataset_file', help='path to the google ngrams tsv file with w1, paraphrase, w2, score triplets')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')

import tqdm
import codecs

from collections import defaultdict

from semeval_2013_common import evaluate, load_gold


def main():
    logger.info('Loading the gold test file from {}'.format(args.test_gold_file))
    test_gold = load_gold(args.test_gold_file)

    logger.info('Predicting paraphrases...')
    noun_compounds = set(test_gold.keys())
    test_predicted = {(w1, w2): defaultdict(float) for (w1, w2) in noun_compounds}

    with codecs.open(args.google_ngrams_dataset_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, paraphrase, w2, score = line.strip().split('\t')
            score = float(score)

            if (w1, w2) in noun_compounds:
                test_predicted[(w1, w2)][paraphrase.replace('[w1]', w1).replace('[w2]', w2)] = score

    test_predicted = { (w1, w2) : sorted(curr_paraphrases.items(), key=lambda x: x[1], reverse=True)
                       for (w1, w2), curr_paraphrases in test_predicted.items()}

    logger.info('Evaluation:')
    isomorphic_score, nonisomorphic_score = evaluate(test_predicted, args.test_gold_file, 'test_baseline_predictions.tsv')
    logger.info('Isomorphic = {:.3f}, non-isomorphic = {:.3f}'.format(isomorphic_score, nonisomorphic_score))


if __name__ == '__main__':
    main()