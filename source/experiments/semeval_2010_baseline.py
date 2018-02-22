# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('test_file', help='path to a tsv file with noun-compounds and their paraphrases')
ap.add_argument('google_ngrams_dataset_file', help='path to the google ngrams tsv file with w1, paraphrase, w2, score triplets')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')

import codecs

from collections import defaultdict

EPSILON = 0.000001


def main():
    logger.info('Loading the test data from {}'.format(args.test_file))
    test_paraphrases = defaultdict(list)
    with codecs.open(args.test_file, 'r', 'utf-8') as f_in:
        for line in f_in:
            nc, paraphrase = line.strip().split('\t')
            w1, w2 = nc.split()
            test_paraphrases[(w1, w2)].append(paraphrase)

    noun_compounds = sorted(list(test_paraphrases.keys()))
    logger.info('Loading the Google N-grams dataset from {}'.format(args.google_ngrams_dataset_file))
    paraphrases_with_score = { (w1, w2) : { paraphrase : EPSILON for paraphrase in curr_paraphrases }
                               for (w1, w2), curr_paraphrases in test_paraphrases.items() }
    with codecs.open(args.google_ngrams_dataset_file, 'r', 'utf-8') as f_in:
        for line in f_in:
            w2, paraphrase, w1, score = line.strip().split('\t')
            score = float(score)

            # The paraphrases in the SemEval 2010 task are lemmatized
            paraphrase = paraphrase.replace('is', 'be')

            if (w1, w2) in noun_compounds:
                if paraphrase in paraphrases_with_score[(w1, w2)]:
                    paraphrases_with_score[(w1, w2)][paraphrase] = score

            # Plural
            if (w1, w2 + 's') in noun_compounds:
                if paraphrase in paraphrases_with_score[(w1, w2 + 's')]:
                    paraphrases_with_score[(w1, w2 + 's')][paraphrase] = score

    out_file = args.test_file.replace('.txt', '_baseline_predictions.tsv')
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for (w1, w2) in noun_compounds:
            nc = w1 + ' ' + w2
            curr_paraphrases = [(p, paraphrases_with_score[(w1, w2)][p])
                                for p in test_paraphrases[(w1, w2)]]
            curr_paraphrases = sorted(curr_paraphrases, key=lambda x: x[1], reverse=True)
            for i, (paraphrase, score) in enumerate(curr_paraphrases):
                f_out.write('\t'.join((str(i+1), nc, paraphrase, '{:.7f}'.format(score))) + '\n')


if __name__ == '__main__':
    main()