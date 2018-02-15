# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
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


def main():
    logger.info('Loading the noun compounds from {}'.format(args.noun_compounds_file))
    with codecs.open(args.noun_compounds_file, 'r', 'utf-8') as f_in:
        noun_compounds = [tuple(line.strip().split('\t')) for line in f_in]

    logger.info('Loading the Google N-grams dataset from {}'.format(args.google_ngrams_dataset_file))
    paraphrases = { (w1, w2) : {} for (w1, w2) in noun_compounds }
    with codecs.open(args.google_ngrams_dataset_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, paraphrase, w2, score = line.strip().split('\t')
            score = float(score)

            if (w1, w2) in noun_compounds:
                paraphrases[(w1, w2)]['{} {} {}'.format(w1, paraphrase, w2)] = score

            if (w2, w1) in noun_compounds:
                paraphrases[(w2, w1)]['{} {} {}'.format(w1, paraphrase, w2)] = score

    out_file = args.noun_compounds_file.replace('.tsv', '_baseline_predictions.tsv')
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for (w1, w2) in noun_compounds:
            curr_paraphrases = paraphrases.get((w1, w2), {})
            curr_paraphrases = sorted(curr_paraphrases.items(), key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((w1, w2, paraphrase, str(score))) + '\n')


if __name__ == '__main__':
    main()