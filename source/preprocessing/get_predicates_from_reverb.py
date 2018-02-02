# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('noun_compounds_file', help='path to a tsv file with noun-compounds')
ap.add_argument('reverb_dataset_file', help='path to the reverb tsv file with w1, relation, w2 triplets')
ap.add_argument('out_file', help='where to save the results')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')

import tqdm
import json
import codecs


def main():
    logger.info('Loading the noun compounds from {}'.format(args.noun_compounds_file))
    with codecs.open(args.noun_compounds_file, 'r', 'utf-8') as f_in:
        noun_compounds = set([tuple(line.strip().split('\t')) for line in f_in])

    logger.info('Loading the ReVerb dataset from {}'.format(args.reverb_dataset_file))
    paraphrases = { (w1, w2) : set() for (w1, w2) in noun_compounds }
    with codecs.open(args.reverb_dataset_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, relation, w2 = line.strip().split('\t')

            if (w1, w2) in noun_compounds:
                paraphrases[(w1, w2)].add('{} {} {}'.format(w1, relation, w2))

            if (w2, w1) in noun_compounds:
                paraphrases[(w2, w1)].add('{} {} {}'.format(w1, relation, w2))

    logger.info('Saving results to {}'.format(args.out_file))
    with codecs.open(args.out_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in paraphrases.items():
            f_out.write(json.dumps({
                'w1' : w1,
                'w2' : w2,
                'paraphrases' : list(curr_paraphrases)
            }) + '\n')


if __name__ == '__main__':
    main()