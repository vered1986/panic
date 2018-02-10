import argparse
ap = argparse.ArgumentParser()
ap.add_argument('in_triplets_file', help='the input triplets file')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import tqdm
import spacy
import codecs

import numpy as np

from collections import defaultdict
from spacy.symbols import ADV, ADJ, NUM, DET, VERB, ADP, PUNCT, PART, SPACE


good_tags = set([VERB, ADP, DET, PART])
good_determiners = set(['that', 'which'])
tags_to_remove = set([ADV, ADJ, NUM, PUNCT, SPACE])
be_inflections = set(['was', 'were', 'are'])
nlp = spacy.load('en', entity=False, add_vectors=False)


def main():
    logger.info('Filtering...')
    extractions = defaultdict(lambda: defaultdict(int))
    with codecs.open(args.in_triplets_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, pattern, w2, count = line.strip().split('\t')
            count = float(count)
            new_pattern = normalize(pattern)
            if new_pattern is not None:
                extractions[(w1, w2)][new_pattern] += count

    output = args.in_triplets_file + '_filtered'
    logger.info('Writing output to {}'.format(output))
    with codecs.open(output, 'w', 'utf-8', buffering=0) as f_out:
        for (w1, w2), patterns in extractions.items():
            # Split to different patttern lengths and spread a probability of 1.0 to each length
            pattern_by_length = defaultdict(list)
            [pattern_by_length[len(pattern.split())].append((pattern, weight))
             for pattern, weight in patterns.items()]

            for length, curr_patterns in pattern_by_length.items():
                ps, weights = zip(*curr_patterns)
                weights = list(np.array(weights) / np.sum(weights))
                for pattern, weight in zip(ps, weights):
                    f_out.write('\t'.join((w1, pattern, w2, str(weight))) + '\n')


def normalize(pattern):
    """
    Normalize patterns by removing determiners, adjectives and adverbs
    :param pattern: the pattern to edit
    :return: the pattern with no determiners, adjectives and adverbs
    """
    pattern_tokens = [t for t in nlp(pattern)]

    # Remove adjectives and adverbs, and specific determiners (but not which, that)
    pattern_tokens = [t for t in pattern_tokens
                      if t.pos not in tags_to_remove and \
                      (t.pos != DET or t.orth_ in good_determiners)]

    # Make sure there is at least one verb or preposition
    if set([t.pos for t in pattern_tokens]).issubset(good_tags):

        # Replace past tense be verbs
        pattern_words = [t.orth_ if t.orth_ not in be_inflections else 'is'
                         for t in pattern_tokens]
        pattern = ' '.join(pattern_words)

        if len(pattern) >= 2:
            return pattern

    return None


if __name__ == '__main__':
    main()