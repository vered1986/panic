import argparse
ap = argparse.ArgumentParser()
ap.add_argument('in_triplets_file', help='the Google ngrams file, tab separated, containing: w1, pattern, w2, count')
ap.add_argument('out_triplets_file', help='where to filtered ngrams file')
ap.add_argument('vocab_file', help='the embeddings vocabulary file, will be used that all the words in the triplet are in the vocabulary')
ap.add_argument('nc_file', help='the noun-compounds file, to make sure that at least one argument is from that list')
args = ap.parse_args()

import tqdm
import codecs

import numpy as np

from collections import defaultdict


determiners = set(['a', 'an', 'the', 'this', 'that', 'his', 'her', 'their', 'our',
                   'your', 'my', 'any', 'every'])


def main():
    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = set([tuple(line.strip().split('\t')) for line in f_in])

    with codecs.open(args.in_triplets_file, 'r', 'utf-8') as f_in:
        with codecs.open(args.out_triplets_file, 'w', 'utf-8', buffering=0) as f_out:
            for line in tqdm.tqdm(f_in):
                try:
                    w1, pattern, w2, count = line.strip().split('\t')
                except:
                    continue

                # Keep everything that connects two words from the list of noun-compounds
                if ((w1, w2) in ncs or (w2, w1) in ncs) and is_pattern_valid(vocab, pattern):
                    f_out.write('\t'.join((w1, pattern, w2, count)) + '\n')

    # Unite patterns that differ by a determiner
    extractions = defaultdict(lambda: defaultdict(int))
    with codecs.open(args.out_triplets_file, 'r', 'utf-8') as f_in:
        for line in f_in:
            w1, pattern, w2, count = line.strip().split('\t')
            count = int(count)
            new_pattern = remove_determiners(pattern)
            if new_pattern is not None:
                extractions[(w1, w2)][new_pattern] += count

    with codecs.open(args.out_triplets_file + '_triplets', 'w', 'utf-8', buffering=0) as f_out:
        for (w1, w2), patterns in extractions.items():
            weights = list(np.array(patterns.values()) / np.sum(patterns.values()))
            for pattern, weight in zip(patterns.keys(), weights):
                f_out.write('\t'.join((w1, pattern, w2, str(weight))) + '\n')


def is_pattern_valid(vocab, pattern):
    """
    Checks whether a pattern should be included
    :param vocab: the general vocabulary (e.g. GloVe vocabulary)
    :param pattern: the pattern to check
    :return: a binary value indicating whether the pattern should be included
    """
    if len(pattern) == 1:
        return False

    pattern_words = pattern.split()
    if len(pattern_words) == 0:
        return False

    # Make sure all the words in the pattern are in the general vocabulary
    if any([w not in vocab for w in pattern_words]):
        return False

    # Not a conjunction
    if pattern_words[0] == 'and' or pattern_words[0] == 'or':
        return False

    # Not a negated pattern
    if ' not ' in pattern_words or "n't" in pattern_words:
        return False

    return True


def remove_determiners(pattern):
    """
    Normalize patterns with determiners
    :param pattern: the pattern to edit
    :return: the pattern with no determiners (in the last word)
    """
    pattern_words = pattern.split()

    if pattern_words[-1] in determiners:
        if len(pattern_words) > 1:
            return ' '.join(pattern_words[:-1])
        else:
            return None

    return pattern


if __name__ == '__main__':
    main()