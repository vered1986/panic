import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--ngram_filename_template', help='the Google ngrams file, tab separated, containing: ngram, count',
                default='eng-all-{}gram-20120701-{}.csv.bz2')
ap.add_argument('--n', help='specific n for ngrams. If None, all 3-5 will be retreived', default=None)
ap.add_argument('--prefix', help='First letter of the words. If None, all letters will be retreived', default=None)
ap.add_argument('out_triplets_file', help='where to save the ngrams')
ap.add_argument('vocab_file', help='the embeddings vocabulary file, will be used that all the words in the triplet are in the vocabulary')
ap.add_argument('nc_file', help='the noun-compounds file, to make sure that at least one argument is from that list')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import os
import bz2
import spacy
import codecs

from collections import defaultdict

conjunction = set(['and', 'or', 'whether', 'if'])
negation = set(['not', "don't", "doesn't", 'nor'])
nlp = spacy.load('en', entity=False, add_vectors=False)


def main():
    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = set([tuple(line.strip().lower().split('\t')) for line in f_in])
    ncs_by_prefix = defaultdict(list)
    [ncs_by_prefix[''.join(w1[:2])].append((w1, w2)) for (w1, w2) in ncs]
    [ncs_by_prefix[''.join(w2[:2])].append((w1, w2)) for (w1, w2) in ncs]

    ns = range(3, 6) if args.n is None else [args.n]
    prefixes = ncs_by_prefix.keys() if args.prefix is None \
        else [prefix for prefix in ncs_by_prefix.keys() if prefix.startswith(args.prefix)]
    ns_str = 'all' if args.n is None else args.n
    prefix_str = 'a-z' if args.prefix is None else args.prefix
    out_file = '{}_{}_{}'.format(args.out_triplets_file, ns_str, prefix_str)

    with codecs.open(out_file, 'w', 'utf-8', buffering=0) as f_out:
        for n in ns:
            for prefix in prefixes:
                curr_ncs = ncs_by_prefix[prefix]
                curr_file = args.ngram_filename_template.format(n, prefix)
                logger.info('Reading file {}, looking for: {}'.format(curr_file, curr_ncs))

                if not os.path.exists(curr_file):
                    logger.warning('File {} does not exist. Continuing.'.format(curr_file))
                    continue

                with bz2.BZ2File(curr_file) as f_in:
                    for line in f_in:
                        try:
                            ngram, count = line.lower().strip().split('\t')
                        except:
                            continue

                        ngram_words = ngram.split()
                        w1, pattern, w2 = ngram_words[0], ' '.join(ngram_words[1:-1]), ngram_words[-1]
                        w1, w2 = [t.lemma_ for t in nlp(unicode(w1))][0], [t.lemma_ for t in nlp(unicode(w2))][0]

                        # Keep everything that connects two words from the list of noun-compounds
                        if ((w1, w2) in curr_ncs or (w2, w1) in curr_ncs) and \
                                is_pattern_valid(vocab, pattern):
                            f_out.write('\t'.join((w1, pattern, w2, count)) + '\n')


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

    pattern_words_set = set(pattern_words)

    # Not a conjunction
    if len(pattern_words_set.intersection(conjunction)) > 0:
        return False

    # Not a negated pattern
    if len(pattern_words_set.intersection(negation)) > 0:
        return False

    return True


if __name__ == '__main__':
    main()