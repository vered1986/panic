import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--ngram_filename_template', help='the Google ngrams file, tab separated, containing: ngram, count',
                default='eng-all-{}gram-20120701-{}.csv.bz2')
ap.add_argument('--n', help='specific n for ngrams. If None, all 3-5 will be retreived', default=None)
ap.add_argument('--prefix', help='First letter of the words. If None, all letters will be retreived', default=None)
ap.add_argument('out_triplets_file', help='where to save the ngrams')
ap.add_argument('vocab_file', help='the embeddings vocabulary file, will be used that all the words in the triplet are in the vocabulary')
ap.add_argument('nc_file', help='the noun-compounds file, to make sure that at least one argument is from that list')
ap.add_argument('templates_file', help='the file with the POS tags templates to include')
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

nlp = spacy.load('en', entity=False, add_vectors=False)


def main():
    with codecs.open(args.templates_file, 'r', 'utf-8') as f_in:
        templates = [line.strip() for line in f_in]
        templates = set([t for t in templates if '[w1]' in t and '[w2]' in t
                         and len(t.split()) == int(args.n)])
        logger.info('Loaded {} templates'.format(len(templates)))

    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = set([tuple(line.strip().lower().split('\t')) for line in f_in])
    ncs_by_prefix = defaultdict(list)
    [ncs_by_prefix[''.join(w1[:2])].append((w1, w2)) for (w1, w2) in ncs]
    [ncs_by_prefix[''.join(w2[:2])].append((w1, w2)) for (w1, w2) in ncs]
    nc_vocab = set(list(ncs_by_prefix.keys()))

    ns = range(3, 6) if args.n is None else [args.n]
    prefixes = ncs_by_prefix.keys() if args.prefix is None \
        else [prefix for prefix in ncs_by_prefix.keys() if prefix.startswith(args.prefix)]
    ns_str = 'all' if args.n is None else args.n
    prefix_str = 'a-z' if args.prefix is None else args.prefix
    out_file = '{}_{}_{}'.format(args.out_triplets_file, ns_str, prefix_str)

    with codecs.open(out_file, 'w', 'utf-8', buffering=0) as f_out:
        for n in ns:
            for prefix in sorted(prefixes):
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

                        # Filter by word, to make it faster - we require that at least one of the words in the NC
                        # will appear in its base form
                        words = set(ngram.split())

                        if len(words.intersection(nc_vocab)) == 0:
                            continue

                        # Make sure all the words are in the vocabulary
                        if not words.issubset(vocab):
                            continue

                        # Now parse it and match by lemma
                        parse = nlp(unicode(ngram))
                        lemmas = [t.lemma_ for t in parse]
                        potential_ncs = [(w1, w2) for (w1, w2) in curr_ncs if set([w1, w2]).issubset(set(lemmas))]

                        for w1, w2 in potential_ncs:
                            template = ' '.join([{ w1 : '[w1]', w2 : '[w2]' }.get(t.lemma_, t.pos_) for t in parse])
                            if template in templates:
                                paraphrase = ' '.join([{ '[w1]' : '[w1]', '[w2]' : '[w2]' }.get(pos, t.orth_)
                                                       for t, pos in zip(parse, template.split())])
                                f_out.write('\t'.join((w1, paraphrase, w2, count)) + '\n')


if __name__ == '__main__':
    main()