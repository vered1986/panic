import argparse
ap = argparse.ArgumentParser()
ap.add_argument('in_triplets_file', help='the reverb file')
ap.add_argument('out_triplets_file', help='where to filtered reverb file')
ap.add_argument('vocab_file', help='the embeddings vocabulary file, will be used that all the words in the triplet are in the vocabulary')
ap.add_argument('nc_file', help='the noun-compounds file, to make sure that at least one argument is from that list')
ap.add_argument('--min_word_occurrences', help='the required number of occurrences for a word', default=10, type=int)
ap.add_argument('--min_pred_occurrences', help='the required number of occurrences for a predicate', default=2, type=int)
ap.add_argument('--use_lemmas', help='whether to use the lemma form or the surface form of the predicate', action='store_true')
args = ap.parse_args()

import tqdm
import codecs

from collections import Counter


def main():
    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = set([tuple(line.strip().split('\t')) for line in f_in])

    with codecs.open(args.in_triplets_file, 'r', 'utf-8') as f_in:
        with codecs.open(args.out_triplets_file, 'w', 'utf-8', buffering=0) as f_out:
            for line in tqdm.tqdm(f_in):
                try:
                    item = line.strip().split('\t')
                    w1, pred, w2 = item[4:7]
                    w1_surface, pred_surface, w2_surface = item[1:4]
                except:
                    continue

                if args.use_lemmas:
                    extraction = w1, pred, w2
                else:
                    extraction = w1_surface, pred_surface, w2_surface

                # Keep everything that connects two words from the list of noun-compounds
                if ((w1.lower(), w2.lower()) in ncs or (w2.lower(), w1.lower()) in ncs) and \
                        is_pattern_valid(vocab, extraction[1]):
                    f_out.write('\t'.join(extraction).lower() + '\n')

    with codecs.open(args.out_triplets_file, 'r', 'utf-8') as f_in:
        extractions = [line.strip().split('\t') for line in f_in]

    # Filter by frequency
    w1s, preds, w2s = zip(*extractions)
    w1s, preds, w2s = Counter(w1s), Counter(preds), Counter(w2s)
    filtered_extractions = [(w1, pred, w2) for (w1, pred, w2) in extractions
                            if w1s[w1] >= args.min_word_occurrences
                            and preds[pred] >= args.min_pred_occurrences
                            and w2s[w2] >= args.min_word_occurrences]

    new_file_extension = '_w{}_p{}.txt'.format(args.min_word_occurrences, args.min_pred_occurrences)
    with codecs.open(args.out_triplets_file.replace('.txt', new_file_extension), 'w', 'utf-8') as f_out:
        for w1, pred, w2 in filtered_extractions:
            f_out.write('\t'.join((w1, pred, w2)) + '\n')


def is_pattern_valid(vocab, pattern):
    """
    Checks whether a pattern should be included
    :param vocab: the general vocabulary (e.g. GloVe vocabulary)
    :param pattern: the pattern to check
    :return: a binary value indicating whether the pattern should be included
    """
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


if __name__ == '__main__':
    main()