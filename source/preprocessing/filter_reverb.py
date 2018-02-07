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

import re
import tqdm
import spacy
import codecs

from collections import Counter

nlp = spacy.load('en')

three_alpha_chars = re.compile('[a-z]{3,}')


def main():
    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

    with codecs.open(args.nc_file, 'r', 'utf-8') as f_in:
        ncs = [tuple(line.strip().split('\t')) for line in f_in]
    w1s, w2s = zip(*ncs)
    nc_vocab = set(w1s + w2s)
    ncs = set(ncs)

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
                if w1.lower() in nc_vocab or w2.lower() in nc_vocab:
                    if (w1.lower(), w2.lower()) in ncs or (w2.lower(), w1.lower()) in ncs:
                        f_out.write('\t'.join(extraction).lower() + '\n')
                    elif is_triplet_valid(vocab, extraction, w1_surface, w2_surface, pred_surface):
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


def is_triplet_valid(vocab, extraction, w1_surface, w2_surface, pred_surface):
    w1, pred, w2 = extraction

    # Take single words or two words that start with a determiner
    w1_words, w2_words = w1.split(), w2.split()
    correct_number_of_words = False
    if len(w1_words) == 2 and (w1_words[0] == 'the' or w1_words[1] == 'a'):
        w1 = w1_words[1]
        w1_words = w1_words[:2]
    elif len(w2_words) == 2 and (w2_words[0] == 'the' or w2_words[1] == 'a'):
        w2 = w2_words[1]
        w2_words = w2_words[:2]

    if len(w1_words) == 1 and len(w2_words) == 1:
        correct_number_of_words = True

    if not correct_number_of_words:
        return False

    if w1 == w2:
        return False

    # Remove negated triplets
    if ' not ' in pred_surface or "n't" in pred_surface:
        return False

    # Make sure all arguments and the words in the predicate are in the general vocabulary
    if w1.lower() not in vocab or w2.lower() not in vocab:
        return False

    if any([w.lower() not in vocab for w in pred.split()]):
        return False

    for (w, w_surface) in [(w1.lower(), w1_surface), (w2.lower(), w2_surface)]:
        if three_alpha_chars.match(w) is None:
            return False

        # Remove acronyms
        if w_surface == w_surface.upper():
            return False

        # Remove proper names
        w_token = [t for t in nlp(unicode(w_surface))][0]
        if w_token.is_stop or w_token.pos_ != 'NOUN':
            return False

    return True


if __name__ == '__main__':
    main()