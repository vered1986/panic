import argparse

ap = argparse.ArgumentParser()
ap.add_argument('filtered_triplets_file', help='the reverb file with the extractions for the specific noun-compounds')
ap.add_argument('original_triplets_file', help='the original reverb file')
ap.add_argument('out_triplets_file', help='where to write the new extractions')
ap.add_argument('vocab_file',
                help='the embeddings vocabulary file, will be used that all the words in the triplet are in the vocabulary')
ap.add_argument('--use_lemmas', help='whether to use the lemma form or the surface form of the words',
                action='store_true')
args = ap.parse_args()

import re
import tqdm
import codecs

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

prepositions = set(['with', 'at', 'from', 'into', 'during', 'including', 'until', 'against',
                    'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'of', 'to',
                    'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before', 'between',
                    'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across',
                    'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out',
                    'around', 'down', 'off', 'above', 'near'])
be = set(['be', 'is', 'are'])

three_alpha_chars = re.compile('[a-z]{3,}')

def main():
    with codecs.open(args.vocab_file, 'r', 'utf-8') as f_in:
        vocab = set([line.strip() for line in f_in])

        with codecs.open(args.filtered_triplets_file, 'r', 'utf-8') as f_in:
            extractions = [line.strip().split('\t') for line in f_in]
            good_patterns = set([pattern for (w1, pattern, w2) in extractions if is_pattern_valid(pattern)])

        logger.info('Found {} good patterns'.format(len(good_patterns)))

        with codecs.open(args.original_triplets_file, 'r', 'utf-8') as f_in:
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

                    if extraction[1] in good_patterns and is_arg_valid(vocab, w1_surface) \
                            and is_arg_valid(vocab, w2_surface):
                        f_out.write('\t'.join(extraction).lower() + '\n')


def is_pattern_valid(pattern):
    """
    Checks whether a pattern should be included
    A pattern should be included if it contains a single word (possibly verb) + a preposition
    :param pattern: the pattern to check
    :return: a binary value indicating whether the pattern should be included
    """
    pattern_words = pattern.split()
    if len(pattern_words) < 2:
        return False

    return (len(pattern_words) == 2 and pattern_words[1] in prepositions) or \
           (len(pattern_words) == 3 and pattern_words[0] in be and pattern_words[-1] in prepositions)


def is_arg_valid(vocab, arg):
    """
    Checks whether an argument should be included
    :param vocab: the general vocabulary
    :param arg: the argument to check
    :return: a binary value indicating whether the argument be included
    """
    if arg not in vocab:
        return False

    if three_alpha_chars.match(arg) is None:
        return False

    return True


if __name__ == '__main__':
    main()

