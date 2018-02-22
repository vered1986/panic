import argparse
ap = argparse.ArgumentParser()
ap.add_argument('in_triplets_file', help='the input triplets file')
ap.add_argument('--min_paraphrase_frequency', help='the minimum frequency for including a paraphrase', type=int, default=0)
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import tqdm
import spacy
import codecs

import numpy as np

from collections import defaultdict, Counter
from spacy.symbols import ADJ, NUM, PUNCT, SPACE, DET, NOUN, PROPN


good_determiners = set(['that', 'which'])
tags_to_remove = set([ADJ, NUM, PUNCT, SPACE])
be_inflections = set(['was', 'were', 'are'])
modals = set(['may', 'will', 'would'])
nlp = spacy.load('en', entity=False, add_vectors=False)


def main():
    logger.info('Filtering...')
    extractions = defaultdict(lambda: defaultdict(int))
    with codecs.open(args.in_triplets_file, 'r', 'utf-8') as f_in:
        for line in tqdm.tqdm(f_in):
            w1, paraphrase, w2, count = line.strip().split('\t')
            count = float(count)
            new_paraphrase = normalize(paraphrase)
            if new_paraphrase is not None:
                extractions[(w1, w2)][new_paraphrase] += count

    # Filter paraphrases by frequency
    paraphrases_to_include = set([paraphrase for paraphrase, count in
                                  Counter([p for p_dict in extractions.values()
                                           for p in dict(p_dict).keys()]).items()
                                  if count >= args.min_paraphrase_frequency])

    output = args.in_triplets_file + '_filtered'
    logger.info('Writing output to {}'.format(output))
    with codecs.open(output, 'w', 'utf-8', buffering=0) as f_out:
        for (w1, w2), paraphrases in extractions.items():
            # Split to different paraphrase lengths and spread a probability of 1.0 to each length
            paraphrase_by_length = defaultdict(list)
            [paraphrase_by_length[len(paraphrase.split())].append((paraphrase, weight))
             for paraphrase, weight in paraphrases.items()
             if paraphrase in paraphrases_to_include]

            for length, curr_paraphrases in paraphrase_by_length.items():
                ps, weights = zip(*curr_paraphrases)
                weights = list(np.array(weights) / np.sum(weights))
                for paraphrase, weight in zip(ps, weights):
                    f_out.write('\t'.join((w1, paraphrase, w2, str(weight))) + '\n')


def normalize(paraphrase):
    """
    Normalize paraphrases by removing determiners, adjectives and adverbs
    :param paraphrase: the paraphrase to edit
    :return: the paraphrase with no determiners, adjectives and adverbs
    """
    paraphrase = paraphrase.replace('[w1]', 'something').replace('[w2]', 'Thing') # For better parsing
    paraphrase_tokens = [t for t in nlp(paraphrase)]

    # Remove the NOUN/ADJ/DET in: [w2] ADP NOUN/ADJ/DET [w1]
    if paraphrase_tokens[-1].orth_ == 'something' and \
            (paraphrase_tokens[-2].pos in set([NOUN, ADJ, DET]) or paraphrase_tokens[-2].orth_ == 'her'):
        paraphrase_tokens = paraphrase_tokens[:-2] + [paraphrase_tokens[-1]]

    # Remove adjectives and adverbs, and specific determiners (but not which, that)
    paraphrase_tokens = [t for t in paraphrase_tokens
                      if t.pos not in tags_to_remove and \
                         t.orth_ not in modals and \
                      (t.pos != DET or t.orth_ in good_determiners)]

    # Named entities (e.g. "price of rice in India")
    if any([t.pos == PROPN for t in paraphrase_tokens]):
        return None

    # Replace past tense be verbs
    paraphrase_words = [t.orth_ if t.orth_ not in be_inflections else 'is'
                     for t in paraphrase_tokens]

    # Last word is "who"
    if paraphrase_words[-1] == 'who':
        paraphrase_words = paraphrase_words[:-1]

    paraphrase = ' '.join(paraphrase_words).replace('something', '[w1]').replace('Thing', '[w2]')

    # Out-of-context
    if paraphrase == '[w2] is [w1]' or paraphrase == '[w2] as [w1]':
        return None

    if len(paraphrase) >= 2 and '[w2] [w1]' not in paraphrase:
        return paraphrase

    return None


if __name__ == '__main__':
    main()