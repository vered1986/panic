import argparse
ap = argparse.ArgumentParser()
ap.add_argument('in_triplets_file', help='the reverb file')
ap.add_argument('out_triplets_file', help='where to filtered reverb file')
args = ap.parse_args()

import codecs

from collections import defaultdict
from google_ngram_downloader import readline_google_store


def main():
    triplets_by_len = defaultdict(list)

    with codecs.open(args.in_triplets_file, 'r', 'utf-8') as f_in:
        for line in f_in:
            w1, pred, w2 = line.strip().split('\t')
            triplets_by_len[len(pred.split()) + 2].append((w1, pred, w2))

    frequencies = { (w1, pred, w2) : 1 for length, instances in triplets_by_len.items()
                    for (w1, pred, w2) in instances }

    for length in range(3, 6):
        ngrams = set([' '.join((w1, pred, w2)) for (w1, pred, w2) in triplets_by_len[length]])

        if len(ngrams) > 0:
            curr_frequencies = get_ngrams_frequency(ngrams, length)

            for ngram, freq in curr_frequencies.items():
                words = ngram.split()
                w1, pred, w2 = words[0], words[1:-1], words[-1]
                frequencies[(w1, pred, w2)] = max(1, freq)

    with codecs.open(args.out_triplets_file, 'w', 'utf-8') as f_out:
        for (w1, pred, w2), freq in frequencies.items():
            f_out.write('\t'.join((w1, pred, w2, str(freq))) + '\n')


def get_ngrams_frequency(ngrams, length):
    """
    Get n-gram frequency from Google n-grams
    :param ngram: the string n-gram
    :return: its frequency in Google n-grams
    """
    count = defaultdict(int)
    fname, url, records = next(readline_google_store(ngram_len=length))

    try:
        record = next(records)

        while record.ngram not in ngrams:
            record = next(records)

        while record.ngram in ngrams:
            count[record.ngram] += record.match_count
            record = next(records)

    except StopIteration:
        pass

    return count


if __name__ == '__main__':
    main()
