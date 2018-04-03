# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('paraphrase_model_dir', help='the path to the trained paraphrasing model')
ap.add_argument('word_embeddings_for_model', help='word embeddings to be used for the language model')
ap.add_argument('dataset_file', help='the path to the human judgements')
ap.add_argument('output_file', help='where to store the result')
ap.add_argument('--k', help='the number of paraphrases to retrieve, default = 15', default=15, type=int)
args = ap.parse_args()

# Log
import logging
logging.basicConfig(level=logging.DEBUG, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

import sys
sys.path.append('../')
sys.path.append('../../source')

import tqdm
import codecs

from model.model import Model
from common import load_binary_embeddings


def main():
    # Read the dataset
    logger.info('Reading the dataset from {}'.format(args.dataset_file))

    with codecs.open(args.dataset_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
        dataset = { (w1, w2) : tuple(map(float, (w1_score, w2_score, nc_score)))
                    for (w1, w2, w1_score, w2_score, nc_score) in lines }

    # Load the word embeddings and the model
    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings_for_model))
    wv, words = load_binary_embeddings(args.word_embeddings_for_model)

    logger.info('Loading paraphrasing model from {}...'.format(args.paraphrase_model_dir))
    model = Model.load_model(args.paraphrase_model_dir, wv)

    words = ['[w1]', '[w2]', '[par]'] + words
    w2index = {w: i for i, w in enumerate(words)}
    UNK = w2index['unk']

    # Predict paraphrases for each noun compound in the dataset
    logger.info('Predicting paraphrases...')
    paraphrases = {}

    for (w1, w2) in tqdm.tqdm(dataset.keys()):
        w1_index, w2_index = w2index.get(w1, UNK), w2index.get(w2, UNK)

        # Returns the top k predicted paraphrase vectors for (first, second)
        par_indices, _, scores = zip(*model.predict_paraphrase(w1_index, w2_index, k=args.k))
        par_text = [get_paraphrase_text(words, model.index2pred[par_index]) for par_index in par_indices]
        paraphrases[(w1, w2)] = par_text

    # Write the paraphrases, along with the compositionality scores to a file
    with codecs.open(args.output_file, 'w', 'utf-8') as f_out:
        for (w1, w2), comp_scores in dataset.items():
            f_out.write('\t'.join([w1, w2, '\t'.join(list(map(str, comp_scores)))]) + '\n')
            f_out.write('==============================================================\n')
            f_out.write('\n'.join(paraphrases[(w1, w2)]))
            f_out.write('\n\n')


def get_paraphrase_text(words, par_indices):
    """
    Gets the paraphrase word indices and returns the text
    :param words: the model vocabulary.
    :param par_indices: the word indices.
    :return: the paraphrase text
    """
    paraphrase_words = [words[i] for i in par_indices]
    paraphrase = ' '.join(paraphrase_words)
    return paraphrase


if __name__ == '__main__':
    main()