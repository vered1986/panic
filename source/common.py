import codecs
import logging

import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def load_binary_embeddings(embeddings_file, vocab=None):
    """
    Load binary word embeddings, stored in two files: a numpy binary file (.npy)
    and a vocabulary file (.vocab).
    :param embeddings_file: the embedding files prefix
    :param vocab: a limited vocabulary
    :return: the word vectors and list of words
    """
    wv = np.load(embeddings_file + '.npy')

    with codecs.open(embeddings_file + '.vocab') as f_in:
        words = [line.strip() for line in f_in]

    # Limit the vocabulary
    if vocab is not None:
        words, vectors = zip(*[(word, wv[i, :]) for i, word in enumerate(words) if word in vocab])
        wv = np.vstack(vectors)
        logger.info('Loaded {} words'.format(len(words)))

    return wv, words


def save_binary_embeddings(embeddings_file, wv, words):
    """
    Save binary word embeddings, stored in two files: a numpy binary file (.npy)
    and a vocabulary file (.vocab).
    :param embeddings_file: the out embedding files prefix
    :param wv: the word matrix
    :param words (list): vocabulary
    """
    np.save(embeddings_file, wv)
    with codecs.open('{}.vocab'.format(embeddings_file), 'w', 'utf-8') as f_out:
        for word in words:
            f_out.write(word + '\n')


def most_similar_words(word_embeddings, vector, k):
    """
    Returns the top k most similar words to word, using cosine similarity
    :param word_embeddings: a matrix of word embeddings
    :param vector: the vector
    :param k: the number of similar words
    :return: the k most similar vectors to vector
    """
    # Apply matrix-vector dot product to get the distances of w from all the other vectors
    similarity = np.dot(word_embeddings, vector.T)

    # Get the top k vectors
    indices = (-similarity).argsort()[:k + 1]

    return indices