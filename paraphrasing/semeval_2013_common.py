import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../source')

import tqdm
import spacy
import codecs
import random
import functools
import subprocess

random.seed(133)

import numpy as np

from collections import defaultdict
from scipy.spatial.distance import cosine

nlp = spacy.load('en', entity=False, add_vectors=False)

prepositions = ['with', 'at', 'from', 'into', 'during', 'including', 'until', 'against',
               'among', 'throughout', 'despite', 'towards', 'upon', 'concerning', 'of', 'to',
               'in', 'for', 'on', 'by', 'about', 'like', 'through', 'over', 'before', 'between',
               'after', 'since', 'without', 'under', 'within', 'along', 'following', 'across',
               'behind', 'beyond', 'plus', 'except', 'but', 'up', 'out',
               'around', 'down', 'off', 'above', 'near']


def rerank(test_predicted_paraphrases, test_features, ranker, minimum_score):
    """
    Rerank the test predict paraphrases according to the ranker
    :param test_predicted_paraphrases: a dictionary of noun-compound to ranked paraphrases
    :param test_features: the features for re-ranking
    :param ranker: the reranker
    :param minimum_score: minimum paraphrase score to keep.
    :return: a dictionary of noun-compound to (re-)ranked paraphrases
    """
    new_test_predicted_paraphrases = { (w1, w2) : [] for (w1, w2) in test_predicted_paraphrases.keys() }

    for ((w1, w2), curr_paraphrases), curr_paraphrase_features in tqdm.tqdm(zip(
            test_predicted_paraphrases.items(), test_features)):
        pars_and_vectors = zip(curr_paraphrases.items(), curr_paraphrase_features)

        # Sort the paraphrases according to the ranking
        def compare_paraphrases(p1, p2):
            return ranker.predict((p2[1] - p1[1]).reshape(1, -1))

        # Consider both the original score (for the specific noun-compound)
        # and the new rank (which paraphrases are more commonly ranked higher)
        sorted_paraphrases = [(paraphrase, (len(curr_paraphrases) - rank) * float(score))
                              for rank, ((paraphrase, score), feature) in
                              enumerate(sorted(pars_and_vectors,
                                               key=functools.cmp_to_key(compare_paraphrases)))]

        sorted_paraphrases = sorted(sorted_paraphrases, key=lambda x: x[1], reverse=True)

        # Keep only paraphrases with score above threshold. Best score = k * 1 = k,
        new_test_predicted_paraphrases[(w1, w2)] = \
            [(paraphrase, score) for (paraphrase, score) in sorted_paraphrases if score >= minimum_score]

    return new_test_predicted_paraphrases


def evaluate(predictions, gold_file, out_prediction_file):
    """
    Uses the Java scorer class to evaluate the current predictions against the gold standard
    :param predictions: a list of noun-compounds and their ranked predicted paraphrases
    :return: the evaluation score
    """
    # Save the evaluations to a file
    with codecs.open(out_prediction_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in predictions.items():
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((w1, w2, paraphrase, str(score))) + '\n')

    # java -classpath bin/ semeval2013.Scorer goldstandard.txt semeval_2013_paraphrases.tsv -verbose -isomorphic=true
    scores = []
    for isomporphic in ['true', 'false']:
        result = subprocess.run(['java', '-classpath', 'bin/', 'semeval2013.Scorer',
                                 gold_file, out_prediction_file, '-verbose',
                                 '-isomorphic={}'.format(isomporphic)],
                                stdout=subprocess.PIPE)
        # Take the last line
        scores.append(float(result.stdout.decode('utf-8').split('\n')[-2]))

    return scores


def predict_paraphrases(model, noun_compounds, words, word2index, UNK, k, unrelated_threshold):
    """
    Gets the language model and retrieves the best k paraphrases for each noun-compound.
    :param model: the trained language model/
    :param noun_compounds: the noun-compounds.
    :param words: the model vocabulary.
    :param word2index: the word to index dictionary.
    :param UNK: the unknown vector.
    :return: the k best paraphrases for each noun-compound.
    """
    paraphrases = {(w1, w2): defaultdict(float) for (w1, w2) in noun_compounds}

    for (w1, w2) in tqdm.tqdm(noun_compounds):
        w1_index, w2_index = word2index.get(w1, UNK), word2index.get(w2, UNK)

        # Returns the top k predicted paraphrase vectors for (first, second)
        par_vectors = model.predict_paraphrase(w1_index, w2_index, k=k)

        for par_index, _, score in par_vectors:
            for paraphrase in get_paraphrase_text(words, model.index2pred[par_index]):
                if 'of said' not in paraphrase and paraphrase.split()[-1] != 'who':
                    paraphrase = paraphrase.replace('[w1]', w1).replace('[w2]', w2)
                    paraphrases[(w1, w2)][paraphrase] = score

        # Remove "unrelated" paraphrases if found enough related ones with higher scores
        curr_paraphrases = sorted(paraphrases[(w1, w2)].items(), key=lambda x: x[1], reverse=True)

        if len(curr_paraphrases) > 0 and 'is unrelated to' in curr_paraphrases[0][0] \
                and curr_paraphrases[0][1] > unrelated_threshold:
            curr_paraphrases = [(p, score) for p, score in curr_paraphrases if 'unrelated' in p]
        else:
            curr_paraphrases = [(p, score) for p, score in curr_paraphrases if 'unrelated' not in p]

        paraphrases[(w1, w2)] = dict(curr_paraphrases)

    return paraphrases


def get_paraphrase_text(words, par_indices):
    """
    Gets the paraphrase word indices and returns the text
    :param words: the model vocabulary.
    :param par_indices: the word indices.
    :return: the paraphrase text
    """
    paraphrase_words = [words[i] for i in par_indices]
    paraphrase = ' '.join(paraphrase_words)
    yield paraphrase


def generate_features(paraphrases, pos2index, prep2index, model, wv, word2index, UNK):
    """
    Gets a list of ranked paraphrases for each noun-compound and extracts
    features from them
    :param paraphrases: a list of ranked paraphrases for each noun-compound
    :param pos2index: a dictionary of POS tags.
    :param prep2index: a dictionary of prepositions.
    :param model: the trained language model/
    :param word2index: the word to index dictionary.
    :param UNK: the unknown vector.
    :return: the features
    """
    features = []
    for (w1, w2), curr_paraphrases in tqdm.tqdm(paraphrases.items()):
        features.append([extract_paraphrase_features(w1, w2, paraphrase, pos2index, prep2index,
                                             model, wv, word2index, UNK)
                 for paraphrase in curr_paraphrases.keys()])

    scores = [list(curr_paraphrases.values()) for curr_paraphrases in paraphrases.values()]
    return features, scores


def extract_paraphrase_features(w1, w2, paraphrase, pos2index, prep2index,
                                model, wv, word2index, UNK):
    """
    Extract features from a paraphrase: its POS tag sequence,
    and which preposition it contains
    :param pos2index: a dictionary of POS tags.
    :param prep2index: a dictionary of prepositions.
    :param model: the trained language model/
    :param word2index: the word to index dictionary.
    :param UNK: the unknown vector.
    :return: a feature vector
    """
    parse = [t for t in nlp(paraphrase)]
    pattern = ' '.join([t.pos_ if t.lemma_ not in {w1, w2} else t.lemma_ for t in parse]).\
        replace(w1, '[w1]').replace(w2, '[w2]')

    # POS tags
    pos_feature = np.zeros(len(pos2index))
    pos_tags = set(pattern.split())
    pos_tags_indices = set([pos2index.get(pos, -1) for pos in pos_tags])

    for index in pos_tags_indices:
        if index > 0:
            pos_feature[index] = 1

    # Which prepositions
    preposition_feature = np.zeros(len(prep2index))
    prepositions = set([t.lemma_ for t in parse if t.pos_ == 'ADP'])
    preposition_indices = set([prep2index.get(prep, -1) for prep in prepositions])

    for index in preposition_indices:
        if index > 0:
            preposition_feature[index] = 1

    # Length
    additional_features = np.array([1 if pattern.endswith('[w1]') else 0,
                                    len(pattern.split())])

    # How likely is each component given the other two?
    paraphrase_with_args = ' '.join([t.orth_ if t.lemma_ not in {w1, w2} else t.lemma_ for t in parse]). \
        replace(w1, '[w1]').replace(w2, '[w2]').split()

    if '[w1]' in paraphrase_with_args and '[w2]' in paraphrase_with_args:
        w1_index, w2_index = word2index.get(w1, UNK), word2index.get(w2, UNK)
        par_indices = tuple([word2index.get(w, UNK) for w in paraphrase_with_args])
        _, predicted_par_vector, par_score = model.predict_paraphrase(w1_index, w2_index, k=1)[0]
        gold_par_index = model.par2index.get(par_indices, -1)

        similarities = np.array([(1 - cosine(predicted_par_vector, model.paraphrase_matrix[gold_par_index])) *
                                 par_score if gold_par_index > 0 else 0.0])
    else:
        similarities = [0.0]
        logger.warning('No similarities computed for {}'.format(paraphrase))

    feature = np.concatenate([pos_feature, preposition_feature, additional_features, similarities])
    return feature


def compute_paraphrase_vector(w1, w2, paraphrase, model, word2index, UNK):
    """
    Gets a textual paraphrase and returns its vector
    :param paraphrase: the textual paraphrase
    :param model: the trained language model/
    :param word2index: the word to index dictionary.
    :param UNK: the unknown vector.
    :return:
    """
    paraphrase = paraphrase.replace(w1, '[w1]').replace(w2, '[w2]')
    par_indices = tuple([word2index.get(w, UNK) for w in paraphrase.split()])
    return model.__compute_state__(par_indices, -1).npvalue()


def load_gold(train_gold_file):
    """
    Loads the train gold file
    :param train_gold_file:
    :return:
    """
    with codecs.open(train_gold_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]

    train_gold = { (w1, w2) : {} for (w1, w2, paraphrase, score) in lines }
    for w1, w2, paraphrase, score in lines:
        train_gold[(w1, w2)][paraphrase] = float(score)

    return train_gold
