# Command line arguments
import argparse
ap = argparse.ArgumentParser()
# ap.add_argument('train_gold_file', help='a tsv file with gold paraphrases and their scores')
ap.add_argument('test_file', help='a tsv file with the test predictions, unranked, without scores')
ap.add_argument('open_ie_lm_model_dir', help='the path to the world knowledge model')
ap.add_argument('word_embeddings', help='word embeddings to be used for world knowledge')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../source')

import tqdm
import codecs

from scipy.spatial.distance import cosine

from open_ie_lm.model import Model
from common import load_binary_embeddings, most_similar_words_with_scores


def main():
    logger.info('Reading word embeddings from {}...'.format(args.word_embeddings))
    wv, words = load_binary_embeddings(args.word_embeddings)
    word2index = {w: i for i, w in enumerate(words)}
    UNK = word2index['unk']

    logger.info('Loading world knowledge model from {}...'.format(args.open_ie_lm_model_dir))
    wk_model = Model.load_model(args.open_ie_lm_model_dir, wv, update_embeddings=False)

    # with codecs.open(args.train_gold_file, 'r', 'utf-8') as f_in:
    #     lines = [line.strip().split('\t') for line in f_in]
    # train_gold = { (w1, w2) : {} for (w1, w2, paraphrase, score) in lines }
    # for w1, w2, paraphrase, score in lines:
    #     train_gold[(w1, w2)][paraphrase] = float(score)
    # logger.info('Loaded train gold paraphrases from {}'.format(args.train_gold_file))

    logger.info('Predicting test set...')
    with codecs.open(args.test_file, 'r', 'utf-8') as f_in:
        lines = [line.strip().split('\t') for line in f_in]
    test_predicted = { tuple(nc.split()) : {} for (nc, paraphrase) in lines }

    for nc, paraphrase in tqdm.tqdm(lines):
        w1, w2 = nc.split()

        # Compute the vectors for the gold data
        w1_index, w2_index = word2index.get(w1, UNK), word2index.get(w2, UNK)
        gold_w1_vec, gold_w2_vec = wv[w1_index, :], wv[w2_index, :]
        par_indices = tuple([word2index.get(w, UNK) for w in paraphrase.split()])
        gold_par_vec = wk_model.__compute_predicate_vector__(par_indices).npvalue()

        # Predict vectors
        pred_w1_vec = wk_model.predict_w1(w2_index, par_indices)
        pred_w2_vec = wk_model.predict_w2(w1_index, par_indices)
        pred_par_vec = wk_model.predict_predicate(w2_index, w1_index)

        distances = [cosine(gold_w1_vec, pred_w1_vec),
                     cosine(gold_w2_vec, pred_w2_vec),
                     cosine(gold_par_vec, pred_par_vec)]

        test_predicted[(w1, w2)][paraphrase] = max(0.0, 1.0 - min(distances))

    out_file = args.test_file.replace('.txt', '_predicted.tsv')
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for (w1, w2), curr_paraphrases in test_predicted.items():
            curr_paraphrases = sorted(curr_paraphrases.items(), key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((w1, w2, paraphrase, str(score))) + '\n')


if __name__ == '__main__':
    main()