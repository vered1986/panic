# Command line arguments
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('train_gold_file', help='a tsv file with gold paraphrases and their scores')
ap.add_argument('train_predicted_paraphrases_file', help='a jsonl file with the train predictions and their scores')
ap.add_argument('test_predicted_paraphrases_file', help='a jsonl file with the test predictions and their scores')
args = ap.parse_args()

import logging
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.INFO)

import sys
sys.path.append('../')
sys.path.append('../../score')

import json
import codecs
import subprocess

import numpy as np


def main():
    with codecs.open(args.train_predicted_paraphrases_file, 'r', 'utf-8') as f_in:
        train_predicted_paraphrases = [json.loads(line.strip()) for line in f_in]
    logger.info('Loaded train predicted paraphrases from {}'.format(args.train_predicted_paraphrases_file))

    train_predicted_paraphrases = [ { 'w1' : example['w1'], 'w2' : example['w2'],
                                      'paraphrases' :
                                          [(p.replace('is ', '').replace('are ', '').replace('was ', ''), score)
                                           for (p, score) in sorted(example['paraphrases'], key=lambda x: x[1], reverse=True)]
                                      } for example in train_predicted_paraphrases ]

    with codecs.open(args.test_predicted_paraphrases_file, 'r', 'utf-8') as f_in:
        test_predicted_paraphrases = [json.loads(line.strip()) for line in f_in]
    logger.info('Loaded test predicted paraphrases from {}'.format(args.test_predicted_paraphrases_file))

    test_predicted_paraphrases = [{'w1': example['w1'], 'w2': example['w2'],
                                    'paraphrases':
                                        [(p.replace('is ', '').replace('are ', '').replace('was ', ''), score)
                                         for (p, score) in
                                         sorted(example['paraphrases'], key=lambda x: x[1], reverse=True)]
                                    } for example in test_predicted_paraphrases]

    thresholds = [0, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1]
    scores = []

    for threshold in thresholds:
        curr_train_predicted_paraphrases = [ { 'w1' : example['w1'], 'w2' : example['w2'],
                                               'paraphrases' : [(p, score)
                                                                for p, score in example['paraphrases']
                                                                if example['paraphrases'][0][1] - score < threshold]
                                               }
                                             for example in train_predicted_paraphrases]
        score = evaluate(curr_train_predicted_paraphrases, threshold)
        scores.append(score)
        logger.info('Threshold = {}, score = {:.3f}'.format(threshold, score))

    best_index = np.argmax(scores)
    best_threshold = thresholds[best_index]
    logger.info('Best threshold: {}, score: {}'.format(best_threshold, scores[best_index]))

    test_predicted_paraphrases = [{'w1': example['w1'], 'w2': example['w2'],
                                   'paraphrases' : [(p, score)
                                                    for p, score in example['paraphrases']
                                                    if example['paraphrases'][0][1] - score < best_threshold]
                                   }
                                  for example in test_predicted_paraphrases]

    out_file = args.test_predicted_paraphrases_file.replace('.jsonl', '.tsv')
    logger.info('Saving results to {}'.format(out_file))
    with codecs.open(out_file, 'w', 'utf-8') as f_out:
        for example in test_predicted_paraphrases:
            curr_paraphrases = sorted(example['paraphrases'], key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((example['w1'], example['w2'], paraphrase, str(score))) + '\n')


def evaluate(predictions, threshold):
    """
    Uses the Java scorer class to evaluate the current predictions against the gold standard
    :param predictions: a list of noun-compounds and their ranked predicted paraphrases
    :return: the evaluation score
    """
    # Save the evaluations to a temporary file
    prediction_file = 'temp/train_{}_predictions.tsv'.format(threshold)
    with codecs.open(prediction_file, 'w', 'utf-8') as f_out:
        for example in predictions:
            curr_paraphrases = sorted(example['paraphrases'], key=lambda x: x[1], reverse=True)
            for paraphrase, score in curr_paraphrases:
                f_out.write('\t'.join((example['w1'], example['w2'], paraphrase, str(score))) + '\n')

    # java -classpath bin/ semeval2013.Scorer goldstandard.txt semeval_2013_paraphrases.tsv -verbose -isomorphic=true
    result = subprocess.run(['java', '-classpath', 'bin/', 'semeval2013.Scorer', args.train_gold_file,
                             prediction_file, '-verbose', '-isomorphic=false'], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')

    # Take the last line
    return float(result.split('\n')[-2])


if __name__ == '__main__':
    main()