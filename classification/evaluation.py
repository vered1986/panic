import codecs
import logging

from sklearn import metrics

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


def output_predictions(predictions_file, relations, predictions, test_set_keys, test_labels):
    """
    Output the model predictions for the test set
    :param predictions_file: the output file path
    :param relations: the ordered list of relations
    :param predictions: the predicted labels for the test set
    :param test_set: the test set - a list of (w1, w2, relation) instances
    :return:
    """
    with codecs.open(predictions_file, 'w', 'utf-8') as f_out:
        for i, (w1, w2) in enumerate(test_set_keys):
            f_out.write('\t'.join([w1, w2, relations[test_labels[i]], relations[predictions[i]]]) + '\n')


def evaluate(y_test, y_pred, relations, do_full_reoprt=False):
    """
    Evaluate performance of the model on the test set
    :param y_test: the test set labels.
    :param y_pred: the predicted values
    :param do_full_reoprt: whether to print the F1, precision and recall of every class.
    :return: mean F1 over all classes
    """
    if do_full_reoprt:
        full_report(y_test, y_pred, relations)
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_test, y_pred, average='weighted')
    return pre, rec, f1, support


def full_report(y_true, y_pred, relations):
    """
    Print a full report on the classes performance
    :param y_true: the gold-standard labels
    :param y_pred: the predictions
    :return: the report
    """
    logger.info(metrics.classification_report(y_true, y_pred, target_names=relations, digits=3))
