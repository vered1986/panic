import tqdm
import codecs
import logging

from itertools import count
from collections import defaultdict

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)


class DatasetReader:
    """
    A class for reading and processing a noun compound dataset
    """
    def __init__(self, dataset_file, word2index, label2index=None):
        """
        Read the dataset and convert words to indices
        :param dataset_file: the tab-separated w1, w2, label file
        :param word2index: the embeddings word to index dictionary
        :param label2index: the label dictionary (provide for loading test and validation)
        """
        UNK = word2index['unk']
        noun_compounds = []
        str_noun_compounds = []
        labels = []

        # Get the labels
        created_labels = False
        if label2index is None:
            label2index = defaultdict(lambda c=count(): next(c))
            created_labels = True

        logger.info('Reading {}...'.format(dataset_file))
        with codecs.open(dataset_file, 'r', 'utf-8') as f_in:
            for line in tqdm.tqdm(f_in):
                w1, w2, label = line.strip().split('\t')
                noun_compounds.append((word2index.get(w1.lower(), UNK), word2index.get(w2.lower(), UNK)))
                str_noun_compounds.append((w1, w2))
                labels.append(label2index[label])

        self.noun_compounds = noun_compounds
        self.str_noun_compounds = str_noun_compounds
        self.labels = labels
        self.label2index = label2index
        self.index2label = [label for label, index in sorted(label2index.items(), key=lambda x: x[1])]

        if created_labels:
            logger.info('{} labels: {}'.format(len(self.label2index), self.label2index))
