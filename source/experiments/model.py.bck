import os
import math
import json
import tqdm
import codecs
import logging

import dynet as dy
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

NUM_LAYERS = 1
DISPLAY_FREQ = 1000
SPECIAL_WORDS = 2


class Model:
    def __init__(self, wv, index2pred, model_dir=None, n_epochs=10, minibatch_size=100, patience=5,
                 update_embeddings=False):
        """
        Initialize the model
        :param wv: pre-trained word embedding vectors
        :param index2pred: paraphrase to index mapping
        :param model_dir: the directory where to save the model
        :param n_epochs: number of training epochs
        :param minibatch_size: the number of instances in a mini batch (default 10)
        :param patience: how many epochs with no improvement on the loss to wait before stopping
        :param update_embeddings: whether to update the embeddings
        """
        self.wv = wv
        self.index2pred = index2pred
        self.pred2index = {p: i for i, p in enumerate(index2pred)}
        self.curr_epoch = 0
        self.n_epochs = n_epochs
        self.embeddings_dim = wv.shape[1]
        self.minibatch_size = minibatch_size
        self.model_dir = model_dir
        self.patience = patience
        self.update_embeddings = update_embeddings

        if not self.update_embeddings:
            lookup_word = lambda w : dy.nobackprop(dy.lookup(self.model_parameters['word_lookup'], w))
        else:
            lookup_word = lambda w: dy.lookup(self.model_parameters['word_lookup'], w)

        self.lookup = lambda w: dy.lookup(self.model_parameters['arg_lookup'], w) \
            if w < SPECIAL_WORDS else lookup_word(w-SPECIAL_WORDS)

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

        # Create the network
        logger.info('Creating the model...')
        self.__create_computation_graph__()

    def fit(self, train_set, val_set):
        """
        Train the model
        """
        def validation_function():
            """
            Compute the loss on the validation set
            :return: the loss on the validation set
            """
            losses = []
            nminibatches = 0

            for minibatch in [val_set[i:i + self.minibatch_size]
                              for i in range(0, len(val_set), self.minibatch_size)]:
                dy.renew_cg()
                nminibatches += 1
                W1, W2, W_p1, W_p2 = dy.parameter(self.model_parameters['W1']), \
                                     dy.parameter(self.model_parameters['W2']), \
                                     dy.parameter(self.model_parameters['W_p1']), \
                                     dy.parameter(self.model_parameters['W_p2'])
                _, par_batch_loss, w1_batch_loss, w2_batch_loss = self.__compute_batch_loss__(
                    W1, W2, W_p1, W_p2, minibatch)
                losses.append((par_batch_loss, w1_batch_loss, w2_batch_loss))

            w1_batch_loss, w2_batch_loss, par_batch_loss = zip(*losses)
            w1_batch_loss = np.sum(w1_batch_loss) / nminibatches
            w2_batch_loss = np.sum(w2_batch_loss) / nminibatches
            par_batch_loss = np.sum(par_batch_loss) / nminibatches
            batch_loss = (w1_batch_loss + w2_batch_loss + par_batch_loss)
            logger.info('Validation: Loss: [w1={:.3f}, w2={:.3f}, paraphrase={:.3f}, total={:.3f}]'.
                        format(w1_batch_loss, w2_batch_loss, par_batch_loss, batch_loss))
            return batch_loss

        logger.info('Training the model...')
        self.__train__(train_set, validation_function)
        logger.info('Training is done!')

    def save_model(self, output_prefix, paraphrase_matrix=True):
        """
        Save the trained model to a file
        """
        if not os.path.exists(output_prefix):
            os.mkdir(output_prefix)

        save_to = output_prefix + '/model'

        self.model_parameters['W1'].save(save_to, '/W1')
        self.model_parameters['W2'].save(save_to, '/W2', append=True)
        self.model_parameters['W_p1'].save(save_to, '/W_p1', append=True)
        self.model_parameters['W_p2'].save(save_to, '/W_p2', append=True)
        self.builder.param_collection().save(save_to, '/builder', append=True)
        self.model_parameters['arg_lookup'].save(save_to, '/arg_lookup_table', append=True)

        if self.update_embeddings:
            self.model_parameters['word_lookup'].save(save_to, '/lookup_table', append=True)

        with codecs.open(output_prefix + '/index2pred.json', 'w', 'utf-8') as f_out:
            json.dump(self.index2pred, f_out)

        if paraphrase_matrix:
            self.save_paraphrase_matrix(output_prefix + '/paraphrases')

    def save_paraphrase_matrix(self, filename):
        """
        Compute the LSTM for all the paraphrases and save to a matrix
        :param paraphrases: the list of predictes
        :param filename: where to save the matrix (in numpy npy format)
        """
        renew_every = 1000
        vecs = []

        for i, pred in tqdm.tqdm(enumerate(self.index2pred)):
            if i % renew_every == 0:
                dy.renew_cg()
            _, _, par_vec = self.__compute_paraphrase_vector__(pred)
            vecs.append(par_vec.npvalue())

        self.paraphrase_matrix = np.vstack(vecs)
        np.save(filename, self.paraphrase_matrix)
        return self.paraphrase_matrix

    def predict_w1(self, w2, paraphrase, k=10):
        """
        Predict the word w1 given w2 and the paraphrase
        :param w2: the index of w2
        :param paraphrase: a list of word indices in the paraphrase
        :param k: the number of most suited w1s to return
        :param vocab: limited vocabulary to predict from
        :return the possible vectors of w1
        """
        dy.renew_cg()
        W1 = dy.parameter(self.model_parameters['W1'])
        w2_vec = self.lookup(w2)
        w1_vec_in_par, _, _ = self.__compute_paraphrase_vector__(paraphrase)
        distribution = dy.softmax(self.__predict_w1__(W1, w1_vec_in_par, w2_vec)).npvalue()
        best_w1_indices = distribution.argsort()[-k:][::-1]
        return [(w1_index, self.lookup(w1_index).npvalue(), distribution[w1_index]) for w1_index in best_w1_indices]

    def predict_w2(self, w1, paraphrase, k=10):
        """
        Predict the word w2 given w1 and the paraphrase
        :param w1: the index of w1
        :param paraphrase: a list of word indices in the paraphrase
        :param k: the number of most suited w2s to return
        :return the possible vectors of w2
        """
        dy.renew_cg()
        W2 = dy.parameter(self.model_parameters['W2'])
        w1_vec = self.lookup(w1)
        _, w2_vec_in_par, _  = self.__compute_paraphrase_vector__(paraphrase)
        distribution = dy.softmax(self.__predict_w2__(W2, w2_vec_in_par, w1_vec)).npvalue()
        best_w2_indices = distribution.argsort()[-k:][::-1]
        return [(w2_index, self.lookup(w2_index).npvalue(), distribution[w2_index]) for w2_index in best_w2_indices]

    def __compute_paraphrase_vector__(self, paraphrase):
        """
        Computes the paraphrase vector from the LSTM
        :param paraphrase: a list of word indices
        :return: the output vectors: representing w1, w2, and the entire paraphrase
        """
        w1_arg, w2_arg = paraphrase.index(0), paraphrase.index(1)
        par_words = [self.lookup(w) for w in paraphrase]
        outputs = self.builder.transduce(par_words)
        w1_arg_vec, w2_arg_vec, par_vec = outputs[w1_arg], outputs[w2_arg], outputs[-1]
        return w1_arg_vec, w2_arg_vec, par_vec

    def predict_paraphrase(self, w1, w2, k=10):
        """
        Predict the paraphrase given the words w1 and w2
        :param w1: the index of w1
        :param w2: the index of w2
        :param k: the number of paraphrases to return
        :return the vectors of the predicted paraphrase
        """
        dy.renew_cg()
        W_p1, W_p2 = dy.parameter(self.model_parameters['W_p1']), dy.parameter(self.model_parameters['W_p2'])
        w1_vec, w2_vec = self.lookup(w1), self.lookup(w2)
        distribution = dy.softmax(self.__predict_paraphrase__(W_p1, W_p2, w1_vec, w2_vec)).npvalue()
        best_par_indices = distribution.argsort()[-k:][::-1]
        return [(par_index, self.__compute_paraphrase_vector__(self.index2pred[par_index])[-1].npvalue(),
                 distribution[par_index]) for par_index in best_par_indices]

    def __predict_w1__(self, W_w_1, par_vec, w2_vec):
        """
        Predict the word w1 given w2 and the paraphrase
        :param W_w_1: the first matrix for word prediction
        :param w2_vec: the index of w2
        :param par_vec: the paraphrase output vector in the index of the [w1] arg
        :return a vector representing the predicted w1
        """

        return W_w_1 * dy.concatenate([par_vec, w2_vec])

    def __predict_w2__(self, W_w_1, par_vec, w1_vec):
        """
        Predict the word w2 given w1 and the paraphrase
        :param W_w_1: the first matrix for word prediction
        :param w1_vec: the index of w1
        :param par_vec: the paraphrase output vector in the index of the [w2] arg
        :return a vector representing the predicted w2
        """
        return W_w_1 * dy.concatenate([par_vec, w1_vec])

    def __predict_paraphrase__(self, W_p1, W_p2, w1_vec, w2_vec):
        """
        Predict the word w2 given w1 and the paraphrase
        :param W_p: the first matrix for paraphrase prediction
        :param w1_vec: the index of w1
        :param w2_vec: the index of w2
        :return a vector representing the predicted paraphrase
        """
        return W_p2 * dy.tanh(W_p1 * dy.concatenate([w1_vec, w2_vec]))

    def __train__(self, train_set, validation_function):
        """
        Train the model
        :param train_set: tuples of (arg1, paraphrase, arg2)
        :param validation_function: returns the validation set result
        """
        trainer = dy.MomentumSGDTrainer(self.model)
        logger.info('Training with len(train) = {}'.format(len(train_set)))
        best_val_loss = np.infty
        patience_count = 0

        for epoch in range(self.curr_epoch, self.n_epochs + self.curr_epoch):
            total_loss = 0.0
            epoch_indices = np.random.permutation(len(train_set))

            # Split to minibatches
            minibatch_size = max(1, min(self.minibatch_size, len(epoch_indices)))
            nminibatches = max(1, int(math.ceil(len(epoch_indices) / minibatch_size)))

            for minibatch in range(nminibatches):
                dy.renew_cg()
                W1, W2, W_p1, W_p2 = dy.parameter(self.model_parameters['W1']), \
                                     dy.parameter(self.model_parameters['W2']), \
                                     dy.parameter(self.model_parameters['W_p1']), \
                                     dy.parameter(self.model_parameters['W_p2'])

                batch_indices = epoch_indices[minibatch_size * minibatch:minibatch_size * (minibatch + 1)]
                batch_instances = [train_set[i] for i in batch_indices]
                loss, par_batch_loss, w1_batch_loss, w2_batch_loss = self.__compute_batch_loss__(
                    W1, W2, W_p1, W_p2, batch_instances)
                batch_loss = w1_batch_loss + w2_batch_loss + par_batch_loss

                if (minibatch + 1) % DISPLAY_FREQ == 0:
                    logger.info(
                        'Epoch {}/{}, batch {}/{}, Loss: [w1={:.3f}, w2={:.3f}, paraphrase={:.3f}, total={:.3f}]'.
                            format(
                        (epoch + 1), self.n_epochs, (minibatch + 1), nminibatches, w1_batch_loss,
                        w2_batch_loss, par_batch_loss, batch_loss))

                loss.backward()
                trainer.update()
                total_loss += batch_loss

            total_loss /= nminibatches
            logger.info('Epoch {}/{}, Loss: {}'.format((epoch + 1), self.n_epochs, total_loss))

            # Early stopping
            curr_val_loss = validation_function()
            if best_val_loss >= curr_val_loss:
                patience_count = 0
                best_val_loss = curr_val_loss

                # Save the best model
                save_to = self.model_dir + '/{}'.format(epoch + 1)
                logger.info('Saving best model trained so far to {}'.format(save_to))
                self.save_model(save_to)
            else:
                patience_count += 1

            if patience_count == self.patience:
                logger.info('Lost patience, stopping training')
                break

    def __compute_batch_loss__(self, W1, W2, W_p1, W_p2, batch_instances):
        """
        Predict the instances in the current batch and return the losses
        :param W1: the parameter
        :param W2: the parameter
        :param W_p: the parameter
        :param batch_instances:
        :return: loss (expression, total), par_batch_loss, w1_batch_loss, w2_batch_loss (floats)
        """
        w1_losses, w2_losses, par_losses = [], [], []
        total_counts = 0.0

        for w1, paraphrase, w2, count in batch_instances:
            total_counts += count
            w1_vec, w2_vec = self.lookup(w1), self.lookup(w2)
            w1_vec_in_par, w2_vec_in_par, _ = self.__compute_paraphrase_vector__(paraphrase)

            # Predict w1, w2 and the paraphrase from each other
            w1_p = self.__predict_w1__(W1, w1_vec_in_par, w2_vec)
            w2_p = self.__predict_w2__(W2, w2_vec_in_par, w1_vec)
            par_p = self.__predict_paraphrase__(W_p1, W_p2, w1_vec, w2_vec)

            w1_losses.append(dy.pickneglogsoftmax(w1_p, w1) * count)
            w2_losses.append(dy.pickneglogsoftmax(w2_p, w2) * count)
            par_losses.append(dy.pickneglogsoftmax(par_p, self.pred2index[paraphrase]) * count)

        w1_loss, w2_loss, par_loss = dy.esum(w1_losses) / total_counts, \
                                      dy.esum(w2_losses) / total_counts, \
                                      dy.esum(par_losses) / total_counts
        loss = dy.esum([w1_loss, w2_loss, par_loss])
        w1_batch_loss, w2_batch_loss, par_batch_loss = w1_loss.value(), w2_loss.value(), par_loss.value()
        return loss, par_batch_loss, w1_batch_loss, w2_batch_loss

    def __create_computation_graph__(self):
        """
        Initialize the model
        """
        dy.renew_cg()
        self.model = dy.ParameterCollection()

        # LSTM for paraphrase
        self.lstm_out_dim = 2 * self.embeddings_dim
        self.builder = dy.BiRNNBuilder(NUM_LAYERS, self.embeddings_dim, self.lstm_out_dim, self.model, dy.LSTMBuilder)

        self.model_parameters = {}
        self.model_parameters['word_lookup'] = self.model.lookup_parameters_from_numpy(self.wv)
        self.model_parameters['arg_lookup'] = self.model.add_lookup_parameters((SPECIAL_WORDS, self.embeddings_dim))

        # Predict w1 from w2 and the paraphrase
        # input_dim = self.embeddings_dim + self.lstm_out_dim
        input_dim = self.embeddings_dim + self.lstm_out_dim
        output_dim = self.wv.shape[0] # vocabulary size
        self.model_parameters['W1'] = self.model.add_parameters((output_dim, input_dim))
        self.model_parameters['W2'] = self.model.add_parameters((output_dim, input_dim))

        # Add the parameter to predict a paraphrase
        input_dim = 2 * self.embeddings_dim
        output_dim = len(self.pred2index)
        hidden_dim = self.embeddings_dim
        self.model_parameters['W_p1'] = self.model.add_parameters((hidden_dim, input_dim))
        self.model_parameters['W_p2'] = self.model.add_parameters((output_dim, hidden_dim))

    @classmethod
    def load_model(cls, model_file_prefix, wv, update_embeddings=False):
        """
        Load the trained model from a file
        """
        # Load the paraphrase file
        with codecs.open(model_file_prefix + '/index2pred.json', 'r', 'utf-8') as f_in:
            index2pred = json.load(f_in)
            index2pred = [tuple(p) for p in index2pred]

        classifier = Model(wv, index2pred)
        classifier.paraphrase_matrix = np.load(model_file_prefix + '/paraphrases.npy')

        # Load the model
        load_from = model_file_prefix + '/model'
        logger.info('Loading the model from {}...'.format(load_from))

        classifier.model_parameters['W1'].populate(load_from, '/W1')
        classifier.model_parameters['W2'].populate(load_from, '/W2')
        classifier.model_parameters['W_p1'].populate(load_from, '/W_p1')
        classifier.model_parameters['W_p2'].populate(load_from, '/W_p2')
        classifier.builder.param_collection().populate(load_from, '/builder')
        classifier.model_parameters['arg_lookup'].populate(load_from, '/arg_lookup_table')

        if update_embeddings:
            classifier.model_parameters['word_lookup'].populate(load_from, '/lookup_table')

        return classifier
