import os
import math
import logging

import dynet as dy
import numpy as np

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

NUM_LAYERS = 1
DISPLAY_FREQ = 1000


class Model:
    def __init__(self, wv, model_dir=None, n_epochs=10, minibatch_size=100, patience=5, update_embeddings=False):
        """
        Initialize the model
        :param wv: pre-trained word embedding vectors
        :param model_dir: the directory where to save the model
        :param n_epochs: number of training epochs
        :param minibatch_size: the number of instances in a mini batch (default 10)
        :param patience: how many epochs with no improvement on the loss to wait before stopping
        :param update_embeddings: whether to update the embeddings
        """
        self.wv = wv
        self.n_epochs = n_epochs
        self.embeddings_dim = wv.shape[1]
        self.minibatch_size = minibatch_size
        self.model_dir = model_dir
        self.patience = patience
        self.update_embeddings = update_embeddings

        if not self.update_embeddings:
            self.lookup = lambda w : dy.nobackprop(dy.lookup(self.model_parameters['word_lookup'], w))
        else:
            self.lookup = lambda w: dy.lookup(self.model_parameters['word_lookup'], w)

        if model_dir is not None:
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

        # Create the network
        logger.info('Creating the model...')
        self.__create_computation_graph__()

    def fit(self, train_set):
        """
        Train the model
        """
        logger.info('Training the model...')
        self.__train__(train_set)
        logger.info('Training is done!')

    def save_model(self, output_prefix):
        """
        Save the trained model to a file
        """
        save_to = output_prefix + '.model'

        self.model_parameters['W1_1'].save(save_to, '/W1_1')
        self.model_parameters['W1_2'].save(save_to, '/W1_2', append=True)
        self.model_parameters['W2_1'].save(save_to, '/W2_1', append=True)
        self.model_parameters['W2_2'].save(save_to, '/W2_2', append=True)
        self.model_parameters['W_p_1'].save(save_to, '/W_p_1', append=True)
        self.model_parameters['W_p_2'].save(save_to, '/W_p_2', append=True)
        self.builder.param_collection().save(save_to, '/builder', append=True)

        if self.update_embeddings:
            self.model_parameters['word_lookup'].save(save_to, '/lookup_table', append=True)

    def save_predicate_matrix(self, predicates, filename):
        """
        Compute the LSTM for all the predicates and save to a matrix
        :param predicates: the list of predictes
        :param filename: where to save the matrix (in numpy npy format)
        """
        renew_every = 1000
        vecs = []

        for i, pred in enumerate(predicates):
            if i % renew_every == 0:
                logger.info('Predicates: {}/{}'.format(i, len(predicates)))
                dy.renew_cg()
            vecs.append(self.__compute_predicate_vector__(pred).npvalue())

        matrix = np.vstack(vecs)
        np.save(filename, matrix)
        return matrix

    def predict_w1(self, w2, predicate):
        """
        Predict the word w1 given w2 and the predicate
        :param w2: the index of w2
        :param predicate: a list of word indices in the predicate
        :param k: the number of most suited w1s to return
        :param vocab: limited vocabulary to predict from
        :return the possible indices of w1
        """
        dy.renew_cg()
        W1_1, W1_2 = dy.parameter(self.model_parameters['W1_1']), dy.parameter(self.model_parameters['W1_2'])
        w2_vec = self.lookup(w2)
        pred_vec = self.__compute_predicate_vector__(predicate)
        w1_p = self.__predict_w1__(W1_1, W1_2, pred_vec, w2_vec).npvalue()
        return w1_p

    def predict_w2(self, w1, predicate):
        """
        Predict the word w2 given w1 and the predicate
        :param w1: the index of w1
        :param w2: the index of w2
        :return the vector of w1
        """
        dy.renew_cg()
        W2_1, W2_2 = dy.parameter(self.model_parameters['W2_1']), dy.parameter(self.model_parameters['W2_2'])
        w1_vec = self.lookup(w1)
        pred_vec = self.__compute_predicate_vector__(predicate)
        w2_p = self.__predict_w2__(W2_1, W2_2, pred_vec, w1_vec).npvalue()
        return w2_p

    def __compute_predicate_vector__(self, predicate):
        """
        Computes the predicate vector from the LSTM
        :param predicate: a list of word indices
        :return: the predicate vector
        """
        pred_words = [self.lookup(w) for w in predicate]
        pred_vec = self.builder.transduce(pred_words)[-1]
        return pred_vec

    def predict_predicate(self, w1, w2):
        """
        Predict the predicate given the words w1 and w2
        :param w1: the index of w1
        :param w2: the index of w2
        :return the vector of the predicted predicate
        """
        dy.renew_cg()
        W_p_1, W_p_2 = dy.parameter(self.model_parameters['W_p_1']), dy.parameter(self.model_parameters['W_p_2'])
        w1_vec, w2_vec = self.lookup(w1), self.lookup(w2)
        pred_p = self.__predict_predicate__(W_p_1, W_p_2, w1_vec, w2_vec).npvalue()
        return pred_p

    def __predict_w1__(self, W_w_1, W_w_2, pred_vec, w2_vec):
        """
        Predict the word w1 given w2 and the predicate
        :param W_w_1: the first matrix for word prediction
        :param W_w_2: the second matrix for word prediction
        :param w2_vec: the index of w2
        :param pred_vec: the predicate vector
        :return a vector representing the predicted w1
        """
        return dy.tanh(W_w_2 * (dy.tanh(W_w_1 * dy.concatenate([pred_vec, w2_vec]))))

    def __predict_w2__(self, W_w_1, W_w_2, pred_vec, w1_vec):
        """
        Predict the word w2 given w1 and the predicate
        :param W_w_1: the first matrix for word prediction
        :param W_w_2: the second matrix for word prediction
        :param w1_vec: the index of w1
        :param pred_words: a list of word indices in the predicate
        :return a vector representing the predicted w2
        """
        return dy.tanh(W_w_2 * (dy.tanh(W_w_1 * dy.concatenate([pred_vec, w1_vec]))))

    def __predict_predicate__(self, W_p_1, W_p_2, w1_vec, w2_vec):
        """
        Predict the word w2 given w1 and the predicate
        :param W_p_1: the first matrix for predicate prediction
        :param W_p_2: the second matrix for predicate prediction
        :param w1_vec: the index of w1
        :param w2_vec: the index of w2
        :return a vector representing the predicted predicate
        """
        pred_p_input = dy.concatenate([w2_vec, w1_vec])
        return dy.tanh(W_p_2 * (dy.tanh(W_p_1 * pred_p_input)))

    def __train__(self, train_set):
        """
        Train the model
        :param train_set: tuples of (arg1, predicate, arg2)
        """
        trainer = dy.MomentumSGDTrainer(self.model)
        logger.info('Training with len(train) = {}'.format(len(train_set)))
        prev_loss = np.infty
        patience_count = 0

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            epoch_indices = np.random.permutation(len(train_set))

            # Split to minibatches
            minibatch_size = max(1, min(self.minibatch_size, len(epoch_indices)))
            nminibatches = max(1, int(math.ceil(len(epoch_indices) / minibatch_size)))

            for minibatch in range(nminibatches):
                dy.renew_cg()
                W1_1, W1_2, W2_1, W2_2, W_p_1, W_p_2 = dy.parameter(self.model_parameters['W1_1']), \
                                                       dy.parameter(self.model_parameters['W1_2']), \
                                                       dy.parameter(self.model_parameters['W2_1']), \
                                                       dy.parameter(self.model_parameters['W2_2']), \
                                                       dy.parameter(self.model_parameters['W_p_1']), \
                                                       dy.parameter(self.model_parameters['W_p_2'])

                batch_indices = epoch_indices[minibatch_size * minibatch:minibatch_size * (minibatch + 1)]
                batch_instances = [train_set[i] for i in batch_indices]
                w1_losses, w2_losses, pred_losses = [], [], []

                for w1, predicate, w2 in batch_instances:
                    w1_vec, w2_vec = self.lookup(w1), self.lookup(w2)
                    pred_vec = self.__compute_predicate_vector__(predicate)

                    # Predict w1, w2 and the predicate from each other
                    w1_p = self.__predict_w1__(W1_1, W1_2, pred_vec, w2_vec)
                    w2_p = self.__predict_w2__(W2_1, W2_2, pred_vec, w1_vec)
                    pred_p = self.__predict_predicate__(W_p_1, W_p_2, w1_vec, w2_vec)

                    w1_losses.append(dy.squared_distance(w1_vec, w1_p))
                    w2_losses.append(dy.squared_distance(w2_vec, w2_p))
                    pred_losses.append(dy.squared_distance(pred_vec, pred_p))

                w1_loss, w2_loss, pred_loss = dy.esum(w1_losses), dy.esum(w2_losses), dy.esum(pred_losses)
                loss = dy.esum([w1_loss, w2_loss, pred_loss])
                w1_batch_loss = w1_loss.value() / len(batch_instances)
                w2_batch_loss = w2_loss.value() / len(batch_instances)
                pred_batch_loss = pred_loss.value() / len(batch_instances)
                batch_loss = w1_batch_loss + w2_batch_loss + pred_batch_loss

                if (minibatch + 1) % DISPLAY_FREQ == 0:
                    logger.info(
                        'Epoch {}/{}, batch {}/{}, Loss: [w1={:.3f}, w2={:.3f}, predicate={:.3f}, total={:.3f}]'.
                            format(
                        (epoch + 1), self.n_epochs, (minibatch + 1), nminibatches, w1_batch_loss,
                        w2_batch_loss, pred_batch_loss, batch_loss))

                loss.backward()
                trainer.update()
                total_loss += batch_loss

            total_loss /= nminibatches
            logger.info('Epoch {}/{}, Loss: {}'.format((epoch + 1), self.n_epochs, total_loss))

            # Early stopping
            if prev_loss > total_loss:
                patience_count = 0
                prev_loss = total_loss

                # Save the best model
                save_to = self.model_dir + '/best'
                logger.info('Saving best model trained so far to {}'.format(save_to))
                self.save_model(save_to)
            else:
                patience_count += 1

            if patience_count == self.patience:
                logger.info('Lost patience, stopping training')
                break

    def __create_computation_graph__(self):
        """
        Initialize the model
        """
        dy.renew_cg()
        self.model = dy.ParameterCollection()

        # LSTM for predicate
        self.lstm_out_dim = 2 * self.embeddings_dim
        self.builder = dy.BiRNNBuilder(NUM_LAYERS, self.embeddings_dim, self.lstm_out_dim, self.model, dy.LSTMBuilder)

        self.model_parameters = {}
        self.model_parameters['word_lookup'] = self.model.lookup_parameters_from_numpy(self.wv)

        # Predict w1 from w2 and the predicate
        input_dim = self.embeddings_dim + self.lstm_out_dim
        output_dim = self.embeddings_dim
        hidden_dim = (input_dim + output_dim) // 2
        self.model_parameters['W1_1'] = self.model.add_parameters((hidden_dim, input_dim))
        self.model_parameters['W1_2'] = self.model.add_parameters((output_dim, hidden_dim))
        self.model_parameters['W2_1'] = self.model.add_parameters((hidden_dim, input_dim))
        self.model_parameters['W2_2'] = self.model.add_parameters((output_dim, hidden_dim))

        # Predict the predicate from w1 and w2
        input_dim = 2 * self.embeddings_dim
        output_dim = self.lstm_out_dim
        hidden_dim = (input_dim + output_dim) // 2
        self.model_parameters['W_p_1'] = self.model.add_parameters((hidden_dim, input_dim))
        self.model_parameters['W_p_2'] = self.model.add_parameters((output_dim, hidden_dim))

    @classmethod
    def load_model(cls, model_file_prefix, wv, update_embeddings=False):
        """
        Load the trained model from a file
        """
        classifier = Model(wv)

        # Load the model
        load_from = model_file_prefix + '.model'
        logger.info('Loading the model from {}...'.format(load_from))

        classifier.model_parameters['W1_1'].populate(load_from, '/W1_1')
        classifier.model_parameters['W1_2'].populate(load_from, '/W1_2')
        classifier.model_parameters['W2_1'].populate(load_from, '/W2_1')
        classifier.model_parameters['W2_2'].populate(load_from, '/W2_2')
        classifier.model_parameters['W_p_1'].populate(load_from, '/W_p_1')
        classifier.model_parameters['W_p_2'].populate(load_from, '/W_p_2')
        classifier.builder.param_collection().populate(load_from, '/builder')

        if update_embeddings:
            classifier.model_parameters['word_lookup'].populate(load_from, '/lookup_table')

        return classifier
