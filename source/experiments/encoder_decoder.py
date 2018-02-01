import os
import math
import logging

import dynet as dy
import numpy as np

from source.common import most_similar_words

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)

NUM_LAYERS = 1
DISPLAY_FREQ = 1000


class Model:
    def __init__(self, wv, model_dir=None, n_epochs=10, minibatch_size=100, patience=5):
        """
        Initialize the model
        :param wv: pre-trained word embedding vectors
        :param model_dir: the directory where to save the model
        :param n_epochs: number of training epochs
        :param minibatch_size: the number of instances in a mini batch (default 10)
        :param patience: how many epochs with no improvement on the loss to wait before stopping
        """
        self.wv = wv
        self.n_epochs = n_epochs
        self.embeddings_dim = wv.shape[1]
        self.minibatch_size = minibatch_size
        self.model_dir = model_dir
        self.patience = patience

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
        self.model.save(output_prefix + '.model')

    def predict_w1(self, w2, predicate, k=1):
        """
        Predict the word w1 given w2 and the predicate
        :param w2: the index of w2
        :param predicate: a list of word indices in the predicate
        :param k: the number of most suited w1s to return
        :return the possible indices of w1
        """
        dy.renew_cg()
        word_lookup = self.model_parameters['word_lookup']
        w2_vec = dy.lookup(word_lookup, w2, update=False)
        pred_words = [dy.lookup(word_lookup, w, update=False) for w in predicate]

        self.r2l_builder.initial_state().add_inputs([w2_vec] + pred_words[::-1])

        # Generate the next word and compare with w1_vec



        pred_vecs = self.builder.initial_state().transduce(pred_words)
        pred_vec = pred_vecs[0] # left most
        w1_p = (mlp * dy.concatenate([w2_vec, pred_vec])).npvalue()
        return most_similar_words(self.wv, w1_p, k)

    def predict_w2(self, w1, predicate, k=1):
        """
        Predict the word w2 given w1 and the predicate
        :param w1: the index of w1
        :param predicate: a list of word indices in the predicate
        :param k: the number of most suited w1s to return
        :return the possible indices of w2
        """
        dy.renew_cg()
        word_lookup = self.model_parameters['word_lookup']
        mlp = dy.parameter(self.model_parameters['W2'])
        w1_vec = dy.lookup(word_lookup, w1, update=False)
        pred_words = [dy.lookup(word_lookup, w, update=False) for w in predicate]
        pred_vec = self.builder.initial_state().transduce(pred_words)[-1]
        w2_p = (mlp * dy.concatenate([w1_vec, pred_vec])).npvalue()
        return most_similar_words(self.wv, w2_p, k)

    def predict_predicate(self, w1, w2, predicate_matrix, k=1):
        """
        Predict the word w2 given w1 and the predicate
        :param w1: the index of w1
        :param w2: the index of w2
        :param predicate_matrix: the predicate matrix
        :param k: the number of most suited w1s to return
        :return the indices of the predicted predicate in the predicate matrix
        """
        dy.renew_cg()
        word_lookup = self.model_parameters['word_lookup']
        mlp = dy.parameter(self.model_parameters['W3'])
        w1_vec = dy.lookup(word_lookup, w1, update=False)
        w2_vec = dy.lookup(word_lookup, w2, update=False)
        pred_p = (mlp * dy.concatenate([w1_vec, w2_vec])).npvalue()
        return most_similar_words(predicate_matrix, pred_p, k)

    def __train__(self, train_set):
        """
        Train the model
        :param train_set: tuples of (arg1, predicate, arg2)
        """
        trainer = dy.AdamTrainer(self.model)
        logger.info('Training with len(train) = {}'.format(len(train_set)))
        word_lookup = self.model_parameters['word_lookup']
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
                mlp_1, mlp_2, mlp_3 = dy.parameter(self.model_parameters['W1']), \
                                      dy.parameter(self.model_parameters['W2']), \
                                      dy.parameter(self.model_parameters['W3'])

                batch_indices = epoch_indices[minibatch_size * minibatch:minibatch_size * (minibatch + 1)]
                batch_instances = [train_set[i] for i in batch_indices]
                losses = []

                for w1, predicate, w2 in batch_instances:
                    w1_vec = dy.lookup(word_lookup, w1, update=False)
                    w2_vec = dy.lookup(word_lookup, w2, update=False)
                    pred_words = [dy.lookup(word_lookup, w, update=False) for w in predicate]
                    pred_vec = self.builder.initial_state().transduce(pred_words)[-1]

                    # Predict w1, w2 and the predicate from each other
                    w1_p = mlp_1 * dy.concatenate([w2_vec, pred_vec])
                    w2_p = mlp_2 * dy.concatenate([w1_vec, pred_vec])
                    pred_p = mlp_3 * dy.concatenate([w1_vec, w2_vec])

                    losses.extend([dy.squared_distance(w1_vec, w1_p),
                                   dy.squared_distance(w2_vec, w2_p),
                                   dy.squared_distance(pred_vec, pred_p)])

                loss = dy.esum(losses)
                batch_loss = loss.value() / len(batch_instances)
                if (minibatch + 1) % DISPLAY_FREQ == 0:
                    logger.info('Epoch {}/{}, batch {}/{}, loss = {}'.format(
                        (epoch + 1), self.n_epochs, (minibatch + 1), nminibatches, batch_loss))
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

    def __predict_next_word__(self, lstm_hidden_state):
        """
        Gets a hidden state of the LSTM and computes
        :param lstm_hidden_state:
        :return:
        """
        R = dy.parameter(self.R)
        bias = dy.parameter(self.bias)
        y_t = lstm_hidden_state.output()
        return bias + (R * y_t)

    def __create_computation_graph__(self):
        """
        Initialize the model
        """
        dy.renew_cg()
        self.model = dy.ParameterCollection()

        # LSTMs for predicate
        self.l2r_builder = dy.LSTMBuilder(NUM_LAYERS, self.embeddings_dim, self.embeddings_dim, self.model)
        self.r2l_builder = dy.LSTMBuilder(NUM_LAYERS, self.embeddings_dim, self.embeddings_dim, self.model)

        self.model_parameters = {}
        self.model_parameters['word_lookup'] = self.model.lookup_parameters_from_numpy(self.wv)
        self.R = self.model.add_parameters(self.wv.shape)
        self.bias = self.model.add_parameters((self.wv.shape[0]))

    @classmethod
    def load_model(cls, model_file_prefix, wv):
        """
        Load the trained model from a file
        """
        classifier = Model(wv)

        # Load the model
        load_from = model_file_prefix + '.model'
        logger.info('Loading the model from {}...'.format(load_from))
        classifier.model.populate(load_from)

        return classifier


class PredicateEncoderDecoder():
    def __init__(self, model, word_lookup, vocab_size, enc_layers, dec_layers,
                 embedding_dim, enc_hidden_state_dim, dec_hidden_state_dim):
        self.model = model
        self.word_lookup = word_lookup

        self.encoder = dy.LSTMBuilder(enc_layers, embedding_dim, enc_hidden_state_dim, self.model)
        self.decoder = dy.LSTMBuilder(dec_layers, enc_hidden_state_dim, dec_hidden_state_dim, self.model)

        self.output_w = self.model.add_parameters((vocab_size, dec_hidden_state_dim))
        self.output_b = self.model.add_parameters((vocab_size))

    def __embed__(self, predicate):
        return [dy.lookup(self.word_lookup, w, update=False) for w in predicate]

    def __encode__(self, init_state, embedded_predicate):
        return init_state.transduce(embedded_predicate)

    def __get_probs__(self, rnn_output):
        output_w = dy.parameter(self.output_w)
        output_b = dy.parameter(self.output_b)
        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def __get_loss__(self, input_predicate, output_predicate):
        dy.renew_cg()
        probs = self.__decode__(input_predicate)
        loss = [-dy.log(dy.pick(p, output_char)) for p, output_char in zip(probs, output_predicate)]
        loss = dy.esum(loss)
        return loss

    def _predict(self, probs):
        predicted_word = np.argmax(probs.npvalue())
        return predicted_word

    def generate(self, input_predicate):
        dy.renew_cg()
        probs = self.__decode__(input_predicate)
        return [self._predict(p) for p in probs]

    def __decode__(self, input_predicate):
        embedded_predicate = self.__embed__(input_predicate)
        encoded_predicate = self.__encode__(self.encoder.initial_state(), embedded_predicate)[-1]

        rnn_state = self.decoder.initial_state()

        probs = []
        for _ in range(len(input_predicate)):
            rnn_state = rnn_state.add_input(encoded_predicate)
            p = self.__get_probs__(rnn_state.output())
            probs.append(p)
        return probs