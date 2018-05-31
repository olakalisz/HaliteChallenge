import os
import tensorflow as tf
import numpy as np

from tsmlstarterbot.common import PER_SHIP_FEATURES, PER_SHIP_ACTIONS

# We don't want tensorflow to produce any warnings in the standard output, since the bot communicates
# with the game engine through stdout/stdin.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
tf.logging.set_verbosity(tf.logging.ERROR)


# Normalize planet features within each frame.
# def normalize_input(input_data):

#     # Assert the shape is what we expect
#     shape = input_data.shape
#     assert len(shape) == 3 and shape[1] == PLANET_MAX_NUM and shape[2] == PER_PLANET_FEATURES

#     m = np.expand_dims(input_data.mean(axis=1), axis=1)
#     s = np.expand_dims(input_data.std(axis=1), axis=1)
#     return (input_data - m) / (s + 1e-6)


class NeuralNet(object):
    FIRST_LAYER_SIZE = 1000
    SECOND_LAYER_SIZE = 500
    THIRD_LAYER_SIZE = 100

    def __init__(self, cached_model=None, seed=None):
        self._graph = tf.Graph()

        with self._graph.as_default():
            if seed is not None:
                tf.set_random_seed(seed)
            self._session = tf.Session()
            self._features = tf.placeholder(dtype=tf.float32, name="input_features",
                                            shape=(None, PER_SHIP_FEATURES))

            # target_distribution describes what the bot did in a real game.
            # For instance, if it sent 20% of the ships to the first planet and 15% of the ships to the second planet,
            # then expected_distribution = [0.2, 0.15 ...]
            self._target_move = tf.placeholder(dtype=tf.float32, name="target_move",
                                                       shape=(None, PER_SHIP_ACTIONS))

            # Combine all the planets from all the frames together, so it's easier to share
            # the weights and biases between them in the network.
            # flattened_frames = tf.reshape(self._features, [-1, PER_SHIP_FEATURES])

            # First layer
            net = tf.contrib.layers.fully_connected(self._features, self.FIRST_LAYER_SIZE)
            net = tf.contrib.layers.dropout(net, keep_prob=0.9)

            # Second layer
            net = tf.contrib.layers.fully_connected(net, self.SECOND_LAYER_SIZE)
            net = tf.contrib.layers.dropout(net, keep_prob=0.9)

            # Third layer
            net = tf.contrib.layers.fully_connected(net, self.THIRD_LAYER_SIZE)
            net = tf.contrib.layers.dropout(net, keep_prob=0.9)

            # Final layer
            output = tf.contrib.layers.fully_connected(net, PER_SHIP_ACTIONS, activation_fn=None)

            # Group the planets back in frames.
            # logits = tf.reshape(fourth_layer, [-1, PLANET_MAX_NUM])

            self._prediction = tf.nn.softmax(output)

            self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._prediction, labels=self._target_move))

            self._optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self._loss)
            self._saver = tf.train.Saver()

            if cached_model is None:
                self._session.run(tf.global_variables_initializer())
            else:
                self._saver.restore(self._session, cached_model)

    def fit(self, input_data, expected_output_data):
        """
        Perform one step of training on the training data.

        :return: training loss on the input data
        """
        loss, _ = self._session.run([self._loss, self._optimizer],
                                    feed_dict={self._features: input_data,
                                               self._target_move: expected_output_data})
        return loss

    def predict(self, input_data):
        """
        Given data from 1 frame, predict the move.
        """
        return self._session.run(self._prediction,
                                 feed_dict={self._features: np.array([input_data])})[0]

    def compute_loss(self, input_data, expected_output_data):
        """
        Compute loss on the input data without running any training.

        :return: training loss on the input data
        """
        return self._session.run(self._loss,
                                 feed_dict={self._features: input_data,
                                            self._target_move: expected_output_data})

    def save(self, path):
        """
        Serializes this neural net to given path.
        :param path:
        """
        self._saver.save(self._session, path)

