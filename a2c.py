import numpy as np
import tensorflow as tf

# Actor model

class Actor(object):

    def __init__(self, graph, sess, s_dim, a_dim, a_bound, lr, policy='lstm'):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_bound = a_bound
        self.learning_rate = lr
        self.policy = policy

        # main network

        with self.graph.as_default():
            with self.sess.as_default():

                num_of_vars = len(tf.compat.v1.trainable_variables())
                self.inputs, self.outputs, self.norm_dist, self.mu, self.sigma = self.create_actor_network()
                self.deltas = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]
                self.loss = - tf.reduce_mean(tf.log(self.norm_dist.prob(self.outputs) + 1e-5) * self.deltas)
                self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                loss_summary = tf.compat.v1.summary.scalar("Actor loss", self.loss)
                mu_summary = tf.compat.v1.summary.histogram("Action mu", self.mu)
                sigma_summary = tf.compat.v1.summary.histogram("Action sigma", self.sigma)
                self.summary = tf.compat.v1.summary.merge([loss_summary, mu_summary, sigma_summary])

    def create_actor_network(self, n_hidden=32, n_dense=32):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.s_dim])
        if self.policy == 'dense':
            inputs_reshaped = tf.keras.layers.Flatten()(inputs)
            hidden = tf.keras.layers.Dense(units=n_hidden, activation=tf.nn.relu)(inputs_reshaped)
        elif self.policy == 'conv':
            inputs_reshaped = tf.reshape(inputs, shape=[tf.shape(inputs)[0], 1, *self.s_dim])
            conv1 = tf.keras.layers.Conv2D(
                filters=n_hidden, kernel_size=[1, self.s_dim[1]],
                strides=[1, self.s_dim[1]],
                activation='relu',
            )(inputs_reshaped)
            hidden = tf.keras.layers.Flatten()(conv1)
        elif self.policy == 'lstm':
            lstm_cell = tf.keras.layers.LSTMCell(units=n_hidden)
            rnn_output = tf.keras.layers.RNN(lstm_cell)(inputs)
            hidden = tf.keras.layers.Flatten()(rnn_output)
        dense_a2 = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        mu = tf.keras.layers.Dense(units=self.a_dim)(dense_a2)
        mu = tf.nn.softplus(mu) + 1e-5
        sigma = tf.keras.layers.Dense(units=self.a_dim)(dense_a2)
        sigma = tf.nn.softplus(sigma) + 1e-5
        norm_dist = tf.contrib.distributions.Normal(mu, sigma)
        outputs = tf.squeeze(norm_dist.sample(1), axis=0)
        outputs = tf.clip_by_value(outputs, 0, self.action_bound)
        return inputs, outputs, norm_dist, mu, sigma

    def train(self, inputs, outputs, deltas):
        with self.graph.as_default():
            return self.sess.run([self.optimize, self.summary], feed_dict={
                self.inputs: inputs,
                self.outputs: outputs,
                self.deltas: deltas
            })

    def predict(self, inputs):
        with self.graph.as_default():
            return self.sess.run(self.outputs, feed_dict={
                self.inputs: inputs
            })

# Critic model

class Critic(object):

    def __init__(self, graph, sess, s_dim, a_dim, lr, policy='lstm'):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = lr
        self.policy = policy

        with self.graph.as_default():
            with self.sess.as_default():

                num_of_vars = len(tf.compat.v1.trainable_variables())
                self.inputs, self.values = self.create_critic_network()
                self.targets = tf.compat.v1.placeholder(tf.float32, shape=[None, self.a_dim])
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]

                self.loss = tf.reduce_mean(tf.square(self.targets - self.values))
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                loss_summary = tf.compat.v1.summary.scalar("Critic loss", self.loss)
                self.summary = tf.compat.v1.summary.merge([loss_summary])

    def create_critic_network(self, n_hidden=256, n_dense=256):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.s_dim])
        if self.policy == 'dense':
            inputs_reshaped = tf.keras.layers.Flatten()(inputs)
            hidden = tf.keras.layers.Dense(units=n_hidden, activation=tf.nn.relu)(inputs_reshaped)
        elif self.policy == 'conv':
            inputs_reshaped = tf.reshape(inputs, shape=[tf.shape(inputs)[0], 1, *self.s_dim])
            conv1 = tf.keras.layers.Conv2D(
                filters=n_hidden, kernel_size=[1, self.s_dim[1]],
                strides=[1, self.s_dim[1]],
                activation='relu',
            )(inputs_reshaped)
            hidden = tf.keras.layers.Flatten()(conv1)
        elif self.policy == 'lstm':
            lstm_cell = tf.keras.layers.LSTMCell(units=n_hidden)
            rnn_output = tf.keras.layers.RNN(lstm_cell)(inputs)
            hidden = tf.keras.layers.Flatten()(rnn_output)
        dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        values = tf.keras.layers.Dense(units=self.a_dim, activation=None)(dense)
        return inputs, values

    def train(self, inputs, targets):
        with self.graph.as_default():
            return self.sess.run([self.optimize, self.summary], feed_dict={
                self.inputs: inputs,
                self.targets: targets
            })

    def predict(self, inputs):
        with self.graph.as_default():
            return self.sess.run(self.values, feed_dict={
                self.inputs: inputs
            })