import numpy as np
import tensorflow as tf

# Actor model

class Actor(object):

    def __init__(self, graph, sess, s_dim, a_dim, a_bound, lr, tau, batch_size, policy='lstm', activation='sigmoid'):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_bound = a_bound
        self.learning_rate = lr
        self.tau = tau
        self.batch_size = batch_size
        self.policy = policy
        self.activation = activation

        # main network

        with self.graph.as_default():
            with self.sess.as_default():

                num_of_vars = len(tf.compat.v1.trainable_variables())
                self.inputs, self.out, self.scaled_out = self.create_actor_network()
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]

                # pertub network

                self.pertub_inputs, self.pertub_out, self.pertub_scaled_out = self.create_actor_network()
                self.pertub_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_of_vars):]

                # target network

                self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
                self.target_network_params = tf.compat.v1.trainable_variables()[(2 * len(self.network_params) + num_of_vars):]

                self.noise = AdaptiveParamNoiseSpec(initial_stddev=0.00001)
                self.update_pertub_network_params = [
                    self.pertub_network_params[i].assign(
                        self.network_params[i] + tf.random.normal(
                            tf.shape(self.network_params[i]),
                            mean=0.,
                            stddev=self.noise.current_stddev
                        )
                    ) for i in range(len(self.pertub_network_params))
                ]

                self.update_target_network_params = [
                    self.target_network_params[i].assign(
                        tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)
                    ) for i in range(len(self.target_network_params))
                ]

                self.action_gradient = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])
                self.unnormalized_actor_gradients = tf.gradients(
                    self.scaled_out, self.network_params, - self.action_gradient
                )
                self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))
                self.num_trainable_vars = len(self.network_params) + len(self.pertub_network_params) + len(self.target_network_params)

    def create_actor_network(self, n_hidden=16, n_dense=32):

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
        if self.activation == 'sigmoid':
            out = tf.keras.layers.Dense(units=self.a_dim, activation=tf.nn.sigmoid)(dense_a2)
        elif self.activation == 'tanh':
            out = tf.keras.layers.Dense(units=self.a_dim, activation=tf.nn.tanh)(dense_a2)
        else:
            out = tf.keras.layers.Dense(units=self.a_dim)(dense_a2)
        out = tf.reshape(out, shape=[tf.shape(inputs)[0], self.a_dim])
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_pertub(self, inputs):
        return self.sess.run(self.pertub_scaled_out, feed_dict={
            self.pertub_inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def update_pertub_network(self):
        self.sess.run(self.update_pertub_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

# Critic model

class Critic(object):

    def __init__(self, graph, sess, s_dim, a_dim, lr, tau, gamma, policy='lstm'):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = lr
        self.tau = tau
        self.gamma = gamma
        self.policy = policy

        with self.graph.as_default():
            with self.sess.as_default():

                num_of_vars = len(tf.compat.v1.trainable_variables())
                self.inputs, self.action, self.out = self.create_critic_network()
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]
                self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
                self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_of_vars):]
                self.update_target_network_params = [
                    self.target_network_params[i].assign(
                        tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)
                    ) for i in range(len(self.target_network_params))
                ]

                self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.weights = tf.compat.v1.placeholder(tf.float32, [None, 1])

                self.abs_errors = tf.abs(self.predicted_q_value - self.out)
                self.loss = tf.reduce_mean(self.weights * tf.square(self.predicted_q_value - self.out))
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                self.action_grads = tf.gradients(self.out, self.action)

                q_summary = tf.compat.v1.summary.scalar("Q-value", tf.reduce_mean(self.predicted_q_value))
                loss_summary = tf.compat.v1.summary.scalar("Critic loss", self.loss)
                self.summary = tf.compat.v1.summary.merge([q_summary, loss_summary])
                self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_critic_network(self, n_hidden=16, n_dense=32):

        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, *self.s_dim])
        action = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

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
        t1 = tf.keras.layers.Dense(units=n_dense)
        t2 = tf.keras.layers.Dense(units=n_dense)
        dense_c2 = tf.nn.relu(t1(hidden) + t2(action))
        out = tf.keras.layers.Dense(units=1, activation=None)(dense_c2)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value, weights):
        return self.sess.run([self.out, self.abs_errors, self.optimize, self.summary], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.weights: weights
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars

# Parameter noise

class AdaptiveParamNoiseSpec(object):

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adaptation_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient
        self.current_stddev = initial_stddev

    def adapt(self, a1, a2):
        diff = a1 - a2
        mean_diff = np.mean(np.square(diff), axis=0)
        dist = np.sqrt(np.mean(mean_diff))
        if dist > self.desired_action_stddev:
            self.current_stddev /= self.adaptation_coefficient
        else:
            self.current_stddev *= self.adaptation_coefficient
        print(dist, self.current_stddev)
    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

# Action noise

class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, action_dim, action_space_low, action_space_high, mu=0.0, theta=0.15, max_sigma=0.1, min_sigma=0.0, decay_period=10000):
        self.t = 0
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = action_space_low
        self.high = action_space_high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        self.t += 1
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.maximum(np.abs(self.low), np.abs(self.high)) * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
        return action + ou_state