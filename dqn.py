import tensorflow as tf

class DDQN(object):

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
                self.inputs, self.action, self.out, self.q = self.create_q_network()
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]
                self.target_inputs, self.target_action, self.target_out, self.target_q = self.create_q_network()
                self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_of_vars):]
                self.update_target_network_params = [
                    self.target_network_params[i].assign(
                        tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau)
                    ) for i in range(len(self.target_network_params))
                ]

                self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.weights = tf.compat.v1.placeholder(tf.float32, [None, 1])

                self.abs_errors = tf.abs(self.predicted_q_value - self.q)
                self.loss = tf.reduce_mean(self.weights * tf.square(self.predicted_q_value - self.q))
                self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

                q_summary = tf.compat.v1.summary.scalar("Q-value", tf.reduce_mean(self.predicted_q_value))
                loss_summary = tf.compat.v1.summary.scalar("Loss", self.loss)
                self.summary = tf.compat.v1.summary.merge([q_summary, loss_summary])
                self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_q_network(self, n_hidden=16, n_dense=32):

        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.s_dim[0], self.s_dim[1]])
        action = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])

        if self.policy == 'dense':
            inputs_reshaped = tf.keras.layers.Flatten()(inputs)
            hidden = tf.keras.layers.Dense(units=n_hidden, activation=tf.nn.relu)(inputs_reshaped)
        elif self.policy == 'conv':
            inputs_reshaped = tf.reshape(inputs, shape=[tf.shape(inputs)[0], 1, self.s_dim[0], self.s_dim[1]])
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
        value_dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        value = tf.keras.layers.Dense(units=1, activation=None)(value_dense)
        advantage_dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        advantages = tf.keras.layers.Dense(units=self.a_dim, activation=None)(advantage_dense)
        output = value + tf.subtract(advantages, tf.reduce_mean(advantages, axis=1, keepdims=True))
        q = tf.reduce_sum(tf.multiply(output, action), axis=1)
        return inputs, action, output, q

    def train(self, inputs, action, predicted_q_value, weights):
        return self.sess.run([self.out, self.abs_errors, self.optimize, self.summary], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value,
            self.weights: weights
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars