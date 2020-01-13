import tensorflow as tf

class DDQN(object):

    def __init__(self, graph, sess, s_dim, a_dim, lr, policy='lstm', n_hidden=64, n_dense=64):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = lr
        self.policy = policy

        with self.graph.as_default():
            with self.sess.as_default():

                num_of_vars = len(tf.compat.v1.trainable_variables())
                self.inputs, self.outputs = self.create_q_network(n_hidden, n_dense)
                self.actions = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])
                self.q = tf.reduce_sum(tf.multiply(self.outputs, self.actions), axis=1)
                self.network_params = tf.compat.v1.trainable_variables()[num_of_vars:]

                self.target_inputs, self.target_outputs = self.create_q_network(n_hidden, n_dense)
                self.target_actions = tf.compat.v1.placeholder(tf.float32, [None, self.a_dim])
                self.target_q = tf.reduce_sum(tf.multiply(self.target_outputs, self.target_actions), axis=1)
                self.target_network_params = tf.compat.v1.trainable_variables()[(len(self.network_params) + num_of_vars):]

                self.update_target_network_params = [
                    self.target_network_params[i].assign(self.network_params[i])
                    for i in range(len(self.target_network_params))
                ]

                self.predicted_q_value = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.weights = tf.compat.v1.placeholder(tf.float32, [None, 1])
                self.abs_errors = tf.abs(self.predicted_q_value - self.q)
                self.loss = tf.reduce_mean(self.weights * tf.square(self.predicted_q_value - self.q))
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                q_summary = tf.compat.v1.summary.scalar("DQN/Q-value", tf.reduce_mean(self.predicted_q_value))
                loss_summary = tf.compat.v1.summary.scalar("DQN/Loss", self.loss)
                self.summary = tf.compat.v1.summary.merge([q_summary, loss_summary])

                self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_q_network(self, n_hidden, n_dense):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.s_dim[0], self.s_dim[1]])
        if self.policy == 'dense':
            inputs_reshaped = tf.keras.layers.Flatten()(inputs)
            hidden = tf.keras.layers.Dense(units=n_hidden, activation=tf.nn.relu)(inputs_reshaped)
        elif self.policy == 'conv':
            conv = tf.keras.layers.Conv1D(filters=n_hidden, kernel_size=4, strides=1, activation='relu')(inputs)
            hidden = tf.keras.layers.Flatten()(conv)
        elif self.policy == 'lstm':
            lstm_cell = tf.keras.layers.LSTMCell(units=n_hidden)
            rnn_output = tf.keras.layers.RNN(lstm_cell)(inputs)
            hidden = tf.keras.layers.Flatten()(rnn_output)
        value_dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        value = tf.keras.layers.Dense(units=1, activation=None)(value_dense)
        advantage_dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        advantages = tf.keras.layers.Dense(units=self.a_dim, activation=None)(advantage_dense)
        outputs = value + tf.subtract(advantages, tf.reduce_mean(advantages, axis=1, keepdims=True))

        return inputs, outputs

    def train(self, inputs, actions, predicted_q_value, weights):
        with self.graph.as_default():
            return self.sess.run([self.outputs, self.abs_errors, self.optimize, self.summary], feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.predicted_q_value: predicted_q_value,
                self.weights: weights
            })

    def predict(self, inputs):
        with self.graph.as_default():
            return self.sess.run(self.outputs, feed_dict={
                self.inputs: inputs,
            })

    def predict_target(self, inputs):
        with self.graph.as_default():
            return self.sess.run(self.target_outputs, feed_dict={
                self.target_inputs: inputs,
            })

    def update_target_network(self):
        with self.graph.as_default():
            self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars