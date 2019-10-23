import tensorflow as tf

# Actor model

class Actor(object):

    def __init__(self, graph, sess, s_dim, a_dim, lr, clip_range, policy='dense', n_hidden=32, n_dense=8):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = lr
        self.clip_range = clip_range
        self.policy = policy

        # main network

        with self.graph.as_default():
            with self.sess.as_default():

                self.inputs, self.outputs, self.dist = self.create_actor_network(n_hidden, n_dense)
                self.actions = tf.compat.v1.placeholder(tf.float32, shape=[None, a_dim])
                self.advantages = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
                self.old_neglogs = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

                self.neglogs = - self.dist.log_prob(self.outputs)
                self.new_neglogs = - self.dist.log_prob(self.actions)
                ratio = tf.exp(self.old_neglogs - self.new_neglogs)
                loss_unclipped = - self.advantages * ratio
                loss_clipped = - self.advantages * tf.clip_by_value(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                logits = self.dist.logits
                probs = tf.nn.softmax(logits)
                self.entropy = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(probs, logits, from_logits=True))
                self.loss = tf.reduce_mean(tf.maximum(loss_unclipped, loss_clipped)) - 0.01 * self.entropy
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
                loss_summary = tf.compat.v1.summary.scalar("Actor/Loss", self.loss)
                ratio_summary = tf.compat.v1.summary.scalar("Actor/Ratio", tf.reduce_mean(ratio))
                self.summary = tf.compat.v1.summary.merge([loss_summary, ratio_summary])

    def create_actor_network(self, n_hidden, n_dense):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.s_dim[0], self.s_dim[1]])
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
        dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        logits = tf.keras.layers.Dense(units=self.a_dim)(dense)
        dist = tf.contrib.distributions.OneHotCategorical(logits=logits)
        outputs = tf.squeeze(dist.sample(1), axis=0)
        return inputs, outputs, dist

    def train(self, inputs, actions, advantages, neglogs):
        with self.graph.as_default():
            return self.sess.run([self.optimize, self.summary], feed_dict={
                self.inputs: inputs,
                self.actions: actions,
                self.advantages: advantages,
                self.old_neglogs: neglogs
            })

    def predict(self, inputs):
        with self.graph.as_default():
            return self.sess.run([self.outputs, self.neglogs], feed_dict={
                self.inputs: inputs
            })

# Critic model

class Critic(object):

    def __init__(self, graph, sess, s_dim, a_dim, lr, clip_range, policy='dense', n_hidden=32, n_dense=8):
        self.graph = graph
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = lr
        self.clip_range = clip_range
        self.policy = policy

        with self.graph.as_default():
            with self.sess.as_default():

                self.inputs, self.values = self.create_critic_network(n_hidden, n_dense)
                self.targets = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
                self.old_values = tf.placeholder(tf.float32, [None, 1])

                values_clipped = self.old_values + tf.clip_by_value(self.values - self.old_values, - self.clip_range, self.clip_range)
                loss_unclipped = tf.square(self.values - self.targets)
                loss_clipped = tf.square(values_clipped - self.targets)
                self.loss = tf.reduce_mean(tf.maximum(loss_unclipped, loss_clipped))
                self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

                target_summary = tf.compat.v1.summary.scalar("Critic/Target", tf.reduce_mean(self.targets))
                loss_summary = tf.compat.v1.summary.scalar("Critic/Loss", self.loss)
                self.summary = tf.compat.v1.summary.merge([target_summary, loss_summary])

    def create_critic_network(self, n_hidden, n_dense):
        inputs = tf.compat.v1.placeholder(tf.float32, shape=[None, self.s_dim[0], self.s_dim[1]])
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
        dense = tf.keras.layers.Dense(units=n_dense, activation=tf.nn.relu)(hidden)
        values = tf.keras.layers.Dense(units=1, activation=None)(dense)
        return inputs, values

    def train(self, inputs, targets, values):
        with self.graph.as_default():
            return self.sess.run([self.optimize, self.summary], feed_dict={
                self.inputs: inputs,
                self.targets: targets,
                self.old_values: values
            })

    def predict(self, inputs):
        with self.graph.as_default():
            return self.sess.run(self.values, feed_dict={
                self.inputs: inputs
            })