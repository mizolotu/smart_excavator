import tensorflow as tf
from baselines.common.models import get_network_builder


class Model(object):
    def __init__(self, name, nenvs, network='mlp', **network_kwargs):
        self.name = name
        self.network_builder = get_network_builder(network)(**network_kwargs)
        self.nenvs = nenvs

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, nenvs, name='actor', network='mlp', **network_kwargs):
        super().__init__(nenvs=nenvs, name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions

    def __call__(self, obs, reuse=False):
        extra_tensors = {}
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            print(self.network_builder)
            x = self.network_builder(obs)

            if isinstance(x, tuple):
                policy_latent, recurrent_tensors = x
                if recurrent_tensors is not None:
                    policy_latent, recurrent_tensors = self.network_builder(obs, self.nenvs)
                    extra_tensors.update(recurrent_tensors)
            else:
                policy_latent = x
                recurrent_tensors = None

            x = tf.layers.dense(policy_latent, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x

class Critic(Model):
    def __init__(self, nenvs, name='critic', network='mlp', **network_kwargs):
        super().__init__(nenvs=nenvs, name=name, network=network, **network_kwargs)
        self.layer_norm = True

    def __call__(self, obs, action, reuse=False):
        extra_tensors = {}
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            try:
                x = tf.concat([obs, action], axis=-1)  # this assumes observation and action can be concatenated
            except:
                action_ = tf.tile(tf.expand_dims(action, 1), [1, tf.shape(obs)[1], 1])
                x = tf.concat([obs, action_], axis=-1)  # this assumes observation and action can be concatenated
            x_ = self.network_builder(x)
            if isinstance(x_, tuple):
                policy_latent, recurrent_tensors = x_
                if recurrent_tensors is not None:
                    policy_latent, recurrent_tensors = self.network_builder(x, self.nenvs)
                    extra_tensors.update(recurrent_tensors)
            else:
                policy_latent = x_
            x = tf.layers.dense(policy_latent, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name='output')
        return x

    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
