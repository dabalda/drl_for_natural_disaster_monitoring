import tensorflow as tf
from stable_baselines.deepq.policies import DQNPolicy
from stable_baselines.common.policies import nature_cnn, register_policy
import numpy as np


class StatePlusImagePolicy(DQNPolicy):
    """
    Policy object that implements a DQN policy, using a feed forward neural network with a separate convolutional
    network for an image.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn",
                 obs_phs=None, layer_norm=False, dueling=False, act_fun=tf.nn.relu, **kwargs):
        super(StatePlusImagePolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps,
                                                n_batch, dueling=dueling, reuse=reuse,
                                                scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)

        if layers is None:
            conv_layers = [500, 100]
            dense_layers = [100, 100, 100, 100, 100]
            common_layers = [200, 200]

        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("action_value"):
                extracted_features = tf.layers.flatten(self.processed_obs)
                image1 = extracted_features[:, : ob_space.dim_image[0] * ob_space.dim_image[1]]
                size_dim1 = ob_space.dim_image[0]
                size_dim2 = ob_space.dim_image[1]
                batch_size = tf.shape(image1)[0]
                image2 = tf.reshape(tensor=image1, shape=[batch_size, size_dim1, size_dim2])
                image3 = tf.expand_dims(image2, 3)
                state = extracted_features[:, ob_space.dim_image[0] * ob_space.dim_image[1]:]

                activ = tf.nn.relu

                # Convolutional input network
                conv1 = tf.contrib.layers.conv2d(inputs=image3, num_outputs=64, kernel_size=[3, 3])
                maxp1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

                conv2 = tf.contrib.layers.conv2d(inputs=maxp1, num_outputs=64, kernel_size=[3, 3])
                maxp2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

                conv3 = tf.contrib.layers.conv2d(inputs=maxp2, num_outputs=64, kernel_size=[3, 3])
                maxp3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

                conv_flat = tf.layers.flatten(maxp3, name='flatten')

                for layer_size in conv_layers:
                    conv_flat = tf.contrib.layers.fully_connected(conv_flat,
                                                               num_outputs=layer_size,
                                                               activation_fn=activ)

                # Dense input network
                for layer_size in dense_layers:
                    state = tf.contrib.layers.fully_connected(state,
                                                              num_outputs=layer_size,
                                                              activation_fn=activ)

                # Common network
                action_out = tf.concat(values=[conv_flat, state], axis=1, name='concat')
                for layer_size in common_layers:
                    action_out = tf.contrib.layers.fully_connected(action_out,
                                                                   num_outputs=layer_size,
                                                                   activation_fn=activ)

                # Output layer
                action_scores = tf.contrib.layers.fully_connected(action_out,
                                                                  num_outputs=self.n_actions,
                                                                  activation_fn=None)

            assert not self.dueling, "Dueling currently not supported"
            q_out = action_scores

        self.q_values = q_out
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=True):
        q_values, actions_proba = self.sess.run([self.q_values, self.policy_proba], {self.obs_ph: obs})
        if deterministic:
            actions = np.argmax(q_values, axis=1)
        else:
            actions = np.zeros((len(obs),), dtype=np.int64)
            for action_idx in range(len(obs)):
                actions[action_idx] = np.random.choice(self.n_actions, p=actions_proba[action_idx])

        return actions, q_values, None

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})


register_policy("StatePlusImagePolicy", StatePlusImagePolicy)