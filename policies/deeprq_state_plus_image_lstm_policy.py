import tensorflow as tf
import tensorflow.contrib.slim as slim


class StatePlusImageLstmPolicy():
    def __init__(self, h_size, scope, env, optimizer, dueling=False):
    """
    Policy object that implements a DRQN policy, using a feed forward neural network with a separate convolutional
    network for an image and an LSTM layer.
    """
    
        with tf.variable_scope(scope):
            ob_space = env.observation_space

            self.scalar_input = tf.placeholder(shape=[None, ob_space.shape[0]], dtype=tf.float32)

            conv_layers = [500, 100]
            dense_layers = [100, 100, 100, 100, 100]

            image1 = self.scalar_input[:, : ob_space.dim_image[0] * ob_space.dim_image[1]]
            size_dim1 = ob_space.dim_image[0]
            size_dim2 = ob_space.dim_image[1]
            batch_size = tf.shape(image1)[0]
            image2 = tf.reshape(tensor=image1, shape=[batch_size, size_dim1, size_dim2])
            image3 = tf.expand_dims(image2, 3)
            state = self.scalar_input[:, ob_space.dim_image[0] * ob_space.dim_image[1]:]

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
            action_out1 = tf.concat(values=[conv_flat, state], axis=1, name='concat')
            action_out2 = tf.contrib.layers.fully_connected(action_out1,
                                                            num_outputs=h_size,
                                                            activation_fn=activ)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=h_size, state_is_tuple=True)

            self.lstm_input_size = tf.placeholder(shape=[], dtype=tf.int32)
            self.batch_size = tf.placeholder(shape=[], dtype=tf.int32)
            self.lstm_state_in = lstm_cell.zero_state(self.batch_size, tf.float32)

            flat = tf.layers.flatten(action_out2)
            flat = tf.reshape(flat, (self.batch_size, self.lstm_input_size, h_size))
            self.lstm, self.lstm_state = tf.nn.dynamic_rnn(inputs=flat,
                                                           cell=lstm_cell,
                                                           dtype=tf.float32,
                                                           initial_state=self.lstm_state_in)

            self.lstm = tf.reshape(self.lstm, shape=[-1, h_size])

            if dueling:
                adv, val = tf.split(self.lstm, 2, 1)
                self.advantage = tf.contrib.layers.fully_connected(slim.flatten(adv),
                                                                   num_outputs=env.action_space.n,
                                                                   activation_fn=None)
                self.value = tf.contrib.layers.fully_connected(slim.flatten(val),
                                                               num_outputs=1,
                                                               activation_fn=None)

                self.q_out = self.value + \
                             tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
            else:
                self.q_out = tf.contrib.layers.fully_connected(self.lstm,
                                                               num_outputs=env.action_space.n,
                                                               activation_fn=None)

            self.predict = tf.argmax(self.q_out, 1)

            self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, env.action_space.n, dtype=tf.float32)

            # Loss
            predict_q = tf.reduce_sum(self.q_out * self.actions_onehot, axis=1)

            # Lample & Chatlot 2016
            mask = tf.concat([tf.zeros([self.batch_size, self.lstm_input_size // 2]),
                              tf.ones([self.batch_size, self.lstm_input_size // 2])], 1)
            mask = tf.reshape(mask, [-1])

            self.loss = tf.reduce_mean((predict_q - self.target_q) ** 2 * mask)


            self.gradients = optimizer.compute_gradients(self.loss)

            self.gradients = [(tf.clip_by_value(grad, -10., 10.), var) if grad is not None else (grad, var)
                              for grad, var in self.gradients]

            self.updateModel = optimizer.apply_gradients(self.gradients)