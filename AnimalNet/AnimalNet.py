import tensorflow as tf
import numpy as np


class AnimalNet():
    def __init__(self, num_classes, dropout_rate=0.,
                 optimizer=tf.train.AdamOptimizer,
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 learning_rate=3e-4,
                 is_training = tf.Variable(True)):
        # initializer = tf.contrib.layers.xavier_initialize,
        # init_scales = None): TODO ???

        # Model hyperparameters
        self.num_classes = num_classes
        self.keep_prob = 1 - dropout_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.regularizer = regularizer(scale=regularization_scale)
        self.is_training = is_training
        # self.initializer = initializer()


    def _load_weights(self, file):
        return np.load(file).item()

    def _get_conv_filter(self, data, name):
        return tf.constant(data[name][0], name="filter")

    def _get_bias(self, data, name):
        return tf.constant(data[name][1], name="biases")

    def _conv_layer(self, bottom, name, net_data, stride=[1, 1, 1, 1], group=1):
        with tf.variable_scope(name):
            filt = self._get_conv_filter(net_data, name)
            conv_biases = self._get_bias(net_data, name)

            if group == 1:
                conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            else:
                input_groups = tf.split(3, group, bottom)
                kernel_groups = tf.split(3, group, filt)
                output_groups = [tf.nn.conv2d(i, k, stride, padding='SAME') for i, k in
                                 zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            # bias = tf.reshape(tf.nn.bias_add(conv, conv_biases), conv.get_shape().as_list())
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def _set_training_op(self, is_training):
        return self.is_training.assign(tf.constant(is_training))


    def inference(self, x):

        net_data = self._load_weights('bvlc_alexnet.npy')

        with tf.variable_scope("AlexNet"):
            # ALEXNET
            conv1 = self._conv_layer(x, "conv1", net_data, stride=[1, 4, 4, 1])
            pool1 = self._max_pool(conv1, "pool1")
            conv2 = self._conv_layer(pool1, "conv2", net_data, group=2)
            pool2 = self._max_pool(conv2, "pool2")
            conv3 = self._conv_layer(pool2, "conv3", net_data)
            conv4 = self._conv_layer(conv3, "conv4", net_data, group=2)
            conv5 = self._conv_layer(conv4, "conv5", net_data, group=2)
            pool5 = self._max_pool(conv5, "pool5")

            with tf.variable_scope("Fully_Connected1"):
                flat_pool5 = tf.contrib.layers.flatten(pool5)
                W = tf.constant(net_data['fc6'][0], name="weights")
                b = tf.constant(net_data['fc6'][1], name="biases")

                fc1 = tf.nn.relu(tf.matmul(flat_pool5, W) + b)

            with tf.variable_scope("Fully_Connected2"):
                W = tf.constant(net_data['fc7'][0], name="weights")
                b = tf.constant(net_data['fc7'][1], name="biases")
                fc2 = tf.nn.relu(tf.matmul(fc1, W) + b)

            with tf.variable_scope("Fully_Connected3_trainable"):
                W = tf.get_variable("weights", shape=[4096, 1024],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[1024],
                                    initializer=tf.constant_initializer(0.1))
                fc3 = tf.nn.relu(tf.matmul(fc2, W) + b)
                fc3 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc3, self.keep_prob), lambda: fc3)

            with tf.variable_scope("Fully_Connected4_trainable"):
                W = tf.get_variable("weights", shape=[1024, 10],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[10],
                                    initializer=tf.constant_initializer(0.1))

                fc4 = tf.matmul(fc3, W) + b
                logits = tf.cond(self.is_training, lambda: tf.nn.dropout(fc4, self.keep_prob), lambda: fc4)

        return logits

    def loss(self, logits, labels):
        with tf.name_scope("Loss"):
            with tf.name_scope("Cross_entropy_loss"):
                function_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels)
                cross_entropy = tf.reduce_mean(function_loss)
                tf.scalar_summary("Cross entropy", cross_entropy)

            with tf.name_scope("Regularization_loss"):
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss = tf.reduce_sum(reg_losses)
                tf.scalar_summary("Regularization loss", reg_loss)

            with tf.name_scope("Total_loss"):
                loss = cross_entropy + reg_loss
                tf.scalar_summary("Total loss", loss)

        return loss

    def accuracy(self, logits, labels):
        with tf.name_scope("Accuracy"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.scalar_summary('Accuracy', accuracy)

        return accuracy