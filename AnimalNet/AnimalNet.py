import tensorflow as tf
import numpy as np


class AnimalNet():
    def __init__(self, num_classes, dropout_rate=0.,
                 optimizer=tf.train.AdamOptimizer,
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 learning_rate=3e-4,
                 batch_size=10,
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

        # Network operations
        self.sess = tf.Session()
        self.input, self.labels, self.forward_op, self.loss_op, self.accuracy_op = \
            self._build_alexnet(self._load_weights('bvlc_alexnet.npy'))

        # Training operation
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Summary
        self.summary_op = tf.merge_all_summaries()

        # Initialize variables
        init_op = tf.initialize_all_variables()
        self.sess.run(init_op)


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

            # conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            # bias = tf.nn.bias_add(conv, conv_biases)

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


    def set_training(self, is_training):
        self.sess.run(self.is_training.assign(tf.constant(is_training)))

    def _build_alexnet(self, net_data):

        with tf.variable_scope("AlexNet"):
            with tf.variable_scope("Input"):
                input = tf.placeholder(tf.float32, [None, 227, 227, 3], name='input')
                labels = tf.placeholder(tf.float32, shape=(None, 10), name='labels')

                # with tf.variable_scope("Conv1_pool1"):
                #     W = tf.get_variable("weights", shape=[11, 11, 3, 96],
                #                         initializer=tf.constant_initializer(net_data["conv1"][0]),
                #                         regularizer=self.regularizer,
                #                         trainable=False)
                #     b = tf.get_variable("biases", shape=[96],
                #                         initializer=tf.constant_initializer(net_data["conv1"][1]),
                #                         trainable=False)
                #     conv = tf.nn.conv2d(input, W, [1, 4, 4, 1], padding='SAME')
                #     bias = tf.nn.bias_add(conv, b)
                #     conv1 = tf.nn.relu(bias)
                #     pool1 = tf.nn.max_pool(conv1,
                #                            ksize=[1, 3, 3, 1],
                #                            strides=[1, 2, 2, 1],
                #                            padding='VALID',
                #                            name='pool1')
                #
                # ALEXNET
                conv1 = self._conv_layer(input, "conv1", net_data, stride=[1, 4, 4, 1])
                pool1 = self._max_pool(conv1, "pool1")
                conv2 = self._conv_layer(pool1, "conv2", net_data, group=2)
                pool2 = self._max_pool(conv2, "pool2")
                conv3 = self._conv_layer(pool2, "conv3", net_data)
                conv4 = self._conv_layer(conv3, "conv4", net_data, group=2)
                conv5 = self._conv_layer(conv4, "conv5", net_data, group=2)
                pool5 = self._max_pool(conv5, "pool5")

            # with tf.variable_scope("Conv2_pool2"):
            #     W = tf.get_variable("weights", shape=[5, 5, 96, 256],
            #                         initializer=tf.constant_initializer(net_data["conv2"][0]),
            #                         regularizer=self.regularizer,
            #                         trainable=False)
            #     b = tf.get_variable("biases", shape=[256],
            #                         initializer=tf.constant_initializer(net_data["conv2"][1]),
            #                         trainable=False)
            #     conv = tf.nn.conv2d(pool1, W, [1, 1, 1, 1], padding='SAME')
            #     bias = tf.nn.bias_add(conv, b)
            #     conv2 = tf.nn.relu(bias)
            #     pool2 = tf.nn.max_pool(conv2,
            #                            ksize=[1, 3, 3, 1],
            #                            strides=[1, 2, 2, 1],
            #                            padding='VALID',
            #                            name='pool2')
            #
            # with tf.variable_scope("Conv3"):
            #     W = tf.get_variable("weights", shape=[3, 3, 256, 384],
            #                         initializer=tf.constant_initializer(net_data["conv3"][0]),
            #                         regularizer=self.regularizer,
            #                         trainable=False)
            #     b = tf.get_variable("biases", shape=[384],
            #                         initializer=tf.constant_initializer(net_data["conv3"][1]),
            #                         trainable=False)
            #     conv = tf.nn.conv2d(pool2, W, [1, 1, 1, 1], padding='SAME')
            #     bias = tf.nn.bias_add(conv, b)
            #     conv3 = tf.nn.relu(bias)
            #
            # with tf.variable_scope("Conv4"):
            #     W = tf.get_variable("weights", shape=[3, 3, 384, 384],
            #                         initializer=tf.constant_initializer(net_data["conv4"][0]),
            #                         regularizer=self.regularizer,
            #                         trainable=False)
            #     b = tf.get_variable("biases", shape=[384],
            #                         initializer=tf.constant_initializer(net_data["conv4"][1]))
            #     conv = tf.nn.conv2d(conv3, W, [1, 1, 1, 1], padding='SAME')
            #     bias = tf.nn.bias_add(conv, b)
            #     conv4 = tf.nn.relu(bias)
            #
            # with tf.variable_scope("Conv5_pool5"):
            #     W = tf.get_variable("weights", shape=[3, 3, 384, 256],
            #                         initializer=tf.constant_initializer(net_data["conv5"][0]),
            #                         regularizer=self.regularizer,
            #                         trainable=False)
            #     b = tf.get_variable("biases", shape=[256],
            #                         initializer=tf.constant_initializer(net_data["conv5"][1]))
            #     conv = tf.nn.conv2d(conv4, W, [1, 1, 1, 1], padding='SAME')
            #     bias = tf.nn.bias_add(conv, b)
            #     conv5 = tf.nn.relu(bias)
            #     pool5 = tf.nn.max_pool(conv5,
            #                            ksize=[1, 3, 3, 1],
            #                            strides=[1, 2, 2, 1],
            #                            padding='VALID',
            #                            name='pool5')
            #
            with tf.variable_scope("Fully_Connected1"):
                flat_pool5 = tf.contrib.layers.flatten(pool5)
                W = tf.get_variable("weights", shape=[9216, 1024],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[1024],
                                    initializer=tf.constant_initializer(0.0))
                fc1 = tf.nn.relu(tf.matmul(flat_pool5, W) + b)
                fc1 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc1, self.keep_prob), lambda: fc1)

            # with tf.variable_scope("Fully-Connected2"):
            #     W = tf.get_variable("weights", shape=[4096, 1024],
            #                         initializer=tf.contrib.layers.xavier_initializer(),
            #                         regularizer=self.regularizer)
            #     b = tf.get_variable("biases", shape=[1024],
            #                              initializer=tf.constant_initializer(0.0))
            #
            #     fc2 = tf.nn.relu(tf.matmul(fc1, W) + b)

            with tf.variable_scope("Fully_Connected3"):
                W = tf.get_variable("weights", shape=[1024, 10],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[10],
                                    initializer=tf.constant_initializer(0.0))

                fc3 = tf.matmul(fc1, W) + b
                fc3 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc3, self.keep_prob), lambda: fc3)

            with tf.variable_scope("Predictions"):
                probs = tf.nn.softmax(fc3)

            with tf.name_scope("Loss"):
                with tf.name_scope("Cross_entropy_loss"):
                    function_loss = tf.nn.softmax_cross_entropy_with_logits(fc3, labels)
                    cross_entropy = tf.reduce_mean(function_loss)
                    tf.scalar_summary("Cross entropy", cross_entropy)

                with tf.name_scope("Regularization_loss"):
                    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                    reg_loss = tf.reduce_sum(reg_losses)
                    tf.scalar_summary("Regularization loss", reg_loss)

                with tf.name_scope("Total_loss"):
                    loss = cross_entropy + reg_loss
                    tf.scalar_summary("Total loss", loss)

            with tf.name_scope("Accuracy"):
                correct_prediction = tf.equal(tf.argmax(fc3, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.scalar_summary('Accuracy', accuracy)

            return input, labels, probs, loss, accuracy

    def predict(self, x):
        feed_dict = {self.input: x}
        return self.sess.run(self.forward_op, feed_dict)

    def train(self, x, y):
        feed_dict = {self.input: x,
                     self.labels: y}
        return self.sess.run(self.train_op, feed_dict)

    def loss(self, x, y):
        feed_dict = {self.input: x,
                     self.labels: y}
        return self.sess.run(self.loss_op, feed_dict)

    def do_summary(self, x, y):
        feed_dict = {self.input: x,
                     self.labels: y}
        return self.sess.run(self.summary_op, feed_dict)

    def accuracy(self, x, y):
        feed_dict = {self.input: x,
                     self.labels: y}
        return self.sess.run(self.accuracy_op, feed_dict)
