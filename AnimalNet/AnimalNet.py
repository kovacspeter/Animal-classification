import tensorflow as tf
import numpy as np


class AnimalNet_Alex():
    def __init__(self, num_classes, dropout_rate=0.,
                 optimizer=tf.train.AdamOptimizer,
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 learning_rate=1e-4,
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


    def inference(self, x, features_layer="AlexNet/Fully_Connected3/Relu:0"):

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

            flat_pool5 = tf.contrib.layers.flatten(pool5)

            with tf.variable_scope("Fully_Connected1"):
                W = tf.constant(net_data['fc6'][0], name="weights")
                b = tf.constant(net_data['fc6'][1], name="biases")

                fc1 = tf.nn.relu(tf.matmul(flat_pool5, W) + b)

            with tf.variable_scope("Fully_Connected2"):
                W = tf.constant(net_data['fc7'][0], name="weights")
                b = tf.constant(net_data['fc7'][1], name="biases")
                fc2 = tf.nn.relu(tf.matmul(fc1, W) + b)

            with tf.variable_scope("Fully_Connected3"):
                W = tf.constant(net_data['fc8'][0], name="weights")
                b = tf.constant(net_data['fc8'][1], name="biases")
                fc3 = tf.nn.relu(tf.matmul(fc2, W) + b)

            features = tf.get_default_graph().get_tensor_by_name(features_layer)
            # OUR CLASSIFIER
            with tf.variable_scope("Fully_Connected4_trainable"):
                W = tf.get_variable("weights", shape=[features.get_shape()[1], 1024],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[1024],
                                    initializer=tf.constant_initializer(0.1))
                fc3 = tf.nn.relu(tf.matmul(features, W) + b)
                fc3 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc3, self.keep_prob), lambda: fc3)

            with tf.variable_scope("Fully_Connected5_trainable"):
                W = tf.get_variable("weights", shape=[1024, 10],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
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

    def get_confusion_matrix(self, logits, labels, sess, feed_dict):
        from sklearn.metrics import confusion_matrix
        with tf.name_scope("Confusion_matrix"):
            y_pred = tf.argmax(logits, 1)
            y = tf.argmax(labels, 1)

            y_gt, y_p = sess.run([y_pred, y], feed_dict)

            conf_mat = confusion_matrix(y_gt, y_p)

        return conf_mat

    def get_problematic_photos(self, logits, labels, sess, feed_dict, x, imgs_per_class):
        import collections
        y_pred = tf.argmax(logits, 1)
        y_true = tf.argmax(labels, 1)
        incorrect_prediction = tf.not_equal(y_pred, y_true)

        incorrect, logits = sess.run([incorrect_prediction, logits], feed_dict)

        incorrectly_classified = np.argmax(logits[incorrect], 1)
        incorrectly_classified_images = x[incorrect]

        sorted_indices = np.argsort(incorrectly_classified)[::-1]

        incorrectly_classified = incorrectly_classified[sorted_indices]
        incorrectly_classified_images = incorrectly_classified_images[sorted_indices]

        imgs = collections.defaultdict(list)

        for index, img in enumerate(incorrectly_classified_images):
            if len(imgs[incorrectly_classified[index]]) < imgs_per_class:
                imgs[incorrectly_classified[index]].append(img)

        return imgs


class AnimalNet_VGG():

    def __init__(self, num_classes, dropout_rate=0.,
                 optimizer=tf.train.AdamOptimizer,
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 learning_rate=3e-4,
                 is_training=tf.Variable(True)):

        # Model hyperparameters
        self.num_classes = num_classes
        self.keep_prob = 1 - dropout_rate
        self.optimizer = optimizer(learning_rate=learning_rate)
        self.regularizer = regularizer(scale=regularization_scale)
        self.is_training = is_training
        self.VGG_FILE = './vgg16_weights.npz'


    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        return weights, keys

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def _conv_layer(self, bottom, name, net_data, stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            filt = tf.constant(net_data[name + '_W'], name="filter")
            conv_biases = tf.constant(net_data[name + '_b'], name="biases")
            conv = tf.nn.conv2d(bottom, filt, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def _set_training_op(self, is_training):
        return self.is_training.assign(tf.constant(is_training))


    def inference(self, x, features_layer="VGG/pool5:0"):
        """
        Load an existing pretrained VGG-16 model.
        See https://www.cs.toronto.edu/~frossard/post/vgg16/

        Args:
            input:         4D Tensor, Input data

        Returns:
            pool5: 4D Tensor, last pooling layer
        """

        with tf.variable_scope("VGG"):
            vgg_weights, vgg_keys = self.load_weights(self.VGG_FILE)

            conv1_1 = self._conv_layer(x, "conv1_1", vgg_weights)
            conv1_2 = self._conv_layer(conv1_1, "conv1_2", vgg_weights)
            pool1 = self._max_pool(conv1_2, "pool1")
            conv2_1 = self._conv_layer(pool1, "conv2_1", vgg_weights)
            conv2_2 = self._conv_layer(conv2_1, "conv2_2", vgg_weights)
            pool2 = self._max_pool(conv2_2, "pool2")
            conv3_1 = self._conv_layer(pool2, "conv3_1", vgg_weights)
            conv3_2 = self._conv_layer(conv3_1, "conv3_2", vgg_weights)
            conv3_3 = self._conv_layer(conv3_2, "conv3_3", vgg_weights)
            pool3 = self._max_pool(conv3_3, "pool3")
            conv4_1 = self._conv_layer(pool3, "conv4_1", vgg_weights)
            conv4_2 = self._conv_layer(conv4_1, "conv4_2", vgg_weights)
            conv4_3 = self._conv_layer(conv4_2, "conv4_3", vgg_weights)
            pool4 = self._max_pool(conv4_3, "pool4")
            conv5_1 = self._conv_layer(pool4, "conv5_1", vgg_weights)
            conv5_2 = self._conv_layer(conv5_1, "conv5_2", vgg_weights)
            conv5_3 = self._conv_layer(conv5_2, "conv5_3", vgg_weights)
            pool5 = self._max_pool(conv5_3, "pool5")

            features = tf.get_default_graph().get_tensor_by_name(features_layer)
            # OUR CLASSIFIER
            flat_pool5 = tf.contrib.layers.flatten(features)

            with tf.variable_scope("Fully_Connected1_trainable"):
                W = tf.get_variable("weights", shape=[flat_pool5.get_shape()[1], 1024],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[1024],
                                    initializer=tf.constant_initializer(0.1))
                fc1 = tf.nn.relu(tf.matmul(flat_pool5, W) + b)
                fc1 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc1, self.keep_prob), lambda: fc1)

            with tf.variable_scope("Fully_Connected2_trainable"):
                W = tf.get_variable("weights", shape=[1024, 10],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[10],
                                    initializer=tf.constant_initializer(0.1))

                fc2 = tf.matmul(fc1, W) + b
                logits = tf.cond(self.is_training, lambda: tf.nn.dropout(fc2, self.keep_prob), lambda: fc2)


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

    def get_confusion_matrix(self, logits, labels, sess, feed_dict):
        from sklearn.metrics import confusion_matrix
        with tf.name_scope("Confusion_matrix"):
            y_pred = tf.argmax(logits, 1)
            y = tf.argmax(labels, 1)

            y_gt, y_p = sess.run([y_pred, y], feed_dict)

            conf_mat = confusion_matrix(y_gt, y_p)

        return conf_mat

    def get_problematic_photos(self, logits, labels, sess, feed_dict, x, imgs_per_class):
        import collections
        y_pred = tf.argmax(logits, 1)
        y_true = tf.argmax(labels, 1)
        incorrect_prediction = tf.not_equal(y_pred, y_true)

        incorrect, logits = sess.run([incorrect_prediction, logits], feed_dict)

        incorrectly_classified = np.argmax(logits[incorrect], 1)
        incorrectly_classified_images = x[incorrect]

        sorted_indices = np.argsort(incorrectly_classified)[::-1]

        incorrectly_classified = incorrectly_classified[sorted_indices]
        incorrectly_classified_images = incorrectly_classified_images[sorted_indices]

        imgs = collections.defaultdict(list)

        for index, img in enumerate(incorrectly_classified_images):
            if len(imgs[incorrectly_classified[index]]) < imgs_per_class:
                imgs[incorrectly_classified[index]].append(img)

        return imgs