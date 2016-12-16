import tensorflow as tf
import numpy as np


class AnimalNet_Alex():
    def __init__(self, num_classes, dropout_rate=0.,
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 is_training = tf.Variable(True),
                 refine_after=0
                 ):

        # Model hyperparameters
        self.num_classes = num_classes
        self.keep_prob = 1 - dropout_rate
        self.regularizer = regularizer(scale=regularization_scale)
        self.is_training = is_training
        self.refine_after = refine_after
        self.assign_ops = []

    def _fc_layer(self, bottom, name, data):
        with tf.variable_scope(name):
            W = data[name][0]
            kernel = tf.get_variable('weights', W.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(kernel.assign(W))

            b = data[name][1]
            biases = tf.get_variable('biases', b.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(biases.assign(b))

            return tf.nn.relu(tf.matmul(bottom, W) + b)

    def _conv_layer(self, bottom, name, data, stride=[1, 1, 1, 1], group=1):
        with tf.variable_scope(name):
            W = data[name][0]
            kernel = tf.get_variable('filter', W.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(kernel.assign(W))

            b = data[name][1]
            biases = tf.get_variable('biases', b.shape, initializer=tf.constant_initializer(0.0))
            self.assign_ops.append(biases.assign(b))

            if group == 1:
                conv = tf.nn.conv2d(bottom, kernel, stride, padding='SAME')
            else:
                input_groups = tf.split(3, group, bottom)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [tf.nn.conv2d(i, k, stride, padding='SAME') for i, k in
                                 zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)

            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)
            return relu

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)

    def _set_training_op(self, is_training):
        return self.is_training.assign(tf.constant(is_training))

    def load_weights(self):
        return np.load('bvlc_alexnet.npy').item()

    def inference(self, x, e, features_layer="AlexNet/fc6/Relu:0"):

        net_data = self.load_weights()

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

            # AlexNet/Flatten/Reshape:0
            flat_pool5 = tf.contrib.layers.flatten(pool5)

            # AlexNet/fc6/Relu:0
            fc6 = self._fc_layer(flat_pool5, "fc6", net_data)
            fc7 = self._fc_layer(fc6, "fc7", net_data)
            fc8 = self._fc_layer(fc7, "fc8", net_data)

            features = tf.get_default_graph().get_tensor_by_name(features_layer)

            features = tf.cond(e > self.refine_after, lambda: features, lambda: tf.stop_gradient(features))
            # OUR CLASSIFIER
            with tf.variable_scope("Fully_Connected4_trainable"):
                W = tf.get_variable("weights", shape=[features.get_shape()[1], 382],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[382],
                                    initializer=tf.constant_initializer(0.1))
                fc3 = tf.nn.relu(tf.matmul(features, W) + b)
                fc3 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc3, self.keep_prob), lambda: fc3)

            with tf.variable_scope("Fully_Connected5_trainable"):
                W = tf.get_variable("weights", shape=[382, self.num_classes],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[self.num_classes],
                                    initializer=tf.constant_initializer(0))

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
                 regularizer=tf.contrib.layers.l2_regularizer,
                 regularization_scale=0.,
                 is_training=tf.Variable(True),
                 refine_after=0):

        # Model hyperparameters
        self.num_classes = num_classes
        self.keep_prob = 1 - dropout_rate
        self.regularizer = regularizer(scale=regularization_scale)
        self.is_training = is_training
        self.assign_ops = []
        self.refine_after = refine_after
        self.VGG_FILE = './vgg16_weights.npz'


    def load_weights(self, weight_file):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        return weights, keys

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=name)


    def _conv_layer(self, bottom, name, data, stride=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            W = data[name + '_W']
            kernel = tf.get_variable('filter', W.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(kernel.assign(W))
            b = data[name + '_b']
            biases = tf.get_variable('biases', b.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(biases.assign(b))

            conv = tf.nn.conv2d(bottom, kernel, stride, padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)
            return relu

    def _set_training_op(self, is_training):
        return self.is_training.assign(tf.constant(is_training))

    def _fc_layer(self, bottom, name, data):
        with tf.variable_scope(name):
            W = data[name + '_W']
            kernel = tf.get_variable('weights', W.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(kernel.assign(W))

            b = data[name + '_b']
            biases = tf.get_variable('biases', b.shape, initializer=tf.random_normal_initializer())
            self.assign_ops.append(biases.assign(b))

            return tf.nn.relu(tf.matmul(bottom, W) + b)

    def inference(self, x, e, features_layer="VGG/pool5:0"):

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

            features = tf.cond(e > self.refine_after, lambda: features, lambda: tf.stop_gradient(features))

            # OUR CLASSIFIER
            flat_pool5 = tf.contrib.layers.flatten(features)

            with tf.variable_scope("Fully_Connected0_trainable"):
                W = tf.get_variable("weights", shape=[flat_pool5.get_shape()[1], 382],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[382],
                                    initializer=tf.constant_initializer(0.1))
                fc0 = tf.nn.relu(tf.matmul(flat_pool5, W) + b)
                fc0 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc0, self.keep_prob), lambda: fc0)


            with tf.variable_scope("Fully_Connected1_trainable"):
                W = tf.get_variable("weights", shape=[382, 192],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[192],
                                    initializer=tf.constant_initializer(0.1))
                fc1 = tf.nn.relu(tf.matmul(fc0, W) + b)
                fc1 = tf.cond(self.is_training, lambda: tf.nn.dropout(fc1, self.keep_prob), lambda: fc1)

            with tf.variable_scope("Fully_Connected2_trainable"):
                W = tf.get_variable("weights", shape=[192, self.num_classes],
                                    initializer=tf.random_normal_initializer(stddev=1e-4),
                                    regularizer=self.regularizer)
                b = tf.get_variable("biases", shape=[self.num_classes],
                                    initializer=tf.constant_initializer(0))

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