from AnimalNet import AnimalNet_Alex, AnimalNet_VGG
import tensorflow as tf
import numpy as np
from data_utils import Dataset
import argparse

# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/'
MAX_STEPS_DEFAULT = 500
BATCH_SIZE_DEFAULT = 50
PRINT_FREQ_DEFAULT = 20
SUBMISSION_DEFAULT = "False"
DROPOUT_DEFAULT = 0
MEAN_SUBTRACTION_DEFAULT = "True"
ARCHITECTURE_DEFAULT = 'alexnet'
SUBMISSION_FILENAME_DEFAULT = 'submission.csv'

def train():

    submission = FLAGS.submission == "True"
    mean_subtraction = FLAGS.mean_subtraction == "True"

    # For reproducible research :)
    tf.set_random_seed(42)
    np.random.seed(42)

    dataset = Dataset(rescale_imgs=True,
                      img_shape=(227, 227),
                      submission=submission,
                      mean_subtraction=mean_subtraction)
    if not submission:
        val_images, val_labels = dataset.val, dataset.val_labels

    with tf.variable_scope("Input"):
        input = tf.placeholder(tf.float32, [None, 227, 227, 3], name='input')
        labels = tf.placeholder(tf.float32, shape=(None, 10), name='labels')

    if FLAGS.architecture == "alexnet":
        net = AnimalNet_Alex(num_classes=10, dropout_rate=FLAGS.dropout)
    else:
        net = AnimalNet_VGG(num_classes=10, dropout_rate=FLAGS.dropout)

    stop_training_op = net._set_training_op(False)
    start_training_op = net._set_training_op(True)

    logits = net.inference(input)
    loss_op = net.loss(logits, labels)
    acc_op = net.accuracy(logits, labels)

    with tf.Session() as sess:

        # Training operation
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

        # Summary
        # summary_op = tf.merge_all_summaries()

        # Initialize variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        for epoch in range(FLAGS.max_steps):
            train_images, train_labels = dataset.get_batch(FLAGS.batch_size)

            train_feed_dict = {input: train_images,
                               labels: train_labels}

            if (epoch+1) % FLAGS.print_freq == 0:
                _, loss, acc = sess.run([stop_training_op, loss_op, acc_op], train_feed_dict)
                print("********** Step :", epoch, "of", FLAGS.max_steps, "**********")

                print("Train set accuracy is: ", acc)
                print("Train set loss is: ", loss)
                print("--------------------------------------------------")

                if not submission:
                    val_feed_dict = {input: val_images,
                                       labels: val_labels}

                    loss, acc = sess.run([loss_op, acc_op], val_feed_dict)

                    print("Validation set accuracy is: ", acc)
                    print("Validation set loss is: ", loss)
                    print("--------------------------------------------------")

            else:
                sess.run([start_training_op, train_op], train_feed_dict)

            # TODO summaries
            # TODO model saving/loading


        #We save your predictions to file
        if submission:
            test_p_file = open(FLAGS.submission_filename,'w')
            test_p_file.write('ImageName,Prediction\n')
            # VGG problem -> not enough memory
            # we need to split test data to batches
            predictions = []
            for i in range(5):
                test_feed_dict = {input: dataset.test[i*100:(i+1)*100]}
                predictions.append(sess.run(logits, test_feed_dict))
            predictions = np.concatenate(predictions)
            test_labels = np.argmax(predictions, 1)

            for i, image in enumerate(dataset.testimages):
                test_p_file.write(image+','+str(test_labels[i])+'\n')
            test_p_file.close()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--dropout', type=float, default=DROPOUT_DEFAULT,
                        help='Dropout on last fully connected layers')
    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation')
    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')
    parser.add_argument('--submission', type=str, default=SUBMISSION_DEFAULT,
                        help='Will also train on validation data')
    parser.add_argument('--mean_subtraction', type=str, default=MEAN_SUBTRACTION_DEFAULT,
                        help='Subtracting mean pixel value from data')
    parser.add_argument('--architecture', type=str, default=ARCHITECTURE_DEFAULT,
                        help='Architecture of network vgg or alexnet')
    parser.add_argument('--submission_filename', type=str, default=SUBMISSION_FILENAME_DEFAULT,
                        help='Name of the file where submissions will be written')

    FLAGS, unparsed = parser.parse_known_args()

    train()