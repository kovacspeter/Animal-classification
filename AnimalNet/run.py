from AnimalNet import AnimalNet_Alex, AnimalNet_VGG
import tensorflow as tf
import numpy as np
from data_utils import Dataset
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools

# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/'
MAX_STEPS_DEFAULT = 500
BATCH_SIZE_DEFAULT = 50
PRINT_FREQ_DEFAULT = 20
DROPOUT_DEFAULT = 0.
L2_REG_DEFAULT = 0.
ARCHITECTURE_DEFAULT = 'alexnet'
SUBMISSION_FILENAME_DEFAULT = 'submission.csv'
LEARNING_RATE_DEFAULT = 1e-4
SUBMISSION_DEFAULT = "False"
MEAN_SUBTRACTION_DEFAULT = "True"
AUGMENTATION_DEFAULT = "False"
FEATURE_LAYER_DEFAULT = ''

def train():

    submission = FLAGS.submission == "True"
    mean_subtraction = FLAGS.mean_subtraction == "True"
    augmentation = FLAGS.augmentation == "True"

    # For reproducible research :)
    tf.set_random_seed(42)
    np.random.seed(42)

    IMG_SIZE = 227

    dataset = Dataset(rescale_imgs=True,
                      img_shape=(IMG_SIZE, IMG_SIZE),
                      submission=submission,
                      mean_subtraction=mean_subtraction)
    if not submission:
        val_images, val_labels = dataset.val, dataset.val_labels

    with tf.variable_scope("Input"):
        input = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='input')
        labels = tf.placeholder(tf.float32, shape=(None, 10), name='labels')

    if FLAGS.architecture == "alexnet":
        net = AnimalNet_Alex(num_classes=10,
                             dropout_rate=FLAGS.dropout,
                             regularization_scale=FLAGS.l2_reg,
                             learning_rate=FLAGS.learning_rate)
    else:
        net = AnimalNet_VGG(num_classes=10,
                            dropout_rate=FLAGS.dropout,
                            regularization_scale=FLAGS.l2_reg,
                            learning_rate=FLAGS.learning_rate)

    stop_training_op = net._set_training_op(False)
    start_training_op = net._set_training_op(True)

    logits = net.inference(input, FLAGS.feature_layer)
    loss_op = net.loss(logits, labels)
    acc_op = net.accuracy(logits, labels)

    summary = tf.merge_all_summaries()

    with tf.Session() as sess:

        if not tf.gfile.Exists(FLAGS.log_dir):
            tf.gfile.MakeDirs(FLAGS.log_dir)
        train_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/train', sess.graph)
        test_writer = tf.train.SummaryWriter(FLAGS.log_dir + '/test', sess.graph)

        # Training operation
        train_op = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

        # Summary
        # summary_op = tf.merge_all_summaries()

        # Initialize variables
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        val_feed_dict = {input: val_images,
                         labels: val_labels}

        for epoch in range(FLAGS.max_steps):
            train_images, train_labels = dataset.get_batch(FLAGS.batch_size)
            train_feed_dict = {input: train_images,
                               labels: train_labels}

            if (epoch+1) % FLAGS.print_freq == 0:
                _, loss, acc, summary_str = sess.run([stop_training_op, loss_op, acc_op, summary], train_feed_dict)
                train_writer.add_summary(summary_str, epoch)
                train_writer.flush()

                print("********** Step :", epoch, "of", FLAGS.max_steps, "**********")

                print("Train set accuracy is: ", acc)
                print("Train set loss is: ", loss)
                print("--------------------------------------------------")

                if not submission:

                    loss, acc, summary_str = sess.run([loss_op, acc_op, summary], val_feed_dict)

                    test_writer.add_summary(summary_str, epoch)
                    test_writer.flush()
                    print("Validation set accuracy is: ", acc)
                    print("Validation set loss is: ", loss)
                    print("--------------------------------------------------")

            else:
                sess.run([start_training_op, train_op], train_feed_dict)

            # TODO model saving/loading

        if not submission:
            cnf_matrix = net.get_confusion_matrix(logits, labels, sess, val_feed_dict)
            plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')

            imgs = net.get_problematic_photos(logits, labels, sess, val_feed_dict, val_images, 5)
            plot_imgs(imgs)


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


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

  classes = ['blue_w', 'dog', 'chimp', 'fox', 'gorilla', 'killer_w', 'seal', 'tiger', 'wolf', 'zebra']

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig("./confusion_matrix_" + FLAGS.architecture)

def plot_imgs(imgs):
  from matplotlib import pyplot as plt
  from scipy.misc import toimage
  from PIL import Image

  classes = ['blue_w', 'dog', 'chimp', 'fox', 'gorilla', 'killer_w', 'seal', 'tiger', 'wolf', 'zebra']

  merged = []
  fig = plt.figure()

  for key in imgs:
    images = [toimage(np.reshape(img, (227, 227, 3))) for img in imgs[key]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    img = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      img.paste(im, (x_offset, 0))
      x_offset += im.size[0]

    ax = fig.add_subplot(10, 1, key+1)
    ax.tick_params(axis=u'both', which=u'both', length=0,
                   labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax.set_ylabel(classes[key], rotation=50)
    ax.imshow(img)
    merged.append(img)

  plt.savefig("./hard_images_" + FLAGS.architecture)

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate of optimizer.')
    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
    parser.add_argument('--dropout', type=float, default=DROPOUT_DEFAULT,
                        help='Dropout on last fully connected layers')
    parser.add_argument('--l2_reg', type=float, default=L2_REG_DEFAULT,
                        help='L2 regularization scale')
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
    parser.add_argument('--augmentation', type=str, default=AUGMENTATION_DEFAULT,
                        help='Image augmentation')
    parser.add_argument('--architecture', type=str, default=ARCHITECTURE_DEFAULT,
                        help='Architecture of network can be vgg or alexnet')
    parser.add_argument('--submission_filename', type=str, default=SUBMISSION_FILENAME_DEFAULT,
                        help='Name of the file where submissions will be written')
    parser.add_argument('--feature_layer', type=str, default=FEATURE_LAYER_DEFAULT,
                        help='Layer tensor name on which we will train our classifier.')

    FLAGS, unparsed = parser.parse_known_args()


    if FLAGS.feature_layer == FEATURE_LAYER_DEFAULT:
        if FLAGS.architecture == 'vgg':
            FLAGS.feature_layer = "VGG/pool5:0"
        else:
            FLAGS.feature_layer = "AlexNet/Fully_Connected3/Relu:0"

    train()