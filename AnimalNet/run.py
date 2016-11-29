from AnimalNet import AnimalNet
import tensorflow as tf
import numpy as np
from data_utils import Dataset
# Directory in which cifar data is saved
DATA_DIR = '../data'
# Directory for tensorflow logs
LOG_DIR_DEFAULT = './logs/'
EPOCHS = 2000
BATCH_SIZE = 50

# For reproducible research :)
tf.set_random_seed(42)
np.random.seed(42)

net = AnimalNet(num_classes=10, batch_size=BATCH_SIZE, dropout_rate=0.2)
dataset = Dataset(DATA_DIR, rescale_imgs=True, img_shape=(227,227))
images, labels = dataset.get_batch(BATCH_SIZE)


for epoch in range(EPOCHS):
    net.set_training(True)
    images, labels = dataset.get_batch(BATCH_SIZE)
    net.train(images, labels)

    if epoch % 20 == 0:
        net.set_training(False)
        print("********** Step :", epoch, "of", EPOCHS, "**********")
        val_images, val_labels = dataset.get_validation()

        train_acc = net.accuracy(images, labels)
        train_loss = net.loss(images, labels)


        print("Train set accuracy is: ", train_acc)
        print("Train set loss is: ", train_loss)
        print("--------------------------------------------------")


        val_acc = net.accuracy(val_images, val_labels)
        val_loss = net.loss(val_images, val_labels)

        print("Validation set accuracy is: ", val_acc)
        print("Validation set loss is: ", val_loss)
        print("--------------------------------------------------")

        # TODO summaries
        # TODO model saving/loading