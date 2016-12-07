import numpy as np
from scipy import misc
try:
    import cv2
    CV2_available = True
except ImportError, e:
    from PIL import Image
    CV2_available = False

DATA_DIR = '../data/'

class Dataset():

    def __init__(self, rescale_imgs=False, img_shape=None, submission=False, mean_subtraction=True):
        # If you want to rescale images by default
        self.rescale_imgs = rescale_imgs
        self.img_shape = img_shape

        self.trainimages = [line.strip().split(" ")[0] for line in open(DATA_DIR + "trainset-overview.txt", "r")]
        self.valimages = [line.split(' ')[0] for line in open(DATA_DIR + 'valset-overview.txt', 'r')]
        self.testimages = [line.strip().split(' ')[0] for line in open(DATA_DIR + 'testset-overview-final.txt', 'r')]

        train_labels = np.array(
            [int(line.strip().split(" ")[1]) for line in open(DATA_DIR + "trainset-overview.txt", "r")])
        val_labels = np.array(
            [int(line.rstrip().split(' ')[1]) for line in open(DATA_DIR + 'valset-overview.txt', 'r')])

        n_classes = len(np.unique(train_labels))

        # Data are just 150MB load them all
        # -----------------------------------
        # Convert to One-Hot-Encoding
        self.train_labels = self._dense_to_one_hot(train_labels, n_classes)
        self.val_labels = self._dense_to_one_hot(val_labels, n_classes)

        # Load actual images
        self.train = np.array([np.asarray(self._load_image(DATA_DIR + img), np.float32) for img in self.trainimages])
        self.val = np.array([np.asarray(self._load_image(DATA_DIR + img), np.float32) for img in self.valimages])
        self.test = np.array([np.asarray(self._load_image(DATA_DIR + img), np.float32) for img in self.testimages])

        all_data = np.concatenate([self.train, self.val, self.test])
        mean = np.mean(all_data)

        if mean_subtraction:
            self.train -= mean
            self.val -= mean
            self.test -= mean

        if submission:
            self.train = np.concatenate([self.train, self.val])
            self.train_labels = np.concatenate([self.train_labels, self.val_labels])

    def _load_image(self, file_name):
        if CV2_available:
            img = cv2.imread(file_name)
            if self.rescale_imgs:
                img = cv2.resize(img, self.img_shape)
        else:
            img = Image.open(file_name)
            img.load()
            img = np.asarray(img, dtype="int32")
            if self.rescale_imgs:
                img = misc.imresize(img, self.img_shape)

        return img

    def _dense_to_one_hot(self, labels_dense, num_classes):
        """
        Convert class labels from scalars to one-hot vectors.
        Args:
        labels_dense: Dense labels.
        num_classes: Number of classes.

        Outputs:
        labels_one_hot: One-hot encoding for labels.
        """
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot

    def get_batch(self, batch_size):
        indices = np.arange(len(self.train))
        np.random.shuffle(indices)
        return self.train[indices[:batch_size]], self.train_labels[indices[:batch_size]]
