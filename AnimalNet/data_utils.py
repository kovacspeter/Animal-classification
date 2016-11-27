import numpy as np
from scipy import misc
try:
    import cv2
    CV2_available = True
except ImportError, e:
    from PIL import Image
    CV2_available = False

class Dataset():

    def __init__(self, data_folder, rescale_imgs=False, img_shape=None):
        # If you want to rescale images by default
        self.rescale_imgs = rescale_imgs
        self.img_shape = img_shape

        trainimages = [data_folder + '/' + line.strip().split(" ")[0] for line in open(data_folder + "/trainset-overview.txt", "r")]
        valimages = [data_folder + '/' + line.split(' ')[0] for line in open(data_folder + '/valset-overview.txt', 'r')]
        testimages = [data_folder + '/' + line.strip().split(' ')[0] for line in open(data_folder + '/testset-overview-final.txt', 'r')]

        train_labels = np.array(
            [int(line.strip().split(" ")[1]) for line in open(data_folder + "/trainset-overview.txt", "r")])
        val_labels = np.array(
            [int(line.rstrip().split(' ')[1]) for line in open(data_folder + '/valset-overview.txt', 'r')])

        n_classes = len(np.unique(train_labels))

        # Data are just 150MB load them all
        # -----------------------------------
        # Convert to One-Hot-Encoding
        self.train_labels = self._dense_to_one_hot(train_labels, n_classes)
        self.val_labels = self._dense_to_one_hot(val_labels, n_classes)
        # Load actual images
        self.train = np.array([self._load_image(img) for img in trainimages])
        self.val = np.array([self._load_image(img) for img in valimages])
        self.test = np.array([self._load_image(img) for img in testimages])


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

    def get_validation(self):
        return self.val, self.val_labels