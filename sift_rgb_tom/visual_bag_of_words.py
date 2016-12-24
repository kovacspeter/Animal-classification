from __future__ import print_function, division
import logging
import time
import pickle

import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import KNeighborsClassifier
import hyper_pars
from sift_feature_extraction import logger, get_image_sift_points, get_sift_features_from_image
from rgb_feature_extraction import get_rgb_features_from_image

logger = logging.getLogger(__file__)
data_path = '../data/'


def get_sift_points(image_paths, set_name):
    # get sift points of interest - set flag to reuse previous points
    points_dict = {}
    for idx, image_path in enumerate(image_paths):
        logger.debug('get points ({}/{})'.format(idx, len(image_paths)))
        points_dict[image_path] = get_image_sift_points(image_path)

    pickle.dump(points_dict, open('data/{}_points.dat'.format(set_name), 'wb'))
    return points_dict


def get_features_dict_from_images(image_paths, points_dict, set_name):
    # collect and combine features from all images, store in dict with image index as key
    start = time.time()
    feature_dict = {}
    for idx, image_path in enumerate(image_paths):
        logger.debug('process image ({}/{})'.format(idx, len(image_paths)))

        # extract features on sift points_dict
        assert(hyper_pars.USE_RGB or hyper_pars.USE_SIFT)
        sift_features = None
        rgb_features = None

        if hyper_pars.USE_SIFT:
            sift_features = get_sift_features_from_image(image_path, points_dict[image_path])
        if hyper_pars.USE_RGB:
            rgb_features = get_rgb_features_from_image(image_path, points_dict[image_path])

        # combine features
        if hyper_pars.USE_SIFT and not hyper_pars.USE_RGB:
            feature_dict[image_path] = sift_features
        elif not hyper_pars.USE_SIFT and hyper_pars.USE_RGB:
            feature_dict[image_path] = rgb_features
        elif hyper_pars.USE_SIFT and hyper_pars.USE_RGB:
            feature_dict[image_path] = np.hstack((sift_features, rgb_features))

    logger.info("feature collection done in {} sec".format(time.time()-start))
    logger.info("saving features...")
    pickle.dump(feature_dict, open('data/{}_features.dat'.format(set_name), 'wb'))
    return feature_dict


def create_code_book(features):
    # create code_book from feature set
    logger.info("creating code book using k-means")
    start = time.time()
    if hyper_pars.CODE_BOOK_KMEANS_USE_MINI_BATCH:
        mini_batch_k_means = MiniBatchKMeans(n_clusters=hyper_pars.CODE_BOOK_KMEANS_CLUSTERS,
                                             verbose=True)
        mini_batch_k_means.fit(features)
        code_book = mini_batch_k_means.cluster_centers_
    else:
        k_means = KMeans(n_clusters=hyper_pars.CODE_BOOK_KMEANS_CLUSTERS, verbose=False)  # n_jobs=2 breaks on MAC
        k_means.fit(features)
        code_book = k_means.cluster_centers_
    logger.debug("clustering done in {} sec".format(time.time()-start))
    logger.debug("saving code book...")
    np.save('data/code_book.npy', code_book)
    return code_book


def create_histogram_from_features(features, clusters):
    # create a histogram of assigned code_book to features
    if features.shape[0] != 0:
        feature_cluster = euclidean_distances(clusters, features).argmin(axis=0)
        histogram = np.bincount(feature_cluster, minlength=clusters.shape[0]) / features.shape[0]
    else:
        # return empty histogram for images without features
        # e.g. killer+whale_0011.jpg has no Harris points
        logger.warn('empty histogram encountered')
        histogram = np.zeros((clusters.shape[0]))
    return histogram


def create_histograms(clusters, image_names, features_dict, name):
    # create histogram from image features
    logger.info('make {} histograms from image features...'.format(name))
    histograms = np.zeros((len(features_dict), clusters.shape[0]))
    for idx, key in enumerate(image_names):
        logger.debug('create histogram {}/{}'.format(idx, len(image_names)))
        histograms[idx] = create_histogram_from_features(features_dict[key], clusters)
    np.save('data/{}_histograms.npy'.format(name), histograms)
    return histograms


def get_train_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"trainset-overview.txt", "r")]


def get_train_labels():
    return [int(line.strip().split(" ")[1]) for line in open(data_path+"trainset-overview.txt", "r")]


def get_val_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"valset-overview.txt", "r")]


def get_val_labels():
    return [int(line.strip().split(" ")[1]) for line in open(data_path+"valset-overview.txt", "r")]


def k_nearest_neighbors(train_histograms, train_labels, val_histograms):
    logger.info('running knn with {} neighbors and {} weights'.format(hyper_pars.CLASSIFIER_KNN_NEIGHBORS,
                                                                      hyper_pars.CLASSIFIER_KNN_WEIGHTS))
    knn = KNeighborsClassifier(n_neighbors=hyper_pars.CLASSIFIER_KNN_NEIGHBORS,
                               weights=hyper_pars.CLASSIFIER_KNN_WEIGHTS,
                               n_jobs=hyper_pars.N_JOBS)
    knn.fit(train_histograms, train_labels)
    predictions = knn.predict(val_histograms)
    return predictions


def classify(train_histograms, train_labels, val_histograms, val_labels):
    predictions = k_nearest_neighbors(train_histograms, train_labels, val_histograms)
    for idx, prediction in enumerate(predictions):
        logger.debug("val label: {}, prediction: {}".format(val_labels[idx], prediction))
    return predictions


def compute_accuracy(predictions, truth):
    return (np.sum(predictions == truth)) / len(truth)


def run_classification():
    train_images = get_train_images()
    train_labels = get_train_labels()
    val_images = get_val_images()
    val_labels = get_val_labels()

    # get sift points
    if False:
        train_points_dict = get_sift_points(train_images, 'train')
        val_points_dict = get_sift_points(val_images, 'val')
    else:
        train_points_dict = pickle.load(open('data/train_points.dat', 'rb'))
        val_points_dict = pickle.load(open('data/val_points.dat', 'rb'))

    # extract features for all images
    if False:
        train_features_dict = get_features_dict_from_images(train_images, train_points_dict, 'train')
        val_features_dict = get_features_dict_from_images(val_images, val_points_dict, 'val')
    else:
        train_features_dict = pickle.load(open('data/train_features.dat', 'rb'))
        val_features_dict = pickle.load(open('data/val_features.dat', 'rb'))

    # create code book / clusters
    if False:
        all_train_features = np.vstack(train_features_dict.values())
        code_book = create_code_book(all_train_features)
    else:
        code_book = np.load('data/code_book.npy')

    # create histograms
    if True:
        train_histograms = create_histograms(code_book, train_images, train_features_dict, 'train')
        val_histograms = create_histograms(code_book, val_images, val_features_dict, 'val')
    else:
        train_histograms = np.load('data/train_histograms.npy')
        val_histograms = np.load('data/val_histograms.npy')

    # classify images
    predictions = classify(train_histograms, train_labels, val_histograms, val_labels)
    logger.info("accuracy: {}".format(compute_accuracy(predictions, val_labels)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')

    # run experiment
    run_classification()
    logger.info("DONE")


