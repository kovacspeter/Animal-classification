from __future__ import print_function, division
from collections import OrderedDict
import logging
import time
import os
import pickle
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import KNeighborsClassifier
import sift
from hyper_pars import *

logger = logging.getLogger(__file__)
data_path = '../data/'


def get_sift_points_from_image(image_path):
    # calculate several types of points of interest from an image
    image_name = os.path.basename(image_path)
    points = []

    # dense points
    if SIFT_USE_DENSE_POINTS:
        dense_points = sift.dense_points(image_path, stride=SIFT_DENSE_POINT_STRIDE)  # N x 2
        logger.debug('dense points: {} for image {}'.format(len(dense_points), image_name))
        points.append(dense_points)

    # Harris points
    if SIFT_USE_HARRIS_POINTS:
        harris_points = sift.compute_hesssian_points(image_path,
                                                     sigma=SIFT_HARRIS_POINT_SIGMA,
                                                     mag_threshold=SIFT_HARRIS_POINT_MAG_THRESHOLD,
                                                     hes_threshold=SIFT_HARRIS_POINT_HES_THRESHOLD,
                                                     nms_neighborhood=SIFT_HARRIS_POINT_NSM_NEIGHBORHOOD)
        logger.debug('harris points: {} for image {}'.format(len(harris_points), image_name))
        points.append(harris_points)

    # Hessian points
    if SIFT_USE_HESSIAN_POINTS:
        hessian_points = sift.compute_harris_points(image_path,
                                                    sigma=SIFT_HESSIAN_POINT_SIGMA,
                                                    mag_threshold=SIFT_HESSIAN_POINT_MAG_THRESHOLD,
                                                    nms_neighborhood=SIFT_HESSIAN_POINT_NSM_NEIGHBORHOOD)
        logger.debug('hessian points: {} for image {}'.format(len(hessian_points), image_name))
        points.append(hessian_points)
    return np.vstack(points)


def get_sift_features_from_image(image_path):
    # calc SIFT features from sift points - 128 dim vector per SIFT point - some SIFT points will be discarded
    points = get_sift_points_from_image(image_path)
    _, sift_features = sift.compute_sift_to_points(image_path, points,
                                                   sigma=1.0, nr_orient_bins=8, nr_spat_bins=4, nr_pix_per_bin=4)
    return sift_features


def get_sift_features_dict_from_images(image_paths, name):
    # collect all sift features from all images, store in dict with image index as key
    start = time.time()
    feature_dict = OrderedDict()  # necessary as we need to maintain order when using dict.values() unpacking
    for idx, image_path in enumerate(image_paths):
        logger.debug('process image ({}/{})'.format(idx, len(image_paths)))
        sift_features = get_sift_features_from_image(image_path)
        len(sift_features)
        feature_dict[idx] = sift_features
    logger.info("feature collection done in {} sec".format(time.time()-start))
    logger.info("saving features...")
    pickle.dump(feature_dict, open(name+'_features.dat', 'wb'))
    return feature_dict


def create_code_book(features):
    # create code_book from feature set
    logger.info("starting k-means clustering...")
    start = time.time()
    if CODE_BOOK_KMEANS_USE_MINI_BATCH:
        mini_batch_k_means = MiniBatchKMeans(n_clusters=CODE_BOOK_KMEANS_CLUSTERS,
                                             verbose=False)
        mini_batch_k_means.fit(features)
        code_book = mini_batch_k_means.cluster_centers_
    else:
        k_means = KMeans(n_clusters=CODE_BOOK_KMEANS_CLUSTERS, verbose=False)  # n_jobs=2 breaks on MAC
        k_means.fit(features)
        code_book = k_means.cluster_centers_
    logger.info("clustering done in {} sec".format(time.time()-start))
    logger.info("saving code book...")
    np.save('code_book.npy', code_book)
    return code_book


def create_histogram_from_features(features, clusters):
    # create a histogram of assigned code_book to features
    if features.shape[0] != 0:
        feature_cluster = euclidean_distances(clusters, features).argmin(axis=0)
        histogram = np.bincount(feature_cluster, minlength=clusters.shape[0]) / features.shape[0]
    else:
        # return empty histogram for images without features
        # e.g. killer+whale_0011.jpg has no Harris points
        histogram = np.zeros((clusters.shape[0]))
    return histogram


def create_histograms(clusters, features_dict, name):
    # create histogram from image features
    logger.info('make {} histograms from image features...'.format(name))
    histograms = np.zeros((len(features_dict), clusters.shape[0]))
    for idx, key in enumerate(features_dict):
        logger.info('create histogram {}/{}'.format(idx, len(features_dict)))
        histograms[idx] = create_histogram_from_features(features_dict[idx], clusters)
    np.save('{}_histograms.npy'.format(name), histograms)
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
    knn = KNeighborsClassifier(n_neighbors=CLASSIFIER_KNN_NEIGHBORS, weights=CLASSIFIER_KNN_WEIGHTS, n_jobs=N_JOBS)
    knn.fit(train_histograms, train_labels)
    predictions = knn.predict(val_histograms)
    return predictions


def classify(train_histograms, train_labels, val_histograms, val_labels):
    predictions = k_nearest_neighbors(train_histograms, train_labels, val_histograms)
    for idx, prediction in enumerate(predictions):
        logger.info("val label: {}, prediction: {}".format(val_labels[idx], prediction))
    return predictions


def compute_accuracy(predictions, truth):
    return (np.sum(predictions == truth)) / len(truth)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    train_images = get_train_images()
    train_labels = get_train_labels()
    val_images = get_val_images()
    val_labels = get_val_labels()

    # extract features for all images
    if True:
        train_features_dict = get_sift_features_dict_from_images(train_images, 'train')
        val_features_dict = get_sift_features_dict_from_images(val_images, 'val')
    else:
        train_features_dict = pickle.load(open('train_features.dat', 'rb'))
        val_features_dict = pickle.load(open('val_features.dat', 'rb'))

    # create code book / clusters
    if True:
        all_train_features = np.vstack(train_features_dict.values())
        code_book = create_code_book(all_train_features)
    else:
        code_book = np.load('code_book.npy')

    # create histograms
    if True:
        train_histograms = create_histograms(code_book, train_features_dict, 'train')
        val_histograms = create_histograms(code_book, val_features_dict, 'val')
    else:
        train_histograms = np.load('train_histogram.npy')
        val_histograms = np.load('val_histogram.npy')

    # classify images
    predictions = classify(train_histograms, train_labels, val_histograms, val_labels)
    print("accuracy: {}".format(compute_accuracy(predictions, val_labels)))
    logger.debug("DONE")
