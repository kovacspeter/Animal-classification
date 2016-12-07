from __future__ import print_function, division
import logging
import time
import numpy as np
from sklearn.metrics import euclidean_distances
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.neighbors import KNeighborsClassifier
from pysift import sift

logger = logging.getLogger(__file__)
data_path = '../AnimalNet/'


def get_sift_points_from_image(image_path):
    # calculate several types of points of interest from an image
    # HYPER PARAMETERS HERE
    dense_points = sift.dense_points(image_path, stride=25)
    hessian_points = sift.compute_hes(image_path, sigma=1.0, magThreshold=15, hesThreshold=10, NMSneighborhood=10)
    harris_points = sift.computeHar(image_path, sigma=1.0, magThreshold=5, NMSneighborhood=10)

    # combine dense, harris and hessian interest points
    points = np.concatenate((dense_points, hessian_points, harris_points))
    logger.debug("There are {} SIFT points (dens={}, hes={}, har={}) in image {}".format(
                len(points), dense_points.shape[0], hessian_points.shape[0], harris_points.shape[0], image_path))
    return points


def get_sift_features_from_image(image_path):
    # calc SIFT features from sift points - 128 dim vector per SIFT point - some SIFT points will be discarded
    points = get_sift_points_from_image(image_path)
    _, sift_features = sift.computeSIFTofPoints(image_path, points,
                                                sigma=1.0, nrOrientBins=8, nrSpatBins=4, nrPixPerBin=4)
    logger.debug("There are {} SIFT features in image {}".format(len(sift_features), image_path))
    return sift_features


def get_sift_features_from_images(image_paths, name):
    # collect all sift features from all images
    logger.debug("no of images: {}".format(len(image_paths)))
    start = time.time()
    features_list = []
    for idx, image_path in enumerate(image_paths):
        logger.debug('process image ({}/{})'.format(idx, len(image_paths)))
        sift_features = get_sift_features_from_image(image_path)
        len(sift_features)
        features_list.append(sift_features)
    logger.debug("no of features in bag: {}".format(sum([len(i) for i in features_list])))
    logger.debug("feature collection done in {} sec".format(time.time()-start))
    logger.debug("saving features...")
    features = np.vstack(features_list)
    np.save('{}_features'.format(name), features)
    return features


def create_clusters(features, use_mini_batch=True):
    # create clusters from feature set
    # HYPER PARAMETERS HERE
    logger.info("starting k-means clustering...")
    start = time.time()
    if use_mini_batch:
        mini_batch_k_means = MiniBatchKMeans(n_clusters=200, max_iter=3, verbose=True)  # HYPER
        mini_batch_k_means.fit(features)
        centroids = mini_batch_k_means.cluster_centers_
    else:
        k_means = KMeans(n_clusters=200, verbose=True)  # HYPER  , n_jobs=2 breaks on MAC
        k_means.fit(features)
        centroids = k_means.cluster_centers_
    logger.info("clustering done in {} sec".format(time.time()-start))
    logger.info("saving code book...")
    np.save('clusters_{}.npy'.format(n_clusters), centroids)
    return centroids


def create_histogram_from_features(features, clusters):
    # create a histogram of assigned clusters to features
    feature_cluster = euclidean_distances(clusters, features).argmin(axis=0)
    histogram = np.bincount(feature_cluster, minlength=clusters.shape[0]) / features.shape[0]
    return histogram


def create_histogram_from_image(image_path, clusters):
    # create a feature occurrence histogram from a image
    features = get_sift_features_from_image(image_path)
    histogram = create_histogram_from_features(features, clusters)
    return histogram


def create_histograms_from_images(image_paths, clusters):
    # create a feature occurrence histogram for each image
    n = len(image_paths)
    k = clusters.shape[0]
    histograms = np.zeros((n, k))
    for i, image_path in enumerate(image_paths):
        logger.info('create histogram {}/{}'.format(i, len(image_paths)))
        histograms[i] = create_histogram_from_image(image_path, clusters)
    return histograms


def get_train_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"trainset-overview.txt", "r")]


def get_train_labels():
    return [int(line.strip().split(" ")[1]) for line in open(data_path+"trainset-overview.txt", "r")]


def get_val_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"valset-overview.txt", "r")]


def get_val_labels():
    return [int(line.strip().split(" ")[1]) for line in open(data_path+"valset-overview.txt", "r")]


def create_histograms(clusters, train_images, val_images):
    logger.info("make train image histograms...")
    train_histograms = create_histograms_from_images(train_images, clusters)
    np.save('train_histogram.npy', train_histograms)

    logger.info("make val image histograms...")
    val_histograms = create_histograms_from_images(val_images, clusters)
    np.save('val_histogram.npy', val_histograms)


def k_nearest_neighbors(train_histograms, train_labels, val_histograms):
    knn = KNeighborsClassifier(n_neighbors=10, weights='uniform', n_jobs=2)
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
    logging.basicConfig(level=logging.INFO)

    train_images = get_train_images()
    train_labels = get_train_labels()
    val_images = get_val_images()
    val_labels = get_val_labels()

    # create codebook
    if True:
        if False:
            train_features = np.load('train_features.npy')
        else:
            train_features = get_sift_features_from_images(train_images, 'train')
        create_clusters(train_features)

    # create histograms
    if True:
        clusters = np.load('clusters_200.npy')
        create_histograms(clusters, train_images, val_images)

    # classify images
    predictions = classify(np.load('train_histogram.npy'), train_labels, np.load('val_histogram.npy'), val_labels)
    print("accuracy: {}".format(compute_accuracy(predictions, val_labels)))
    logger.debug("DONE")
