from __future__ import print_function
from pysift import sift
import numpy as np
from collections import Counter
from sklearn.cluster import k_means, MiniBatchKMeans, KMeans
import logging
import time

logger = logging.getLogger(__file__)
data_path = '../data/'


def euclidean_distance(x, y):
    assert(len(x) == len(y))
    d = np.sqrt(np.sum((x - y)**2))
    return d


def distances(a, X, distance_fn=euclidean_distance):
    dists = np.zeros(X.shape[0])
    for r in range(0, X.shape[0]):
        dists[r] = distance_fn(a, X[r])
    return dists


def assign_features_to_clusters(features, clusters):
    # assign each feature to the nearest cluster
    nr_features = features.shape[0]
    assignments = np.zeros(nr_features, dtype=int)
    for idx, feature in enumerate(features):
        dist = distances(feature, clusters)
        idx_nn = np.where(dist == np.min(dist))[0][0]
        assignments[idx] = idx_nn
    return assignments


def create_histogram_from_features(features, clusters):
    # create a histogram of assigned clusters to features
    assignments = assign_features_to_clusters(features, clusters)
    histogram = np.zeros(clusters.shape[0], dtype=np.float)
    for i in assignments:
        histogram[i] += 1
    histogram /= len(assignments)
    return histogram


def create_histogram_from_image(image_path, clusters):
    # create a feature occurrence histogram from a image
    sift_features = get_sift_features_from_image(image_path)
    histogram = create_histogram_from_features(sift_features, clusters)
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


def get_sift_points_from_image(image_path):
    # calculate several types of points of interest from an image
    # HYPER PARAMETERS HERE
    dense_points = sift.dense_points(image_path, stride=15)
    hessian_points = sift.compute_hes(image_path, sigma=1.0, magThreshold=10, hesThreshold=5, NMSneighborhood=10)
    harris_points = sift.computeHar(image_path, sigma=1.0, magThreshold=10, NMSneighborhood=10)

    # combine dense, harris and hessian interest points
    points = np.concatenate((dense_points, hessian_points, harris_points))
    logger.debug("There are {} SIFT points (dens={}, hes={}, har={}) in image {}".format(
                len(points), dense_points.shape[0], hessian_points.shape[0], harris_points.shape[0], image_path))
    return points


def get_sift_features_from_image(image_path):
    points = get_sift_points_from_image(image_path)
    # calc SIFT features - 128 dim vector per SIFT point - some SIFT points will be discarded
    _, sift_features = sift.computeSIFTofPoints(image_path, points,
                                                sigma=1.0, nrOrientBins=8, nrSpatBins=4, nrPixPerBin=4)
    logger.debug("There are {} features in image {}".format(len(sift_features), image_path))
    return sift_features


def get_all_sift_features(image_paths):
    # collect all sift features from all images
    start = time.time()
    features = []
    for idx, image_path in enumerate(image_paths):
        logger.debug('process image ({}/{})'.format(idx, len(image_paths)))
        sift_features = get_sift_features_from_image(image_path)
        features.extend(sift_features)
    logger.debug("no of features in bag: {}".format(len(features)))
    logger.debug("feature collection done in {} sec".format(time.time()-start))
    logger.debug("saving features...")
    np.save('all_features.npy'.format(), features)
    return features


def create_clusters(features, use_mini_batch=True, n_clusters=50):
    # create clusters from feature set
    # HYPER PARAMETERS HERE
    logger.debug("starting k-means clustering...")
    start = time.time()
    if use_mini_batch:
        mini_batch_k_means = MiniBatchKMeans(n_clusters=n_clusters, max_iter=10, verbose=True)
        mini_batch_k_means.fit(features)
        centroids = mini_batch_k_means.cluster_centers_
    else:
        centroids = k_means(features, n_clusters=n_clusters, max_iter=1)[0]  # HYPER  , n_jobs=2
    logger.debug("clustering done in {} sec".format(time.time()-start))
    logger.debug("saving code book...")
    np.save('clusters_{}.npy'.format(n_clusters), centroids)
    return centroids


def knn(data_point, train_data, train_label, k):
    # return the most common neighbour label from the k nearest for a data point
    dist = distances(data_point, train_data)
    sorted_idx = list(dist.argsort())
    labels = [train_label[sorted_idx.index(i)] for i in range(k)]
    return Counter(labels).most_common(1)[0][0]


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


def classify(train_histograms, train_labels, val_histograms, val_labels):
    k = 100
    predictions = []
    for i in xrange(len(val_labels)):
        prediction = knn(val_histograms[i], train_histograms, train_labels, k)
        print("val label: {}, prediction: {}".format(val_labels[i], prediction))
        predictions.append(prediction)
    return predictions


def compute_accuracy(predictions, truth):
    return (np.sum(predictions == truth)) / len(truth)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    train_images = get_train_images()
    train_labels = get_train_labels()
    val_images = get_val_images()
    val_labels = get_val_labels()

    # all_features = get_all_sift_features(train_images)
    # create_clusters(np.load('all_features.npy'))
    create_histograms(np.load('clusters_50.npy'), train_images, val_images)
    predictions = classify(np.load('train_histogram.npy'), train_labels, np.load('val_histogram.npy'), val_labels)
    print("accuracy: {}".format(compute_accuracy(predictions, val_labels)))

    logger.debug("DONE")
