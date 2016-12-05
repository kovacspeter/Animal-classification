from pysift import sift
import numpy as np
from sklearn.cluster import k_means, MiniBatchKMeans, KMeans
import logging
import time

logger = logging.getLogger(__file__)
data_path = '../data/'

def euclidean_distance(x, y):
    assert(len(x) == len(y))
    d = np.sqrt(np.sum((x-y)**2))
    return d


def distances(a,X,distance_fn=euclidean_distance):
    dists = np.zeros(X.shape[0])
    for r in range(0, X.shape[0]):
        dists[r] = distance_fn(a, X[r])

    return dists


def cluster_assignment(features, clusters):
    nr_features = features.shape[0]
    assignments = np.zeros(nr_features, dtype=int)
    for idx, feature in enumerate(features):
        dist = distances(feature, clusters)
        idx_nn = np.where(dist == np.min(dist))[0][0]
        assignments[idx] = idx_nn
    return assignments


def create_histogram_from_features(features, clusters):
    assignments = cluster_assignment(features, clusters)
    histogram = np.zeros(clusters.shape[0], dtype=np.float)
    for i in assignments:
        histogram[i] += 1
    histogram /= len(assignments)
    return histogram


def make_histogram_from_image(imagepath, clusters):
    # Extract point locations from the image using your selected point method and parameters.
    # calc SIFT features - 128 dim vector per SIFT point - some SIFT points will be discarded
    points = get_sift_points_from_image(imagepath, sigma)
    _, sift_features = sift.computeSIFTofPoints(imName1, points, sigma=sigma, nrOrientBins=8, nrSpatBins=4, nrPixPerBin=4)
    histogram = create_histogram_from_features(sift_features, clusters)
    return histogram


def make_histograms_from_images(imagepaths, clusters):
    N = len(imagepaths)
    k = clusters.shape[0]
    histograms = np.zeros((N, k))
    for i, imagepath in enumerate(imagepaths):
        histograms[i] = make_histogram_from_image(imagepath, clusters)
    return histograms


def get_sift_points_from_image(image_path):
    # HYPER
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
    logger.info("There are {} features in image {}".format(len(sift_features), image_path))
    return sift_features


def get_all_sift_features(image_paths):
    start = time.time()
    all_features = []
    for idx, image_path in enumerate(image_paths):
        logger.debug('process image ({}/{})'.format(idx, len(image_paths)))
        sift_features = get_sift_features_from_image(image_path)
        all_features.extend(sift_features)
    logger.debug("no of features in bag: {}".format(len(all_features)))
    logger.debug("feature collection done in {} sec".format(time.time()-start))
    logger.debug("saving features...")
    np.save('all_features.npy'.format(), all_features)
    return all_features


def define_clusters(features, use_mini_batch=True):
    logger.debug("starting k-means clustering...")
    start = time.time()
    if use_mini_batch:
        mini_batch_k_means = MiniBatchKMeans(n_clusters=100, max_iter=1, verbose=True)
        mini_batch_k_means.fit(features)
        centroids = mini_batch_k_means.cluster_centers_
    else:
        centroids = k_means(features, n_clusters=100, max_iter=1)[0]  # HYPER  , n_jobs=2
    logger.debug("clustering done in {} sec".format(time.time()-start))
    return centroids


def get_train_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"trainset-overview.txt", "r")]


def get_val_images():
    return [data_path+line.strip().split(" ")[0] for line in open(data_path+"valset-overview.txt", "r")]


def generate_code_book(all_features):
    code_book = define_clusters(all_features)
    logger.debug("saving code book...")
    np.save('code_book.npy'.format(), code_book)


def classify(clusters):
    train_histograms = make_histograms_from_images(trainimages[:10], clusters)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    use_features = True
    use_code_book = True

    # set code book
    if use_code_book:
        clusters = np.load('code_book.npy')
    elif use_features:
        train_images = get_train_images()
        all_features = np.load('all_features.npy')
        clusters = define_clusters(train_images)
    else:
        train_images = get_train_images()
        all_features = get_all_sift_features(train_images)
        define_clusters(train_images)


    logger.debug("DONE")
