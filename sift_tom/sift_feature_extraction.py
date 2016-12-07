from collections import OrderedDict
import os
import pickle
from hyper_pars import *
import sift
import time
import numpy as np
import logging

logger = logging.getLogger(__file__)


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
        harris_points = sift.compute_hessian_points(image_path,
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
