import os
import logging

import numpy as np

import hyper_pars
import sift

logger = logging.getLogger(__file__)


def get_sift_points_from_image(image_path):
    # calculate several types of points of interest from an image
    image_name = os.path.basename(image_path)
    points = []

    # dense points
    if hyper_pars.SIFT_USE_DENSE_POINTS:
        dense_points = sift.dense_points(image_path, stride=hyper_pars.SIFT_DENSE_POINT_STRIDE)  # N x 2
        logger.debug('dense points: {} for image {}'.format(len(dense_points), image_name))
        points.append(dense_points)

    # Harris points
    if hyper_pars.SIFT_USE_HARRIS_POINTS:
        harris_points = sift.compute_hessian_points(image_path,
                                                    sigma=hyper_pars.SIFT_HARRIS_POINT_SIGMA,
                                                    mag_threshold=hyper_pars.SIFT_HARRIS_POINT_MAG_THRESHOLD,
                                                    hes_threshold=hyper_pars.SIFT_HARRIS_POINT_HES_THRESHOLD,
                                                    nms_neighborhood=hyper_pars.SIFT_HARRIS_POINT_NSM_NEIGHBORHOOD)
        logger.debug('harris points: {} for image {}'.format(len(harris_points), image_name))
        points.append(harris_points)

    # Hessian points
    if hyper_pars.SIFT_USE_HESSIAN_POINTS:
        hessian_points = sift.compute_harris_points(image_path,
                                                    sigma=hyper_pars.SIFT_HESSIAN_POINT_SIGMA,
                                                    mag_threshold=hyper_pars.SIFT_HESSIAN_POINT_MAG_THRESHOLD,
                                                    nms_neighborhood=hyper_pars.SIFT_HESSIAN_POINT_NSM_NEIGHBORHOOD)
        logger.debug('hessian points: {} for image {}'.format(len(hessian_points), image_name))
        points.append(hessian_points)

    # combine
    points = np.vstack(points)

    # save points
    logger.debug("saving sift points...")
    np.save('data/code_book.npy', points)
    return points


def get_sift_features_from_image(image_path, points):
    # calc SIFT features from sift points - 128 dim vector per SIFT point - some SIFT points will be discarded
    _, sift_features = sift.compute_sift_to_points(image_path, points,
                                                   sigma=1.0, nr_orient_bins=8, nr_spat_bins=4, nr_pix_per_bin=4)
    return sift_features
