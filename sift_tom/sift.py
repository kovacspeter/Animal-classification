# a SIFT implementation in python
# including some simple interest point detectors
# meant for educational purposes only.
#
# Feel free to distribute, yet, please keep my name so I can get bug reports
#
# Author: Jan van Gemert (j.c.vanGemert@uva.nl)
#

import numpy as np
from pylab import imread, matplotlib
import sys
from numpy import exp, pi, cos, sin, ceil, sqrt, arctan2
import scipy.ndimage.filters as filters


def compute_derivatives(im_name, sigma=1.0):
    im_rgb = imread(im_name)
    im_hsv = matplotlib.colors.rgb_to_hsv(im_rgb)
    im = im_hsv[:, :, 2]

    # Derivatives
    dx = filters.gaussian_filter(im, sigma=sigma, order=[1, 0])
    dy = filters.gaussian_filter(im, sigma=sigma, order=[0, 1])
    return dx, dy


def grad_as_hsv(im_name, sigma=1.0):
    dx, dy = compute_derivatives(im_name, sigma=sigma)
    ang = arctan2(dy, dx)
    mag = sqrt(dx * dx + dy * dy)

    threshold = mag.max()/2
    mag[mag > threshold] = threshold

    # convert to  HSV
    hsv = np.zeros((dx.shape[0], dx.shape[1], 3), dtype=np.float32)
    hsv[..., 2] = 1
    hsv[..., 0] = (ang + pi) / (2 * pi)
    hsv[..., 1] = mag / mag.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb


def dense_points(im_name, stride=5):
    im_rgb = imread(im_name)
    [rows, cols, _] = im_rgb.shape
    
    # divide 1 stride on both sides
    offset_r = rows % stride
    offset_c = cols % stride

    r = np.arange(offset_r/2, rows, stride)
    c = np.arange(offset_c/2, cols, stride)
    x, y = np.meshgrid(c, r)
    grid = np.array([x.flatten(), y.flatten()]).T
    return grid


def compute_hessian_points(im_name, sigma=1, mag_threshold=10, hes_threshold=5, nms_neighborhood=10):
    im_rgb = imread(im_name)
    
    # convert to grayscale
    im_hsv = matplotlib.colors.rgb_to_hsv(im_rgb)
    im = im_hsv[:, :, 2]
    
    # Derivatives
    dxx = filters.gaussian_filter(im, sigma=sigma, order=[2, 0])
    dyy = filters.gaussian_filter(im, sigma=sigma, order=[0, 2])
    lapl = sigma * (dxx + dyy)

    # non max suppression and thresholding of maxima
    data_max = filters.maximum_filter(lapl, nms_neighborhood)
    maxima = (lapl == data_max)
    maxima = np.logical_and(maxima, data_max > hes_threshold)

    # non max suppression and thresholding of minima
    data_min = filters.minimum_filter(lapl, nms_neighborhood)
    minima = (lapl == data_min)
    minima = np.logical_and(minima, data_min < -hes_threshold)

    extrema = np.logical_or(maxima, minima)

    dx = filters.gaussian_filter(im, sigma=sigma, order=[1, 0])
    dy = filters.gaussian_filter(im, sigma=sigma, order=[0, 1])
    mag = sigma * np.sqrt(dx*dx + dy * dy)
    extrema = np.logical_and(extrema, mag > mag_threshold)

    [r, c] = np.where(extrema)

    return np.array([c, r]).T


def compute_harris_points(im_name, sigma=1, k=0.04, mag_threshold=10, nms_neighborhood=10):
    dx, dy = compute_derivatives(im_name, sigma=sigma)

    # structure tensor
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # average with integration scale
    i_scale = 2*sigma
    dx2 = filters.gaussian_filter(dx2, sigma=i_scale)
    dy2 = filters.gaussian_filter(dy2, sigma=i_scale)
    dxy = filters.gaussian_filter(dxy, sigma=i_scale)

    # cornerness, Det - k(Trace)^2
    R = dx2 * dy2 - dxy * dxy - k * (dx2 + dy2) * (dx2 + dy2)

    # non-maximum suppression 
    data_max = filters.maximum_filter(R, nms_neighborhood)
    maxima = (R == data_max)
    maxima = np.logical_and(maxima, data_max > 0)

    # threshold magnitude
    mag = sigma * np.sqrt(dx * dx + dy * dy)
    maxima = np.logical_and( maxima, mag > mag_threshold)

    [r, c] = np.where(maxima)

    return np.array([c, r]).T


def compute_sift_to_points(im_name, points, sigma=1.0, nr_orient_bins=8, nr_spat_bins=4, nr_pix_per_bin=4):

    dx, dy = compute_derivatives(im_name, sigma=sigma)
    r, c = dx.shape
    mag = np.sqrt(dx*dx + dy * dy)

    # put derivative vectors in an array, and add system precision 
    deriv_vecs = np.array([dx.flatten(), dy.flatten()]).T + sys.float_info.epsilon

    # L2 normalize so dotprod is the cosine
    deriv_vecs = (deriv_vecs.T / np.linalg.norm(deriv_vecs, 2, axis=1).T).T

    bin_centers = np.arange(nr_orient_bins) * (2 * pi)/(nr_orient_bins * 2)
    bin_vec = np.array([cos(bin_centers), sin(bin_centers)])

    # compute cosine similarity
    angularsim_mat = np.dot(deriv_vecs, bin_vec)

    # invert, cause higher is better
    angularsim_mat = 1.0 - angularsim_mat

    # kernel density estimator of angular bins
    sigma = 0.1 * pi/nr_orient_bins
    angularsim_mat = exp(-(angularsim_mat**2) / (2*(sigma**2)))
    # normalize to sum to 1
    angularsim_mat = (angularsim_mat.T / np.linalg.norm(angularsim_mat + sys.float_info.epsilon, 1, axis=1)).T

    # multiply with magnitude
    angularsim_mat = (angularsim_mat.T * mag.flatten()).T

    # reshape to 3D for the spatial aggregation of the histogram
    angularsim_mat = angularsim_mat.reshape((r, c, nr_orient_bins))

    # spatial interpolation
    agg_ang = filters.gaussian_filter(angularsim_mat, sigma=[nr_pix_per_bin/3, nr_pix_per_bin/3, 0.1])

    nr_pix_per_bin_div2 = ceil(nr_pix_per_bin / 2)
    feat_dim = nr_spat_bins*nr_spat_bins*nr_orient_bins
    feats = np.zeros((r-nr_pix_per_bin*nr_spat_bins, c-nr_pix_per_bin*nr_spat_bins, feat_dim))
    idx = 0
    for i in np.arange(nr_spat_bins):
        for j in np.arange(nr_spat_bins):
            r_begin = i * nr_pix_per_bin + nr_pix_per_bin_div2
            r_end = r - nr_pix_per_bin * (nr_spat_bins-i-1) - nr_pix_per_bin_div2
            c_begin = j * nr_pix_per_bin + nr_pix_per_bin_div2
            c_end = c - nr_pix_per_bin * (nr_spat_bins-j-1) - nr_pix_per_bin_div2

            feats[:, :, idx:nr_orient_bins+idx] = agg_ang[r_begin:r_end, c_begin:c_end, :]
            idx += nr_orient_bins

    points_c, points_r = points[:, 0], points[:, 1]
    in_border = points_r>nr_pix_per_bin*nr_spat_bins
    in_border = np.logical_and(in_border, points_r < r-nr_pix_per_bin*nr_spat_bins)
    in_border = np.logical_and(in_border, points_c > nr_pix_per_bin*nr_spat_bins)
    in_border = np.logical_and(in_border, points_c < c-nr_pix_per_bin*nr_spat_bins)
    points_r = points_r[in_border]
    points_c = points_c[in_border]
    point_feat = feats[points_r, points_c, :]
    point_feat = (point_feat.T / np.linalg.norm(point_feat + sys.float_info.epsilon, 2, axis=1).T).T

    return np.array([points_c, points_r]).T, point_feat

