# a SIFT implementation in python
# including some simple interest point detectors
# meant for educational purposes only.
#
# Feel free to distribute, yet, please keep my name so I can get bug reports
#
# Author: Jan van Gemert (j.c.vanGemert@uva.nl)
#


import numpy as np
import scipy.spatial
from pylab import *
import sys

import scipy.ndimage.filters as filters

def computeDerivs(imName, sigma=1.0):
    imRGB = imread(imName)
    imHSV = matplotlib.colors.rgb_to_hsv(imRGB)
    im = imHSV[:,:,2]
    # Derivatives
    dx = filters.gaussian_filter(im, sigma=sigma, order = [1,0]) 
    dy = filters.gaussian_filter(im, sigma=sigma, order = [0,1]) 
    return dx, dy


def gradAsHSV(imName, sigma=1.0):
    #print 'showGradasHSV'

    dx, dy = computeDerivs(imName,sigma=sigma)
    ang = np.arctan2(dy,dx)
    mag = sqrt( dx*dx + dy * dy )

    threshold = mag.max()/2
    mag[ mag > threshold ] = threshold

    # convert to  HSV
    hsv = np.zeros( (dx.shape[0], dx.shape[1], 3), dtype=np.float32 )
    hsv[...,2] = 1
    hsv[...,0] = (ang+np.pi) / (2*np.pi)
    hsv[...,1] = mag / mag.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb

def densePoints(imName, stride=5):
    #print 'dense sampling', 
    imRGB = imread(imName)
    [rows,cols,clrs] = imRGB.shape
    
    # divide 1 stride on both sides
    offsetR = rows % stride
    offsetC = cols % stride
    
    
    r = np.arange( offsetR/2, rows, stride)
    c = np.arange( offsetC/2, cols, stride)
    x, y = meshgrid(c, r) 
    grid = np.array( [x.flatten(), y.flatten()] ).T
    #print 'nr Points', grid.shape[0]
    return grid


def computeHes(imName, sigma=1, magThreshold = 10, hesThreshold=5, NMSneighborhood = 10):
    #print 'hessian'
    imRGB = imread(imName)
    
    # convert to grayscale
    imHSV = matplotlib.colors.rgb_to_hsv(imRGB)
    im = imHSV[:,:,2]
    
    # Derivatives
    dxx = filters.gaussian_filter(im, sigma=sigma, order = [2,0])
    dyy = filters.gaussian_filter(im, sigma=sigma, order = [0,2])
    lapl = sigma * (dxx + dyy)

    # non max suppression and thresholding of maxima
    data_max = filters.maximum_filter(lapl, NMSneighborhood)
    maxima = (lapl == data_max)
    maxima = logical_and( maxima, data_max > hesThreshold)

    # non max suppression and thresholding of minima
    data_min = filters.minimum_filter(lapl, NMSneighborhood)
    minima = (lapl == data_min)
    minima = logical_and( minima, data_min < -hesThreshold)

    extrema = logical_or(maxima, minima)

    dx = filters.gaussian_filter(im, sigma=sigma, order = [1,0])
    dy = filters.gaussian_filter(im, sigma=sigma, order = [0,1])
    mag = sigma * sqrt( dx*dx + dy * dy )
    extrema = logical_and( extrema, mag > magThreshold)

    #print 'Hes, nr Points', sum(extrema)

    [r,c] = np.where(extrema)

    return np.array([c,r]).T


def computeHar(imName, sigma=1, k=0.04, magThreshold = 10, NMSneighborhood=10):
    #print 'harris'
    
    dx, dy = computeDerivs(imName,sigma=sigma)

    #structure tensor
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # average with integration scale
    iScale=2*sigma
    dx2 = filters.gaussian_filter(dx2, sigma=iScale)
    dy2 = filters.gaussian_filter(dy2, sigma=iScale)
    dxy = filters.gaussian_filter(dxy, sigma=iScale)

    # cornerness, Det - k(Trace)^2
    R = dx2 * dy2 - dxy * dxy  - k * (dx2 + dy2) * (dx2 + dy2)

    # non-maximum suppression 
    data_max = filters.maximum_filter(R, NMSneighborhood)
    maxima = (R == data_max)
    maxima = logical_and( maxima, data_max > 0)

    # threshold magnitude
    mag = sigma * sqrt( dx*dx + dy * dy )
    maxima = logical_and( maxima, mag > magThreshold)

    #print 'Har, nr Points', sum(maxima)
    [r,c] = np.where(maxima)

    return np.array([c,r]).T


def computeSIFTofPoints(imName, points, sigma=1.0, nrOrientBins=8, nrSpatBins = 4, nrPixPerBin = 4):
    #print 'computeSIFTofPoints: %d points' % points.shape[0],

    dx, dy = computeDerivs(imName,sigma=sigma)
    #print dx.shape
    r, c = dx.shape
    #ang = np.arctan2(dy,dx)
    mag = sqrt( dx*dx + dy * dy )

    # put derivative vectors in an array, and add system precision 
    derivVecs = np.array( [ dx.flatten(), dy.flatten() ]).T + sys.float_info.epsilon

    # L2 normalize so dotprod is the cosine
    derivVecs = (derivVecs.T / np.linalg.norm(derivVecs,2,axis=1).T).T

    ## convert angles to vectors for easy dist-computation
    binCenters = np.arange(nrOrientBins) * (2*pi)/(nrOrientBins*2)
    binVec = np.array([ cos(binCenters), sin(binCenters) ])

    # compute cosine similarity
    angularsimMat =  np.dot(derivVecs, binVec)

    # invert, cause higher is better
    angularsimMat = 1.0 - angularsimMat

    # kernel density estimator of angular bins
    sigma = 0.1 * pi/nrOrientBins
    angularsimMat = exp(-(angularsimMat**2) / (2*(sigma**2)) )
    # normalize to sum to 1
    angularsimMat = (angularsimMat.T / np.linalg.norm(angularsimMat+ sys.float_info.epsilon,1,axis=1)).T

    # multiply with magnitude
    angularsimMat = (angularsimMat.T * mag.flatten()).T

    # reshape to 3D for the spatial aggregation of the histogram
    angularsimMat = angularsimMat.reshape( (r,c,nrOrientBins) )

    #spatial interpolation
    aggAng = filters.gaussian_filter(angularsimMat, sigma=[nrPixPerBin/3, nrPixPerBin/3, 0.1])

    nrPixPerBinDiv2 = ceil(nrPixPerBin / 2)
    featDim = nrSpatBins*nrSpatBins*nrOrientBins
    feats = np.zeros( (r-nrPixPerBin*nrSpatBins,c-nrPixPerBin*nrSpatBins,featDim) )
    idx = 0
    for i in np.arange(nrSpatBins):
        for j in np.arange(nrSpatBins):
            rBegin = i * nrPixPerBin + nrPixPerBinDiv2
            rEnd = r - nrPixPerBin  * (nrSpatBins-i-1) - nrPixPerBinDiv2
            cBegin = j * nrPixPerBin + nrPixPerBinDiv2
            cEnd = c - nrPixPerBin  * (nrSpatBins-j-1) - nrPixPerBinDiv2

            #print rBegin, rEnd, cBegin, cEnd
            feats[:,:,idx:nrOrientBins+idx] = aggAng[rBegin:rEnd, cBegin:cEnd, :]
            idx = idx + nrOrientBins
    #print feats[1:3,1:3,:]
    #print feats.shape

    
    pointsC, pointsR = points[:,0], points[:,1]
    inBorder = pointsR>nrPixPerBin*nrSpatBins
    inBorder = np.logical_and(inBorder, pointsR<r-nrPixPerBin*nrSpatBins)

    inBorder = np.logical_and(inBorder, pointsC>nrPixPerBin*nrSpatBins)
    inBorder = np.logical_and(inBorder, pointsC<c-nrPixPerBin*nrSpatBins)
    pointsR = pointsR[inBorder]
    pointsC = pointsC[inBorder]
    
    #print pointsR.shape
    #print pointsC.shape
    
    pointFeat = feats[pointsR, pointsC, :]
    pointFeat = (pointFeat.T / np.linalg.norm(pointFeat + sys.float_info.epsilon,2,axis=1).T).T
    #print pointFeat
    #print pointFeat.shape

    #print pointFeat[1:3,:]
    #figure()
    #imshow(pointFeat)
    #show()
    #print 'kept:', pointFeat.shape
    #sys.stdout.flush()

    return np.array([pointsC, pointsR]).T, pointFeat

