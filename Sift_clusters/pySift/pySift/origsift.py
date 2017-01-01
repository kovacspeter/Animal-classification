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
    dx = filters.gaussian_filter(im, sigma=sigma, order = [1,0]) + sys.float_info.epsilon
    dy = filters.gaussian_filter(im, sigma=sigma, order = [0,1]) + sys.float_info.epsilon
    return dx, dy


def computeHes(imName, sigma=1):
    #print 'hessian'
    imRGB = imread(imName)
    imHSV = matplotlib.colors.rgb_to_hsv(imRGB)
    im = imHSV[:,:,2]
    # Derivatives
    dxx = filters.gaussian_filter(im, sigma=sigma, order = [2,0])
    dyy = filters.gaussian_filter(im, sigma=sigma, order = [0,2])
    lapl = dxx + dyy

    threshold = 10

    neighborhood_size = 5
    data_max = filters.maximum_filter(lapl, neighborhood_size)
    maxima = (lapl == data_max)
    maxima = logical_and( maxima, data_max > threshold)

    data_min = filters.minimum_filter(lapl, neighborhood_size)
    minima = (lapl == data_min)
    minima = logical_and( minima, data_min < -threshold)

    extrema = logical_or(maxima, minima)

    magThreshold = 2
    magThreshold = 5
    dx = filters.gaussian_filter(im, sigma=sigma, order = [1,0])
    dy = filters.gaussian_filter(im, sigma=sigma, order = [0,1])
    mag = sqrt( dx*dx + dy * dy )
    extrema = logical_and( extrema, mag > magThreshold)

    print 'Hes, nr Points', sum(extrema)

    idx = np.where(extrema)

    return idx


def computeHarris(imName, fScale=1, iScale=2, k=0.04):
    print 'harris'
    dx, dy = computeDerivs(imName,sigma=fScale)

    #structure tensor
    dx2 = dx * dx
    dy2 = dy * dy
    dxy = dx * dy

    # average with integration scale
    dx2 = filters.gaussian_filter(dx2, sigma=iScale)
    dy2 = filters.gaussian_filter(dy2, sigma=iScale)
    dxy = filters.gaussian_filter(dxy, sigma=iScale)

    # cornerness, Det – k(Trace)^2
    R = dx2 * dy2 - dxy * dxy  - k * (dx2 + dy2) * (dx2 + dy2)

    neighborhood_size = 3
    data_max = filters.maximum_filter(R, neighborhood_size)
    maxima = (R == data_max)
    maxima = logical_and( maxima, data_max > 0)

    magThreshold = 2
    mag = sqrt( dx*dx + dy * dy )
    maxima = logical_and( maxima, mag > magThreshold)

    print 'Har, nr Points', sum(maxima)
    idx = np.where(maxima)

    return idx




def gradAsHSV(imName):
    print 'showGradasHSV'

    dx, dy = computeDerivs(imName,sigma=1)
    ang = np.arctan2(dy,dx)
    mag = sqrt( dx*dx + dy * dy )

    threshold = mag.max()/2
    mag[ mag > threshold ] = threshold

    # convert to  HSV
    hsv = np.zeros( (dx.shape[0], dx.shape[1], 3), dtype=np.float32 )
    hsv[...,2] = 1
    hsv[...,0] = (ang+np.pi) / (2*np.pi)
    #hsv[...,1] = (flowMag - flowMag.min() ) / (flowMag.max() - flowMag.min() )
    hsv[...,1] = mag / mag.max()
    #hsv[...,1] = 1
    print mag.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    return rgb


def im2angleOrient(imName, nrBins=18):
    print 'im2angleOrient'

    dx, dy = computeDerivs(imName,sigma=1)
    #print dx.shape
    r, c = dx.shape
    #ang = np.arctan2(dy,dx)
    mag = sqrt( dx*dx + dy * dy )

    # keep original vectors
    derivVecs = np.array( [ dx.flatten(), dy.flatten() ]).T

    # L2 normalize so dotprod is the cosine
    derivVecs = (derivVecs.T / np.linalg.norm(derivVecs,2,axis=1).T).T

    ## convert angles to vectors for easy dist-computation
    binCenters = np.arange(nrBins) * (2*pi)/(nrBins*2)
    binVec = np.array([ cos(binCenters), sin(binCenters) ])

    # compute cosine similarity
    angularsimMat =  np.dot(derivVecs, binVec)

    # invert, cause higher is better
    angularsimMat = 1.0 - angularsimMat

    # kernel density estimator
    sigma = 0.1 * pi/nrBins
    angularsimMat = exp(-(angularsimMat**2) / (2*(sigma**2)) )
    # normalize to sum to 1
    angularsimMat = (angularsimMat.T / np.linalg.norm(angularsimMat,1,axis=1)).T

    # multiply with magnitude
    angularsimMat = (angularsimMat.T * mag.flatten()).T

    # reshape to 3D for the spatial aggregation of the histogram
    angularsimMat = angularsimMat.reshape( (r,c,nrBins) )
    aggAng = filters.gaussian_filter(angularsimMat, sigma=[1.5,1.5, 0.1])

    # find the max response
    maxBinIdx = np.argmax( aggAng, axis=2)
    magBin = np.max( aggAng, axis=2)


    angleIm = binCenters[ maxBinIdx ]

    hsv = np.zeros( (r,c, 3), dtype=np.float32 )
    hsv[...,2] = 1
    hsv[...,0] = (angleIm / pi)
    #hsv[...,1] = (flowMag - flowMag.min() ) / (flowMag.max() - flowMag.min() )
    #threshold = mag.max()/2
    #mag[ mag > threshold ] = threshold
    #hsv[...,1] = mag / mag.max()
    hsv[...,1] = magBin / magBin.max()
    #hsv[...,1] = 1



    #print mag.max()
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    #figure()
    #imshow(rgb)
    #show()
    return rgb


def computeSIFT(imName, nrOrientBins=8, nrSpatBins = 4, nrPixPerBin = 4, sigma=1.0):
    print 'computeSIFT'

    dx, dy = computeDerivs(imName,sigma=sigma)
    #print dx.shape
    r, c = dx.shape
    #ang = np.arctan2(dy,dx)
    mag = sqrt( dx*dx + dy * dy )

    # put derivative vectors in an array
    derivVecs = np.array( [ dx.flatten(), dy.flatten() ]).T

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
    angularsimMat = (angularsimMat.T / np.linalg.norm(angularsimMat,1,axis=1)).T

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


    [hesR, hesC] = computeHes(imName)
    inBorder = hesR>nrPixPerBin*nrSpatBins
    inBorder = np.logical_and(inBorder, hesR<r-nrPixPerBin*nrSpatBins)

    inBorder = np.logical_and(inBorder, hesC>nrPixPerBin*nrSpatBins)
    inBorder = np.logical_and(inBorder, hesC<c-nrPixPerBin*nrSpatBins)
    hesR = hesR[inBorder]
    hesC = hesC[inBorder]
    #print hesR.shape
    #print hesC.shape
    pointFeat = feats[hesR, hesC, :]
    pointFeat = (pointFeat.T / np.linalg.norm(pointFeat,2,axis=1).T).T
    #print pointFeat
    #print pointFeat.shape

    #print pointFeat[1:3,:]
    #figure()
    #imshow(pointFeat)
    #show()
    print pointFeat.shape
    sys.stdout.flush()

    return np.array([hesC, hesR]).T, pointFeat


