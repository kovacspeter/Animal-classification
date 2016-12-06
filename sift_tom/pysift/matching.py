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

def simpleMatch(sift1, sift2):
    simMat = scipy.spatial.distance.cdist( sift1, sift2 )
    match1to2 = np.argmin(simMat, axis=1)
    matchTuples = zip(xrange(len(match1to2)), match1to2 )
    return matchTuples, match1to2

def visualiseMatchesVER(imName1, imName2, matchTuples, matchIdx, pMat1, pMat2, plottitle=''):
    #print imName1, imName2
    im1 = imread(imName1)
    im2 = imread(imName2)
    h1,w1,c1 = im1.shape
    h2,w2,c2 = im2.shape
    # on top of each other
    imW = max(w1,w2)
    imH = h1+h2
    im = np.ones( (imH, imW, 3), dtype=uint8 )
    #print im.shape
    #figure(figsize=( imW/50.0, imH/50.0) )
    figure()
    title(plottitle)
    im[:h1,:w1,:] = im1
    im[h1:,:w2,:]=im2
    imshow(im)
    for i in matchIdx:
        match = matchTuples[i]
        m1 = match[0]
        m2 = match[1]
        p1 = pMat1[m1,:]
        p2 = pMat2[m2,:]
        x = [ p1[0], p2[0] ]
        y = [ p1[1], p2[1] + h1 ]
        plot( x,y , 'o-')
    #plot(pMat[:,0], pMat[:,1], 'b.')
    show()



def computeHomography(p1, p2):
    #print 'compute homography'
    nr = p1.shape[0]

    # get vectors
    x1 = p1[:,0].reshape(nr,1)
    y1 = p1[:,1].reshape(nr,1)
    x2 = p2[:,0].reshape(nr,1)
    y2 = p2[:,1].reshape(nr,1)

    one = np.ones( (nr,1) )
    zer = np.zeros( (nr,1) )

    #print x1
    #A = np.zeros( (nr*2, 8) )
    #A[0:nr,:] = np.concatenate( (x1, y1, one, zer, zer, zer, -x1*x2, -y1*x2), axis=1 )
    #A[nr:,:]  = np.concatenate( (zer, zer, zer, x1, y1, one, -x1*y2, -y1*y2), axis=1 )
    A = np.zeros( (nr*2, 6) )
    A[0:nr,:] = np.concatenate( (x1, zer, one, zer, zer, zer), axis=1 )
    A[nr:,:]  = np.concatenate( (zer, zer, zer, zer, y1, one, ), axis=1 )

    b = np.concatenate( (x2, y2), axis=0)

    #if nr == 3:
    #    print 'solve'
    #    H = ones( 9 )
    #    H[0:8] = np.linalg.solve(A,b).T
    #    H = H.reshape( (3,3) )
    #else:
    #print 'lstsq'
    H = np.zeros( 9 )
    H[0:6] = np.linalg.lstsq(A,b)[0].T
    #H[0:8] = np.linalg.lstsq(A,b)[0].T
    H[8] = 1
    #print solution, residuals
    #H[0:8] = solution.T
    H = H.reshape( (3,3) )

    p1h = np.concatenate( (p1, np.ones( (p1.shape[0],1) ) ), axis=1)
    cp1h = np.dot(H,p1h.T)
    cp1 = cp1h / cp1h[2,:]
    cp1 = cp1[0:2,:]

    #print 'orig\n', p2.T
    #print 'proj\n', cp1
    #print H
    #sys.exit()
    return H, cp1.T

def getReprojectionError(H1to2, match1to2Sorted, match1to2Val, pMat1, pMat2, minDist=10):
    print 'compute projError'
    print H1to2
    p1h = np.concatenate( (pMat1[:,0:2], np.ones( (pMat1.shape[0],1) ) ), axis=1)
    p2h = np.concatenate( (pMat2[:,0:2], np.ones( (pMat2.shape[0],1) ) ), axis=1)
    #print p1h.shape
    cp1h = np.dot(H1to2,p1h.T)
    #print cp1h
    cp1 = cp1h / cp1h[2,:]
    cp1 = cp1[0:2,:]
    #print 'projected\n', cp1
    m2 = pMat2[match1to2Sorted]
    #print 'matched\n', m2[:,0:2].T
    cp1minm2 = cp1 - m2[:,0:2].T
    #print 'min\n', cp1minm2
    dists = np.linalg.norm(cp1minm2, axis=0)
    #print 'dists\n', dists
    lowDists = np.nonzero(dists < minDist)
    #print 'lowDistIdx', lowDists
    #print 'lowDists\n', dists[lowDists]
    #sortDistsIdx = np.argsort( dists )c
    #print sortDistsIdx
    #print dists[sortDistsIdx[:500]]
    #
    #figure()
    #plot( match1to2Val[lowDists], dists[lowDists], '.')
    #show()
    return dists, lowDists[0]

def getPointsFromMatches(matchTuples, nrMatchIDs, pMat1, pMat2):
    p1 = []
    p2 = []
    for m in nrMatchIDs:
        m1 = matchTuples[m][0]
        m2 = matchTuples[m][1]
        p1.append( [pMat1[m1][0], pMat1[m1][1]] )
        p2.append( [pMat2[m2][0], pMat2[m2][1]] )
    #print p1
    #print p2
    return np.array(p1), np.array(p2)

def projectPoints(H1to2,p1):
    #print 'p1\n', p1

    p1h = np.concatenate( (p1[:,0:2], np.ones( (p1.shape[0],1) ) ), axis=1)
    #print 'p1h\n', p1h
    #print p1h.shape

    cp1h = np.dot(H1to2,p1h.T)
    #print cp1h
    cp1 = cp1h / cp1h[2,:]
    #print 'projected\n', cp1
    cp1 = cp1[0:2,:]
    #print 'projected\n', cp1
    return cp1

def getPixelDistsFromPoints(H1to2, p1, p2):

    cp1 = projectPoints(H1to2,p1)

    #print 'matched\n', p2.T
    cp1minm2 = cp1 - p2.T
    #sys.exit()
    dists = np.linalg.norm(cp1minm2, axis=0)
    return dists

def getNormInlierPixelDistsFromPoints(H1to2, p1, p2, minDist):

    cp1 = projectPoints(H1to2,p1)

    #print cp1

    #print 'matched\n', p2.T
    cp1minm2 = cp1 - p2.T

    #print H1to2
    #print 'min\n', cp1minm2
    # normalize scale of x and y to the projection
    #cp1minm2[0,:] =  cp1minm2[0,:] / H1to2[0,0]
    #cp1minm2[1,:] =  cp1minm2[1,:] / H1to2[1,1]

    #print 'minDistW', minDistW, 'minDistH', minDistH
    ## normalize scale of x and y to the bounding box ratio
    #cp1minm2[0,:] = cp1minm2[0,:] / minDistW
    #cp1minm2[1,:] = cp1minm2[1,:] / minDistH

    #print 'minNorm\n', cp1minm2

    #sys.exit()
    dists = np.linalg.norm(cp1minm2, axis=0)

    matchIDs = np.nonzero(dists < minDist)[0]

    return matchIDs


def calcInliers(H1to2, matchTuples, pMat1, pMat2, minDist):
    #print 'calcInliers'
    #print 'pMat1\n', pMat1
    p1, p2 = getPointsFromMatches(matchTuples, xrange(len(matchTuples)), pMat1, pMat2)
    #print 'p1\n', p1
    #print 'p2\n', p2

    #dists = getPixelDistsFromPoints(H1to2, p1, p2)
    #matchIDs = np.nonzero(dists < minDist)[0]

    matchIDs = getNormInlierPixelDistsFromPoints(H1to2, p1, p2, minDist)

    nrInliers = len(matchIDs)
    #print 'nrInliers', nrInliers
    return nrInliers, matchIDs



def homographyOK(H):
    #print H
    if H[0,0] < 0.2 or H[0,0] > 5:
        return False
    if H[1,1] < 0.2 or H[1,1] > 5:
        return False
    if H[1,1]/H[0,0] < 0.2 or H[1,1]/H[0,0] > 5:
        return False

    return True

def ransac(matchTuples, matchIDs, pMat1, pMat2, minPixDistance=10, nrIter=1000, doInnerLoop=True ):
    #print 'ransac'
    sys.stdout.flush()

    nrMatch = 4
    maxInliers = 0
    
    bestH = np.array([[1, 0, 0], [0,1,0], [0,0,1]])
    inlierMatchIDs = []
    #matchIDs = np.arange(len(matchTuples))
    i = 0
    while i <nrIter:

        #print i
        if i % 100 == 0:
            print i,
            sys.stdout.flush()

        # randomize
        np.random.shuffle(matchIDs)
        # pick random matches
        p1,p2 = getPointsFromMatches(matchTuples, matchIDs[0:nrMatch], pMat1, pMat2)
        H1to2, _ = computeHomography(p1, p2)
        #print H1to2
        i+=1
        if homographyOK(H1to2):
            #i+=1

            # TODO: fix distance measurements after projection

            nrInliers, inlierIDs = calcInliers(H1to2, matchTuples, pMat1, pMat2, minPixDistance)
            
            if nrInliers > maxInliers:
                # LO-ransac: run it again on the inliers
                if doInnerLoop:
                    inlierIDs = ransac(matchTuples, inlierIDs, pMat1, pMat2, minPixDistance, 10, False)
                    p1,p2 = getPointsFromMatches(matchTuples, inlierIDs, pMat1, pMat2)
                    H1to2, _ = computeHomography(p1, p2)
                    nrInliers, inlierIDs = calcInliers(H1to2, matchTuples, pMat1, pMat2, minPixDistance)
                maxInliers = nrInliers
                bestH = H1to2
                inlierMatchIDs = inlierIDs

    #print 'maxInliers', maxInliers
    #print bestH
    return inlierMatchIDs


