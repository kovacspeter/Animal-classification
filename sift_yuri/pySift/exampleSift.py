from pylab import *
from pySift import sift, matching

# to show some visualisations, yes=True, no=False
doVisualisations = True

sigma = 1.0

imNames = ['mila1.jpg', 'mila2.jpg']
#imNames = ['dam1.jpg', 'dam2.jpg'] # slightly bigger images (ie: slower)

pointsList = []

# Get the interest points for the two images.
for imName in imNames:

    # dense sampling
    denPoints = sift.densePoints(imName, stride=15)
    
    # compute interesting points (hessian)
    hesPoints = sift.computeHes(imName, sigma, magThreshold = 10, hesThreshold=5, NMSneighborhood = 10)
    
    # compute interesting points (harris)
    harPoints = sift.computeHar(imName, sigma, magThreshold = 10, NMSneighborhood=10)
    
    # add these harris and hessian interest points together
    points = np.concatenate( (hesPoints, harPoints) )
    # store points in a list
    pointsList.append( points )

    # if wanted, show color coded interest points
    if doVisualisations:
        figure()
        # display image
        imRGB = imread(imName)
        imshow(imRGB)
        # dense 
        plot( denPoints[:,0], denPoints[:,1], 'go', ms=4)
        # hessian 
        plot( hesPoints[:,0], hesPoints[:,1], 'bo', ms=4)
        # harris 
        plot( harPoints[:,0], harPoints[:,1], 'ro', ms=4)
        # show the result
        show()


# get the image names
imName1 = imNames[0]
imName2= imNames[1]

# get the interest points for the images
points1 = pointsList[0] 
points2 = pointsList[1]


# compute SIFT for each interest point
print points1.shape
points1, sift1 = sift.computeSIFTofPoints(imName1, points1, sigma, nrOrientBins=8, nrSpatBins = 4, nrPixPerBin = 4)
points2, sift2 = sift.computeSIFTofPoints(imName2, points2, sigma, nrOrientBins=8, nrSpatBins = 4, nrPixPerBin = 4)
print points1.shape, sift1.shape
print np.sum(sift1[0,:])

# do a simple matching
matchTuples, match1to2 = matching.simpleMatch(sift1, sift2)
# show the simple matching
#matching.visualiseMatchesVER(imName1, imName2, matchTuples, range(len(match1to2)), points1, points2, "All best matches")

# do spatial verification
inlierMatchIDs = matching.ransac(matchTuples, range(len(match1to2)), points1, points2, minPixDistance=20, nrIter=1000)
print 'nrInliers', len(inlierMatchIDs)

if doVisualisations:
    # show the spatial verified matches
    matching.visualiseMatchesVER(imName1, imName2, matchTuples, inlierMatchIDs, points1, points2, "After spatial verification")





