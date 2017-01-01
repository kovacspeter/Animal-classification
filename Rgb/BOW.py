import os
import numpy as np
import scipy.misc
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import k_means


""" Define training image locations"""
img_dir = "../data/dataset/trainset/"
sub_dirs = [img_dir + d for d in os.listdir(img_dir) if not d.startswith('.')]
images = [d + '/' + f for d in sub_dirs for f in os.listdir(d) 
          if not f.startswith('.')]
labels = [f for f in os.listdir(img_dir) if not f.startswith('.')]


# Cluster a set of features into k clusters. note: use mini batch, it's faster
def cluster_data(features, k, nr_iter=25):
    centroids = k_means(features, n_clusters=k, max_iter=nr_iter)[0]
    return centroids


# This function takes as input an image and how many patches to return
def patch_vectors(img, patch_nr, patch_size=(3, 3)):
    # We define some useful values
    patch_h, patch_w = patch_size
    
    # We define the output array with the expected size
    output = np.zeros((patch_nr, patch_h*patch_w*3))
    
    patches = extract_patches_2d(img, patch_size, patch_nr)  # randomly extract [patch_nr] of patches size [patch_size]
    output[:, :] = np.reshape(patches, (patch_nr, np.prod(np.shape(patches)[1:])))[:, :]
    
    return output


# This function will take a list of image file names and the desired number of clusters,
# Extract patches from each image using the patch_vectors function, then perform clustering
# On the patches from all the input images. These clusters will be returned as a (k x N)
# array, where k is the number of clusters, and N is the length of the vectors.
def patch_clusters(image_filenames, nr_clusters, nr_patches_per_img=200, patch_size=(3, 3)):
    
    # Again, we define helpful values:
    patch_h, patch_w = patch_size
    
    # We also define a helpful intermediary array for storing all the
    # patches from the input images.
    patches = np.zeros((nr_patches_per_img*len(image_filenames), np.prod(patch_size)*3))
    
    # For each image, we fill in the patches array with N patch vectors
    for i, each_filename in enumerate(image_filenames):
        img = scipy.misc.imread(each_filename)
        idx = i*nr_patches_per_img
        patches[idx:idx+nr_patches_per_img, ...] = patch_vectors(img, nr_patches_per_img, patch_size)

    cluster_output = cluster_data(patches, nr_clusters)

    return cluster_output


# We will cluster patches extracted from the first 5 images in each Desert/Boat/forest series
image_filenames = images

# We set the number of clusters to 32.
cl = patch_clusters(image_filenames, 32)


# A function to generate histograms for an image
def make_histogram_from_image(imagepath, cl):
    img = scipy.misc.imread(imagepath)
    test_patches = extract_patches_2d(img, (3, 3), max_patches=500)
    assert len(test_patches) == 500
    test_patches = np.reshape(test_patches, (np.shape(test_patches)[0], np.prod(np.shape(test_patches)[1:])))
    
    return create_histogram(test_patches, cl)


# Create a bag-of-words histogram from a set of features given the clusters. tip: use numpy histogram for speed
def create_histogram(samples, clusters):
    assignments = cluster_assignment(samples, clusters)
    histogram = np.zeros(clusters.shape[0], dtype=np.float)
    
    # Go over all the assignments and place them in the correct histogram bin.
    for i in range(len(assignments)):
        histogram[assignments[i]] += 1
        
    # Normalize the histogram such that the sum of the bins is equal to 1.
    for j in range(len(histogram)):
        histogram[j] /= float(len(assignments))
    
    return histogram


# For each sample, find the nearest cluster.
def cluster_assignment(samples, clusters):
    nr_samples = samples.shape[0]
    assignments = np.zeros(nr_samples, dtype=int)
    
    # For each data sample, compute the distance to each cluster.
    # Assign each sample to the cluster with the smallest distance.
    for i in range(len(samples)):
        dists = distances(samples[i], clusters)
        idx = np.where(dists == np.min(dists))[0][0]
        assignments[i] = idx

    return assignments


# Compute the Euclidean distance between 2 features. tip: use numpy euclidean distance for speed
def euclidean_distance(x, y):
    assert(len(x) == len(y))
    
    d = 0.0
    nr_dimensions = len(x)
    
    # Compute the distance value.
    for i in range(nr_dimensions):
        d += (x[i]-y[i])**2

    return np.sqrt(d)


def distances(a, X, distance_fn=euclidean_distance):
    dists = np.zeros(X.shape[0])
    
    for i in range(X.shape[0]):
        dists[i] = distance_fn(a, X[i])

    return dists


# Put all histograms of a certain class in a dictionary:
hs = dict((label, []) for label in labels)
hsk = hs.keys()

# For each image from 1 t/m 5
for i in xrange(1, 6):
    # For each series label
    for j, s in enumerate(hsk, start=1):
        h = make_histogram_from_image(img_dir+s+"/"+s+"_"+str(i).zfill(4)+'.jpg', cl)
        hs[s].append(h)


def nn_classifier(test_X, train_X, train_y):
    predictions = np.zeros(test_X.shape[0])

    for i in range(len(test_X)):
        dists = distances(test_X[i], train_X)
        sorted_zip = sorted(zip(dists, train_y))
        predictions[i] = sorted_zip[0][1]

    return predictions


def compute_accuracy(predictions, truth):
    return np.float_(np.sum(predictions == truth)) / truth.shape[0]


def knn(test_x, train_x, train_y, k):
    predictions = []
    for i in range(len(test_x)):
        # Initialize the predicted label to nonsense class label
        prediction = -10

        # Compute the Euclidean distance between data_point and each element in train_data.
        # Find the k train_data elements with the smallest distance.
        # Set the prediction as the most often occurring label among the k nearest neighbours.
        dists = distances(test_x[i], train_x)
        sorted_zip = sorted(zip(dists, train_y))
        top_k = zip(*sorted_zip[:k])[1]
        prediction = max(set(top_k), key=top_k.count)
        predictions.append(prediction)
        
    return predictions


# Predict the distance-weighted label of a data point
def weighted_nn(test_x, train_x, train_y):
    predictions = []
    for i in range(len(test_x)):
        prediction = -1
        label_score = 0.0

        # Go through all the training points and calculate the weights
        # Then see which class has the highest score based on these weights
        weights = np.zeros(len(train_x))
        dists = distances(test_x[i], train_x)
        sums = {}
        for j in range(len(weights)):
            weights[j] = 1 / dists[j]**2
            if train_y[j] in sums:
                sums[train_y[j]] += weights[j]
            else:
                sums[train_y[j]] = weights[j]

        # divide max of weight by total sum of weights
        label_score = sums[max(sums, key=sums.get)] / sum(sums.itervalues())
        prediction = max(sums, key=sums.get)
        predictions.append(prediction)
        
    return predictions


names = labels
# Concatenate training data:
X = []
y = []
for k in hs.keys():
    for v in hs[k]:
        X.append(v)
        y.append(names.index(k))
X = np.array(X)
y = np.array(y)

p = []
t = []
print 'Actual'.rjust(10)+'Predicted'.rjust(20)+'\n-----------------------------------'
for s in names:
    for i in xrange(1, 21):
        h = make_histogram_from_image(img_dir+s+"/"+s+"_"+str(i).zfill(4)+'.jpg', cl)
        h = np.reshape(h, (1, len(h)))
        # p.append(np.int_(nn_classifier(h, X, y)[0]))  # NN=1
        # p.append(np.int_(weighted_nn(h, X, y)[0]))  # WEIGHTED NN
        p.append(np.int_(knn(h, X, y, 10)[0]))  # KNN
        t.append(names.index(s))
        print s.rjust(10)+names[p[-1]].rjust(20)
print '\nOverall Accuracy:', compute_accuracy(np.array(p), np.array(t))
for i in xrange(10):
    print '\n'+names[i].upper().rjust(7)+' Accuracy:', compute_accuracy(np.array(p[i*5:(i*5)+5]),
                                                                        np.array(t[i*5:(i*5)+5]))
