import numpy as np
import pandas as pd
import pylab as pl
import os
from PIL import Image
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# Setup a standard image size
STANDARD_SIZE = (256, 256)


def img_to_matrix(filename, verbose=False):
    """ Takes a filename and turns it into a numpy array of RGB pixels"""
    img = Image.open(filename)
    
    if verbose:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img


def flatten_image(img):
    """Takes in an (m, n) numpy array and flattens it
        into an array of shape (1, m * n)"""
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    
    return img_wide[0]


""" Define training image locations"""
img_dir = "../data/dataset/trainset/"
sub_dirs = [img_dir + d for d in os.listdir(img_dir) if not d.startswith('.')]
images = [d + '/' + f for d in sub_dirs for f in os.listdir(d)
          if not f.startswith('.')]
labels = [f for f in os.listdir(img_dir) if not f.startswith('.')]

""" Import training data and set labels"""
data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    data.append(img)

train_x = np.array(data)
train_y = np.repeat(labels, 100)


""" Define test image locations"""
img_dir = "data/dataset/validationset/"
sub_dirs = [img_dir+ d for d in os.listdir(img_dir) if not d.startswith('.')]
images = [d + '/' + f for d in sub_dirs for f in os.listdir(d)
          if not f.startswith('.')]
labels = [f for f in os.listdir(img_dir) if not f.startswith('.')]

""" Import test data and set labels"""
val_data = []
for image in images:
    img = img_to_matrix(image)
    img = flatten_image(img)
    val_data.append(img)

test_x = np.array(val_data)
test_y = np.repeat(labels, 20)


""" Reducing features with Randomized PCA"""
pca = PCA(svd_solver='randomized', n_components=5)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)


""" K-Nearest Neighbors"""
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_x, train_y)
print "Accuracy is: ", knn.score(test_x, test_y)
# pd.crosstab(test_y, knn.predict(test_x), rownames=["Actual"], 
# colnames=["Predicted"])


""" Gradient Boosting"""
clf = GradientBoostingClassifier(learning_rate=0.03, n_estimators=100)
clf.fit(train_x, train_y)
print "Accuracy is: ", clf.score(test_x, test_y)
# pd.crosstab(test_y, clf.predict(test_x), rownames=["Actual"], 
# 			colnames=["Predicted"])


""" Ada Boost Classifier"""
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
bdt.fit(train_x, train_y)
print "Accuracy is: ", bdt.score(test_x, test_y)
# pd.crosstab(test_y, bdt.predict(test_x), rownames=["Actual"], 
# 			colnames=["Predicted"])


"""Neural network model with backpropagation"""
clf = MLPClassifier(alpha=1, hidden_layer_sizes=(300, ))
clf.fit(train_x, train_y)
print "Accuracy is: ", clf.score(test_x, test_y)
# pd.crosstab(test_y, clf.predict(test_x), rownames=["Actual"],
# 			colnames=["Predicted"])

