# Classify images using Bag of Words approach

Steps:
1. extract image features using SIFT
2. cluster features with nearest neighbour clustering
3. define code book using clusters
4. define images as histogram of cluster assigned SIFT features
5. classify image using k means nearest neighbour

intermediate results are stored in numpy arrays npy
uncomment intermediate steps to rerun with different hyper parameters
