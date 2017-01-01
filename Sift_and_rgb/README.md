# Classify images using Bag of Words approach

Steps:
1. extract image features using SIFT
2. cluster features with k means clustering (k=100)
3. define a code book using clusters
4. for each image, assign code book clusters to each feature and create a cluster histogram
5. classify image using k nearest neighbour using the histogram as feature (k=10)

intermediate results are stored in binary numpy arrays (.npy) and can be reused

