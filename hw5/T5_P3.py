# CS 181, Spring 2020
# Homework 5: Principal Component Analysis

import numpy as np
import matplotlib.pyplot as plt

# This line loads the images for you. Don't change it!
pics = np.load("data/images.npy")

# Your code here. You may change anything below this line.
class PCA(object):
    # d is the number of principal components
    def __init__(self, d):
        self.d = d

    # X is a (N x 28 x 28) array where 28x28 is the dimensions of each of the N images. This method should apply PCA to a dataset X.
    def apply(self, X):
        pass


# Example of how to plot an image. We ask that image sin your writeup be grayscale images, just as in this example.
plt.figure()
plt.imshow(pics[0].reshape(28,28), cmap='Greys_r')
plt.show()
