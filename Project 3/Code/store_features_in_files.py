# -*- coding: utf-8 -*-

'''
This script flattens the image out, computer HOG and Daisy features
for both train and test data
'''

"""
Training data:
1. Flatten image out
2. Daisy Features
3. HoG Features

Test data:
1. Flatten image out
2. Daisy Features
3. HoG Features

"""


import sklearn
import numpy as np
import scipy.misc # to visualize only
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import svm, linear_model, naive_bayes
from sklearn import metrics
import math
import matplotlib.pyplot as plt
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.feature import daisy


x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]

test = np.fromfile('test_x.bin', dtype='uint8')
test = test.reshape((20000,60,60))

x_flat = x.reshape(100000,3600)
test_flat = test.reshape(20000,3600)

"""
Flatten Image Out
"""

np.savetxt("train_flattened_out.csv",x_flat, delimiter =",")
np.savetxt("test_flattened_out.csv",test_flat, delimiter =",")


"""
Daisy Features
"""

print "Daisy: Saving features' loop for train"
daisy_features_train_set = np.zeros((len(x),104))
for i in range(len(x)):
    descs, descs_img = daisy(x[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_train_set[i] = descs.reshape((1,104))

print daisy_features_train_set.shape

np.savetxt("train_daisy.csv",daisy_features_train_set, delimiter =",")



print "Daisy: Saving features' loop for test"
daisy_features_test_set = np.zeros((len(test),104))
for i in range(len(test)):
    descs, descs_img = daisy(test[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_test_set[i] = descs.reshape((1,104))

print daisy_features_test_set.shape

np.savetxt("test_daisy.csv",daisy_features_test_set, delimiter =",")


"""
HoG Features

"""

print "HoG: Saving features' loop for train"
hog_features_train_set = np.zeros((len(x),3600))
for i in range(len(x)):
    fd, hog_image = hog(x[i], orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    hog_features_train_set[i] = fd.reshape((1,3600))

print hog_features_train_set.shape

np.savetxt("train_hog.csv",hog_features_train_set, delimiter =",")



print "HoG: Saving features' loop for test"
hog_features_test_set = np.zeros((len(test),3600))
for i in range(len(test)):
    fd, hog_image = hog(test[i], orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    hog_features_test_set[i] = fd.reshape((1,3600))

print hog_features_test_set.shape

np.savetxt("test_hog.csv",hog_features_test_set, delimiter =",")









