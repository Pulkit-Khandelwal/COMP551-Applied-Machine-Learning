# -*- coding: utf-8 -*-


'''
Perform Logistic Regression, SVM and feed-forward neural network
using HOG features on the test data
'''

import sklearn
import numpy as np
import scipy.misc # to visualize only
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.feature_selection import SelectKBest
from sklearn.neural_network import MLPClassifier
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
 
print "HoG: Saving features' loop for train"
hog_features_train_set = np.zeros((len(x),3600))
for i in range(len(x)):
    fd, hog_image = hog(x[i], orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    hog_features_train_set[i] = fd.reshape((1,3600))

print "HoG: Saving features' loop for test"
hog_features_test_set = np.zeros((len(test),3600))
for i in range(len(test)):
    fd, hog_image = hog(test[i], orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    hog_features_test_set[i] = fd.reshape((1,3600))

x_train = hog_features_train_set
y_train = y
x_test = hog_features_test_set

'''
"""
Logistic Regression
"""

logreg = linear_model.LogisticRegression(C=1e5)
print "Training Logistic Regression"
logreg.fit(x_train,y_train)
print "Testing Logistic Regression"
predicted_logreg = logreg.predict(x_test)
np.savetxt("predicted_logreg_hog.csv",predicted_logreg, delimiter =",")

"""
SVM
"""

sv = svm.SVC()
print "Training SVM"
sv.fit(x_train,y_train)
print "Testing SVM"
predicted_svm = sv.predict(x_test)
np.savetxt("predicted_svm_hog.csv",predicted_svm, delimiter =",")
'''

"""
Neural Network
"""
print "Training Neural Network"
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(5,5), random_state=1, tol=0.000000001, max_iter=1000)
clf.fit(x_train,y_train)
print "Testing Neural Network"
predicted_nn = clf.predict(x_test)
np.savetxt("predicted_nn_hog.csv",predicted_nn, delimiter =",")

