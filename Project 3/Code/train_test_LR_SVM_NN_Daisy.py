# -*- coding: utf-8 -*-

'''
Perform Logistic Regression, SVM and feed-forward neural network
using Daisy features on the test data
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

print "Daisy: Saving features' loop for train"
daisy_features_train_set = np.zeros((len(x),104))
for i in range(len(x)):
    descs, descs_img = daisy(x[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_train_set[i] = descs.reshape((1,104))
    
print "Daisy: Saving features' loop for test"
daisy_features_test_set = np.zeros((len(test),104))
for i in range(len(test)):
    descs, descs_img = daisy(test[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_test_set[i] = descs.reshape((1,104))
 

x_train = daisy_features_train_set
y_train = y
x_test = daisy_features_test_set

"""
Logistic Regression
"""

logreg = linear_model.LogisticRegression(C=1e5)
print "Training Logistic Regression"
logreg.fit(x_train,y_train)
print "Testing Logistic Regression"
predicted_logreg = logreg.predict(x_test)
np.savetxt("predicted_logreg_daisy.csv",predicted_logreg, delimiter =",")

"""
SVM
"""

sv = svm.SVC()
print "Training SVM"
sv.fit(x_train,y_train)
print "Testing SVM"
predicted_svm = sv.predict(x_test)
savetxt("predicted_logreg_daisy.csv",predicted_svm, delimiter =",")

"""
Neural Network
"""
print "Training Neural Network"
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,10), random_state=1, tol=0.0000000001, max_iter=10000)
clf.fit(x_train,y_train)
print "Testing Neural Network"
predicted_nn = clf.predict(x_test)
savetxt("predicted_logreg_daisy.csv",predicted_nn, delimiter =",")

