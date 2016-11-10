# -*- coding: utf-8 -*-

'''
Perform Logistic Regression, SVM and feed-forward neural network
using raw pixel values on the test data
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

x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]
unfolded_data = x.reshape(100000,3600)

test = np.fromfile('test_x.bin', dtype='uint8')
test = test.reshape((20000,60,60))

x_flat = x.reshape(100000,3600)
test_flat = test.reshape(20000,3600)

x_train = unfolded_data
y_train = y
x_test = test_flat


"""
Logistic Regression
"""
logreg = linear_model.LogisticRegression(C=1e5)
print "Training Logistic Regression"
logreg.fit(x_train,y_train)
print "Testing Logistic Regression"
predicted_logreg = logreg.predict(x_test)
np.savetxt("predicted_logreg_raw_pixels.csv",predicted_logreg, delimiter =",")

"""
SVM
"""

sv = svm.SVC()
print "Training SVM"
sv.fit(x_train,y_train)
print "Testing SVM"
predicted_svm = sv.predict(x_test)
np.savetxt("predicted_svm_raw_pixels.csv",predicted_svm, delimiter =",")

"""
Neural Network
"""
print "Training Neural Network"
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,10), random_state=1, tol=0.0000000001, max_iter=100000)
clf.fit(x_train,y_train)
print "Testing Neural Network"
predicted_nn = clf.predict(x_test)
np.savetxt("predicted_nn_raw_pixels.csv",predicted_nn, delimiter =",")


