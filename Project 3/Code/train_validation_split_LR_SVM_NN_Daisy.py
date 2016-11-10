# -*- coding: utf-8 -*-

'''
Perform Logistic Regression, SVM and feed-forward neural network
using Daisy features on the validation set
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
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.feature import daisy

x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]
unfolded_data = x.reshape(100000,3600)


print "Daisy: Saving features' loop for train"
daisy_features_train_set = np.zeros((len(x),104))
for i in range(len(x)):
    descs, descs_img = daisy(x[i], step=180, radius=20, rings=2, histograms=6,
                         orientations=8, visualize=True)
    daisy_features_train_set[i] = descs.reshape((1,104))


x_train = daisy_features_train_set[:80000]
y_train = y[:80000]
x_test = daisy_features_train_set[80000:]
y_test = y[80000:]

"""

"""
Logistic Regression
"""

logreg = linear_model.LogisticRegression(C=1e5)
print "Training Logistic Regression"
logreg.fit(x_train,y_train)
expected= y_test
print "Testing Logistic Regression"
predicted_logreg = logreg.predict(x_test)

print "Logistic Regression Classification Report",(metrics.classification_report(expected, predicted_logreg))
print "Logistic Regression Confusion Matrix",(metrics.confusion_matrix(expected, predicted_logreg))
print "Logistic Regression Accuracy",(metrics.accuracy_score(expected, predicted_logreg))


"""
SVM
"""

sv = svm.SVC()
print "Training SVM"
sv.fit(x_train,y_train)
expected = y_test
print "Testing SVM"
predicted_svm = sv.predict(x_test)

print "SVM Classification Report",(metrics.classification_report(expected, predicted_svm))
print "SVM Confusion Matrix",(metrics.confusion_matrix(expected, predicted_svm))
print "SVM Accuracy",(metrics.accuracy_score(expected, predicted_svm))

"""

"""
Neural Network
"""

print "Training Neural Network"
clf = MLPClassifier(solver='adam', alpha=1e-5,
                    hidden_layer_sizes=(10,10), random_state=1, tol=0.0000000001, max_iter=10000)

clf.fit(x_train,y_train)
print "Testing Neural Network"
predicted_nn = clf.predict(x_test)

print "NN Classification Report",(metrics.classification_report(expected, predicted_nn))
print "NN Confusion Matrix",(metrics.confusion_matrix(expected, predicted_nn))
print "NN Accuracy",(metrics.accuracy_score(expected, predicted_nn))



