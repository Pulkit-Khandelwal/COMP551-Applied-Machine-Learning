# -*- coding: utf-8 -*-

'''
Implements an RNN as adapted from a paper by
Geoffery Hinton et al., 2015
'''
from __future__ import print_function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import model_from_yaml

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
from sklearn.utils import shuffle
import csv

batch_size = 200
nb_classes = 19
nb_epochs = 50
hidden_units = 100

learning_rate = 1e-6
clip_norm = 1.0

x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]

test = np.fromfile('test_x.bin', dtype='uint8')
test = test.reshape((20000,60,60))

x_flat = x.reshape(100000,3600)
test_flat = test.reshape(20000,3600)


x_flat, y = shuffle(x_flat, y, random_state=0)

x_flat = x_flat/255
test_flat = test_flat/255

x_flat, y = shuffle(x_flat, y, random_state=0)

x_train = x_flat[:80000]
x_test = x_flat[80000:]
y_train = y[:80000]
y_test = y[80000:]


x_train = x_train.reshape(80000, -1, 1)
x_test = x_test.reshape(20000, -1, 1)


y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test, nb_classes)

print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(output_dim=hidden_units,
                    init=lambda shape, name: normal(shape, scale=0.001, name=name),
                    inner_init=lambda shape, name: identity(shape, scale=1.0, name=name),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epochs,
          verbose=1, validation_data=(x_test, y_test))

#save the trained for future use

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores[0])
print('IRNN test accuracy:', scores[1])





