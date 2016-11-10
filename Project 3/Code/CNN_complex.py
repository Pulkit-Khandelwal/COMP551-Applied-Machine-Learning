# -*- coding: utf-8 -*-

'''
A slightly more complex model than the CNN_simple.py convNet
layer.
'''

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

seed = 7
np.random.seed(seed)

x = np.fromfile('train_x.bin', dtype='uint8')

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]

test = np.fromfile('test_x.bin', dtype='uint8')

# reshape to be [samples][pixels][width][height]
x_flat = x.reshape(100000, 1, 60, 60)
test_flat = test.reshape(20000, 1, 60, 60)

x_flat = x_flat/255
test_flat = test_flat/255

x_train = x_flat[:2000]
x_test = x_flat[2000:]
y_train = y[2500:2650]
y_test = y[2500:2650]


y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print num_classes

def complex_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# build the model
model = complex_model()
# fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), nb_epoch=10, batch_size=200, verbose=2)
# model evaluation
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

classes = model.predict_classes(x_test, batch_size=200)
proba = model.predict_proba(x_test, batch_size=200)





