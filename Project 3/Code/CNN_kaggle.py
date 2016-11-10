# -*- coding: utf-8 -*-

'''
The script used for the submission for the
kaggle competition. A very dense model of multiple
convolution, maxpool and dropout layers with varying
number of filter sizes have been used.
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
from sklearn.utils import shuffle
import csv


seed = 7
np.random.seed(seed)

x = np.fromfile('train_x.bin', dtype='uint8')

y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:100001,1]

test = np.fromfile('test_x.bin', dtype='uint8')

# reshape image
x_flat = x.reshape(100000, 1, 60, 60)
test_flat = test.reshape(20000, 1, 60, 60)

x_flat = x_flat/255
test_flat = test_flat/255

x_flat, y = shuffle(x_flat, y, random_state=0)

x_train = x_flat
x_test = test_flat
y_train = y


y_train = np_utils.to_categorical(y_train)

num_classes = y_train.shape[1]

print num_classes

def larger_model():
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, border_mode='valid', input_shape=(1, 60, 60), activation='relu'))
    model.add(Convolution2D(64, 7, 7, activation='relu'))    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
# build the model
model = larger_model()
# fit the model
model.fit(x_train, y_train, nb_epoch=10, batch_size=200, verbose=1)

# predict on the test dadaset
classes = model.predict_classes(x_test, batch_size=200)
proba = model.predict_proba(x_test, batch_size=200)

# save the classes and probability for each prediction in the test dataset
np.savetxt("classes_sample_kaggle.csv",classes, delimiter =",")
np.savetxt("prob_sample_kaggle.csv",proba, delimiter =",")

cols = np.zeros(len(classes))
print classes.shape
print cols.shape

for i in range(len(classes)):
    cols[i] = i
    
rows = zip(cols,classes)

# save the predictions as a csv file
with open('output_sample_kaggle.csv', 'wb') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


