#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:11:05 2016

@author: pulkit
"""

#import libraries

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers.recurrent import GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import pydot
import matplotlib.image as mpimg
from keras.datasets import mnist
from keras import callbacks
from keras.utils import np_utils
from keras import backend as K
import tensorflow as tf
tf.python.control_flow_ops = tf
from keras.utils.visualize_util import plot
from IPython.display import Image

numpy.random.seed(7)


dataset = pandas.read_csv('pops.csv')

dataset['timestamp'] = pandas.to_datetime(dataset.timestamp)
dataset['SEDAC GRUMP v1 2000 Population Density Adjusted'] = dataset['SEDAC GRUMP v1 2000 Population Density Adjusted'].fillna(method='ffill')

dataset = dataset.sort('timestamp')

dataset['SEDAC GRUMP v1 2000 Population Density Adjusted'] = dataset['SEDAC GRUMP v1 2000 Population Density Adjusted'].astype(float)

dataset = dataset[['SEDAC GRUMP v1 2000 Population Density Adjusted','location-long','location-lat']]
dataset = dataset.as_matrix()
print dataset.shape


# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back,1:])
	return numpy.array(dataX), numpy.array(dataY)

 # reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)


print trainX.shape
print trainY.shape
print testX.shape
print testY.shape

model = Sequential()

model.add(LSTM(4, input_dim=3,return_sequences=True))
model.add(Activation("relu"))
model.add(LSTM(200, input_dim=4,return_sequences=True))
model.add(Activation("relu"))
model.add(Dropout(0.2)) 
model.add(LSTM(100, input_dim=200, return_sequences=True))
model.add(Activation("relu"))
model.add(Dropout(0.2)) 
model.add(LSTM(500, input_dim=500, return_sequences=False))
model.add(Activation("relu"))
model.add(Dense(2))
model.add(Activation("relu"))
model.compile(loss='mean_squared_error', optimizer='adam')

plot(model, to_file='./model.png',show_shapes=True)

#model.fit(trainX, trainY, nb_epoch=20, batch_size=1, verbose=2)
history = model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=1,validation_split = 0.2)



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Performance of the model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper right')
plt.show()


# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions

print trainPredict.shape

print testPredict.shape

print trainY.shape

print testY.shape

"""
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)
trainPredict = scaler.inverse_transform(trainPredict)
"""

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print trainY
print trainPredict


# shift train predictions for plotting
trainPredictPlot = numpy.zeros((len(dataset),2))
print trainPredictPlot.shape
trainPredictPlot[:] = numpy.nan
print trainPredictPlot.shape
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.zeros((len(dataset),2))
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict
# plot baseline and predictions

print trainPredictPlot
print testPredictPlot
plt.plot(dataset[:,1])
plt.plot(trainPredictPlot[:,0])
plt.plot(testPredictPlot[:,0])
plt.show()


