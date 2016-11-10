'''
A	fully	connected	feedforward	neural	network (from	lecture	14),	trained	by	
backpropagation,	where	the	network	architecture	(number	of	nodes	/	layers),	learning	
rate	and	termination	are	determined	by	cross-validation.		This	method	must	be	fully	
implemented	by	your	team,	and	corresponding	code	submitted.
'''

import numpy as np
import random
import sklearn
import cPickle
import gzip
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.feature import daisy

# main method neural network
def neural_net(netArch, X, test_data, learning_rate):


    #   netArch is a array where the length is the number of layers and each index holds the corresponding about of neurons
    #   i.e. [# of features,6,2, # of outputs (19 digits)] -> 4 layers in total, 6 neurons in 2nd layer
    #   note that first layer is input and  last is output, between is hidden layers
    numLayers = len(netArch)
    weights = [np.random.randn(y, x) for x, y in zip(netArch[:-1], netArch[1:])]
    biases = [np.random.randn(y, 1) for y in netArch[1:]]

    # without any training, print result with random weights
    print "Random baseline: {0} / {1}".format(calc_result(weights, biases, test_data), len(test_data))

    numExamples = len(X)
    batch_size = 1000
    numEpoch = 10

    # Randomly select a mini-batch (i.e. subset of training examples).
    # Calculate error on mini-batch, apply to update weights, and repeat
    for i in xrange(numEpoch):
        random.shuffle(X) #to randomize batches in each iteration
        # split data into batches of batch_size
        batches = [X[j:j+batch_size] for j in xrange(0, numExamples, batch_size)]
        # loop through each batch and update weights & bias through back propagation
        for b in batches:
            weights, biases = gradDesc(b, learning_rate, weights, biases, numLayers)
        #print every even epoch to track progress
        if (i % 2 == 0):
                print "{0}-epoch: {1} / {2}".format(i, calc_result(weights, biases, test_data), len(test_data))
        #otherwise print done
        else:
                print "{0}-epoch done".format(i)

    print "Final Result, {0}-epoch: {1} / {2}".format(numEpoch, calc_result(weights, biases, test_data), len(test_data))

#gradient descent on current batch using feedforward and back propagation
def gradDesc(batch,l_r, weights, biases, numLayers):
    gradient_b = [np.zeros(b.shape) for b in biases]
    gradient_w = [np.zeros(w.shape) for w in weights]
    #calculate error over all examples in batch by looping each example to back propagation
    for x, y in batch:
        delta_b, delta_w = back_prop(x, y, numLayers, weights, biases)
        # update batch gradients
        gradeint_b = [gb+db for gb, db in zip(gradient_b, delta_b)]
        gradient_w = [gw+dw for gw, dw in zip(gradient_w, delta_w)]
    #updates weights & biases from all of batch gradient
    weights = [w-(l_r/len(batch))*gw for w, gw in zip(weights, gradient_w)]
    biases = [b-(l_r/len(batch))*gb  for b, gb in zip(biases, gradient_b)]
    return (weights, biases)

def back_prop(x,y, numLayers, weights, biases):
    #initial values from first layer
    feedforward = x
    #a & z correspond to variables given in class
    z = []
    a = [feedforward] 
    # for all the z and a values, update with weights and biases
    for b, w in zip(biases, weights):
            feedforward = sig(np.dot(w, feedforward)+b)
            z.append(np.dot(w, feedforward)+b)
            a.append(feedforward)
    #cost function
    cost = a[-1] - y
    #use that to calculate delta
    delta = cost * \
        sig_deriv(z[-1])
    delta_b = [np.zeros(b.shape) for b in biases]
    delta_w = [np.zeros(w.shape) for w in weights]
    # initialize last layer 
    delta_b[-1] = delta
    delta_w[-1] = np.dot(delta, a[-2].transpose())
    #calculate cost function
    for layer in xrange(2, numLayers):
        sp = sig_deriv(z[-layer])
        delta = np.dot(weights[-l+1].transpose(), delta) * sp
        delta_b[-layer] = delta
        delta_w[-layer] = np.dot(delta, a[-layer-1].transpose())
    return (delta_b, delta_w)    

def sig(x):
    return 1.0/(1.0+np.exp(-x))

def sig_deriv(x):
    return sig(x)*(1-sig(x))

#outputs # of correct guesses on test data with given weights and biases
def calc_result(weights, biases, test_data):
        # whichever neuron has the highest value in the output last_layer
        # is the prediction
        results = [(np.argmax(calc_net(weights, biases, x)[0:18]), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in results)

# takes individual test example and applies weight and biases 
# returns output layer for that 
def calc_net(weights, biases, a):
        for b, w in zip(biases, weights):
            a = sig(np.dot(w, a)+b)
        return a

# convert digit to 19 neurons in last layer 
# where the neuron = 1 to the corresponding digit
# all the others = 0
def last_layer(y):
    r = np.zeros((19, 1))
    r[y] = 1.0
    return r

# Start of script

import sys
sys.stdout = open('NN_results.txt', 'w')

# reformat training data
x = np.fromfile('train_x.bin', dtype='uint8')
x = x.reshape((100000,60,60))

#make HoG features
print "HoG: Saving features' loop for train"
hog_features_train_set = np.zeros((len(x),3600))
for i in range(len(x)):
    fd, hog_image = hog(x[i], orientations=16, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualise=True)
    hog_features_train_set[i] = fd.reshape((1,3600))

print "Formating data..."
y = np.genfromtxt("train_y.csv", delimiter=",", dtype= np.float64)
y = y[1:80001,1]

#train on 80% of data
train_x = [np.reshape(f, (3600, 1)) for f in hog_features_train_set[0:80000]]
train_y = [last_layer(k) for k in y[0:80000]]

#test on 20% of data
test_x = [np.reshape(f, (3600, 1)) for f in hog_features_train_set[80001:100000]]
test_y = y[80001:100000]

X = zip(train_x, train_y)
test_data = zip(test_x, test_y)

# test various network architectures
numTest = len(test_data)
print "number of validation sets: %d" % numTest

# example of how to run a neural net
print "--------------------------------"
print "NN with [3600,5,5,19] with learning rate = .5"
neural_net([3600,5,5,19], X, test_data, .5)
print "--------------------------------"




