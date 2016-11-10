import numpy as np
import sys
import begin
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
import seaborn as sns
from IPython import display
import time
import cPickle
import gzip
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn import neighbors
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

def minremove(img, threshold=0.7):
    """
        Return new image with pixels below threshold set to zero
    """
    new_img = np.empty(img.shape)
    for (i, j), val in np.ndenumerate(img):
        if val < threshold:
            new_img[i][j] = 0
        else:
            new_img[i][j] = val
    return new_img

def image_scan(image, window_size=28, step=5):
    """
        Generator that produces windows from an image.
    """
    steps = np.arange(0, len(image) - window_size, step)
    for row in steps:
        for col in steps:
            yield image[row:row+window_size, col:col+window_size]

def window_predict(windows, image, predictor, architecture='nn'):
    """
        Gets probability of each window on given model.
    """
    image_intensity = np.mean(image)
    probs = {}
    #get average intensity, don't predict for low intensity windows
    for w in windows:
        #skip low intensit windows
        window_intensity = np.mean(w)
        if window_intensity < image_intensity:
            continue
        
        if architecture == 'cnn':
            p = predictor.predict_proba(w.reshape(1, 28, 28, 1), verbose=0)
        else: 
            p = predictor.predict_proba(normalize(w.reshape(1,784)))

        guess = np.argmax(p)
        guess_prob = p[0][guess]
        probs.setdefault(guess, []).append(guess_prob)
    return sorted(probs.items(), key=lambda x: max(x[1]))


def get_sum(predictions):
    """
        Adds two best guesses
    """
    if len(predictions) < 2:
        return predictions[-1][0] * 2
    num1 = predictions[-1][0]
    num2 = predictions[-2][0]
    return num1 + num2

#### TRAIN MNIST PREDICTORS ###

def mnist_NN():
    """
        Standard feed-forward NN trained on MNIST data.
    """
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255., mnist.target
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]


    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1,
                    learning_rate_init=.1)

    mlp.fit(X_train, y_train)
    #mlp.fit(X, y)
    print("Training set score: %f" % mlp.score(X_train, y_train))
    print("Test set score: %f" % mlp.score(X_test, y_test))

    return mlp

def mnist_CNN_load(json_path, hfive_path):
    """
        Loads CNN trained on MNIST data
    """

    # load json and create model
    json_file = open(json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(hfive_path)

    return model

## VALIDATE ACCURACY ##

def validate(x, y, predictor, architecture='nn'):
    """
        Returns percentage of correct guesses.
    """
    accuracy = 0
    for i, (img, num) in enumerate(zip(x, y)):
        if i % 1000 == 0:
            print("%s of %s images scanned. Accuracy so far: %s" % (i, len(x), accuracy / float(len(x))))
        filtered = minremove(img)
        windows = image_scan(filtered)
        probs = window_predict(windows, filtered, predictor, architecture=architecture)
        addition = get_sum(probs)
        if addition == num[1]:
            accuracy = accuracy + 1
        else:
            continue
    return accuracy / float(len(x))

@begin.start
def run(x_path, y_path):
    X = np.fromfile(x_path, dtype='uint8').reshape(100000, 60, 60) / 255.
    Y = np.genfromtxt(y_path, delimiter=",", dtype=int, skip_header=1)

    #print("Validating NN")
    #pred_nn = mnist_NN()
    #nn_accuracy = validate(X, Y, pred_nn, architecture='nn')
    #print("NN accuracy: %s" % (nn_accuracy))

    print("Validating CNN")
    pred_cnn = mnist_CNN_load("CNN_mnist_keras.json","CNN_mnist_keras.h5")
    cnn_accuracy = validate(X, Y, pred_cnn, architecture='cnn')
    print("CNN accuracy: %s" % (cnn_accuracy))

