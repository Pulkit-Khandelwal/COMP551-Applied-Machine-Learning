from __future__ import division
import math
import numpy as np
from sklearn import metrics

#read data and normalize the feature: age

dataset = np.genfromtxt("classification.csv", delimiter=",", dtype= np.float64)
print dataset.shape
mean = np.mean(dataset[:,0])
std = np.std(dataset[:,0])
dataset[:,0] = (dataset[:,0] - mean) / std

for i in range(len(dataset)):
	if dataset[i,10] == 2:
		dataset[i, 10] = 0

#dataset's shape is n by 15 matrix where the last column has the labels 'y'
n = len(dataset)
m = len(dataset[0])


#Separate Data by Class for easier processing
class_zero_data = []
class_one_data = []

for i in range(n):
		if dataset[i,-1] == 0:
			class_zero_data.append(dataset[i])
		else:
			class_one_data.append(dataset[i])

class_zero_data = np.array(class_zero_data)
class_one_data = np.array(class_one_data)

#check output until now:
print "dataset", dataset
print "n= %d" % n
print "m= %d" % m
print "class_zero_data"
print class_zero_data
print "class_one_data"
print class_one_data


#Assume the data has the first 10 columns as continuous inputs
#and the next four columns have binary inputs and the last column has the label for 'y'

n_zero = len(class_zero_data)
n_one = len(class_one_data)

#calculate the vector for the means for each class for each feature

mean_zero = np.mean(class_zero_data[:,:-1], axis=0)
mean_one = np.mean(class_one_data[:,:-1], axis=0)

#calculate the vector for the variance for each class for each feature

var_zero = np.var(class_zero_data[:,:-1], axis=0)
var_one = np.var(class_one_data[:,:-1], axis=0)

#prior probabilites
prior_zero = float(n_zero)/n
prior_one = float(n_one)/n

#check output until now:
print "count of zero examples", n_zero
print "count of one examples", n_one

print "mean of the examples of class zero", mean_zero
print "mean of the examples of class one", mean_one

print "variance of the examples of class zero", var_zero
print "variance of the examples of class one", var_one

print "prior probabilites for class zero", prior_zero
print "prior probabilites for class one", prior_one


#Assume test data to be in the same format as the input space
#it is a matrix in the same format "here: dataset"
#make prediction here:

#prediction on training data

prob_features_given_zero = np.zeros((n,m-1))
prob_features_given_one = np.zeros((n,m-1))

for i in range(n):
	for j in range(m-1):
		prob_features_given_zero[i,j] = (1/float((math.sqrt(2*(math.pi)*var_zero[j])))) * (math.exp((-1* (math.pow((dataset[i,j]-mean_zero[j]),2)/float(2*var_zero[j])))))

for i in range(n):
	for j in range(m-1):
		prob_features_given_one[i,j] = (1/float((math.sqrt(2*(math.pi)*var_one[j])))) * (math.exp((-1* (math.pow((dataset[i,j]-mean_one[j]),2)/float(2*var_one[j])))))


print "prob_features_given_zero", prob_features_given_zero
print "prob_features_given_one", prob_features_given_one


final_prob_zero = (np.prod(prob_features_given_zero, axis=1)) * prior_zero
final_prob_one = (np.prod(prob_features_given_one, axis=1)) * prior_one

print "final_prob_zero", final_prob_zero
print "final_prob_one", final_prob_one


prediction_output = np.zeros((n))

for i in range(n):
	if final_prob_zero[i] >= final_prob_one[i]:
		prediction_output[i] = 0
	else:
		prediction_output[i] = 1

print "final prediction", prediction_output

#calculate accuracy
correct = 0
for i in range(n):
	if prediction_output[i] == dataset[i,-1]:
		correct +=1

print "accuracy", float(correct)/n

tp = 0
for i in range(n):
	if prediction_output[i] == dataset[i,-1] and prediction_output[i] == 1:
		tp +=1

print "true positive", tp


print(metrics.classification_report(dataset[:,-1], prediction_output))
print(metrics.confusion_matrix(dataset[:,-1], prediction_output))

#prediction on testdata

dataset_test = np.genfromtxt("classification_test.csv", delimiter=",", dtype= np.float64)
print dataset_test.shape
mean = np.mean(dataset_test[:,0])
std = np.std(dataset_test[:,0])
dataset_test[:,0] = (dataset_test[:,0] - mean) / std

for i in range(len(dataset_test)):
	if dataset_test[i,10] == 2:
		dataset_test[i, 10] = 0


prob_features_given_zero = np.zeros((n,m-1))
prob_features_given_one = np.zeros((n,m-1))

for i in range(n):
	for j in range(m-1):
		prob_features_given_zero[i,j] = (1/float((math.sqrt(2*(math.pi)*var_zero[j])))) * (math.exp((-1* (math.pow((dataset_test[i,j]-mean_zero[j]),2)/float(2*var_zero[j])))))

for i in range(n):
	for j in range(m-1):
		prob_features_given_one[i,j] = (1/float((math.sqrt(2*(math.pi)*var_one[j])))) * (math.exp((-1* (math.pow((dataset_test[i,j]-mean_one[j]),2)/float(2*var_one[j])))))


print "prob_features_given_zero", prob_features_given_zero
print "prob_features_given_one", prob_features_given_one


final_prob_zero = (np.prod(prob_features_given_zero, axis=1)) * prior_zero

final_prob_one = (np.prod(prob_features_given_one, axis=1)) * prior_one


print "final_prob_zero", final_prob_zero
print "final_prob_one", final_prob_one


#prediction on test data
prediction_output = np.zeros((n))

for i in range(n):
	if final_prob_zero[i] >= final_prob_one[i]:
		prediction_output[i] = 0
	else:
		prediction_output[i] = 1

print "final prediction", prediction_output

countervar = 0
for i in range(len(prediction_output)):
	if prediction_output[i] == 1:
		countervar +=1

print "number of ones in prediction are", countervar

#write the predicted output to csv file

out = open('predicted_my_implementation.csv', 'w')
for row in prediction_output:
    out.write('%d' % row)
    out.write('\n')
out.close()

