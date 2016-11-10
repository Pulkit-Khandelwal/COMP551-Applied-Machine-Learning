from __future__ import division
import math
import numpy as np
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import math
import matplotlib.pyplot as plt

#read and organise dataset

dataset = np.genfromtxt("classification.csv", delimiter=",", dtype= np.float64)
print dataset.shape
mean = np.mean(dataset[:,0])
std = np.std(dataset[:,0])
dataset[:,0] = (dataset[:,0] - mean) / std

for i in range(len(dataset)):
	if dataset[i,10] == 2:
		dataset[i, 10] = 0

#dataset: assume n by m matrix

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

#Assume the data has the first 10 columns as continuous inputs
#and the next four columns have binary inputs and the last column has the label

n_zero = len(class_zero_data)
n_one = len(class_one_data)

#calculate the vector for the means for each class for
#each feature

mean_zero = np.mean(class_zero_data[:,:m-5], axis=0)
mean_one = np.mean(class_one_data[:,:m-5], axis=0)


#calculate the vector for the variance for each class for
#each feature


var_zero = np.var(class_zero_data[:,:m-5], axis=0)
var_one = np.var(class_one_data[:,:m-5], axis=0)


#prior probabilites
prior_zero = float(n_zero)/n
prior_one = float(n_one)/n


#check output until now:
print "dataset", dataset
print "n= %d" % n
print "m= %d" % m
print "class_zero_data"
print class_zero_data
print "class_one_data"
print class_one_data

print "count of zero examples", n_zero
print "count of one examples", n_one

print "mean of the examples of class zero", mean_zero
print "mean of the examples of class one", mean_one

print "variance of the examples of class zero", var_zero
print "variance of the examples of class one", var_one

print "prior probabilites for class zero", prior_zero
print "prior probabilites for class one", prior_one


#for the ten continuous variables using gaussian distribution

prob_features_given_zero = np.zeros((n,m-5))
prob_features_given_one = np.zeros((n,m-5))

for i in range(n):
	for j in range(m-5):
		prob_features_given_zero[i,j] = (1/float((math.sqrt(2*(math.pi)*var_zero[j])))) * (math.exp((-1* (math.pow((dataset[i,j]-mean_zero[j]),2)/float(2*var_zero[j])))))

for i in range(n):
	for j in range(m-5):
		prob_features_given_one[i,j] = (1/float((math.sqrt(2*(math.pi)*var_one[j])))) * (math.exp((-1* (math.pow((dataset[i,j]-mean_one[j]),2)/float(2*var_one[j])))))


print "prob_features_given_zero", prob_features_given_zero
print "prob_features_given_one", prob_features_given_one


#for the four binary varibales using bernoulli distribution

prob_features_is_zero_given_y_is_zero = np.zeros(4)
prob_features_is_one_given_y_is_zero = np.zeros(4)
prob_features_is_zero_given_y_is_one = np.zeros(4)
prob_features_is_one_given_y_is_one = np.zeros(4)

counter = 0

for j in range(4):
	for i in range(n_zero):
		if class_zero_data[i,j] == 0:
			counter +=1
    
	prob_features_is_zero_given_y_is_zero[j] = float(counter)/n_zero
	prob_features_is_one_given_y_is_zero[j] = 1 - prob_features_is_zero_given_y_is_zero[j]
	counter = 0

print "prob_features_is_zero_given_y_is_zero", prob_features_is_zero_given_y_is_zero

print "prob_features_is_one_given_y_is_zero", prob_features_is_one_given_y_is_zero

counter = 0 

for j in range(4):
	for i in range(n_one):
		if class_one_data[i,j] == 0:
			counter +=1
    
	prob_features_is_zero_given_y_is_one[j] = float(counter)/n_one
	prob_features_is_one_given_y_is_one[j] = 1 - prob_features_is_zero_given_y_is_one[j]
	counter = 0

print "prob_features_is_zero_given_y_is_one", prob_features_is_zero_given_y_is_one

print "prob_features_is_one_given_y_is_one", prob_features_is_one_given_y_is_one

product_predict_assume_class_zero = np.ones((n))
product_predict_assume_class_one = np.ones((n))

for i in range(n):
	for j in range(4):
		if dataset[i,j] == 0:
			product_predict_assume_class_zero[i] = product_predict_assume_class_zero[i] * prob_features_is_zero_given_y_is_zero[j]
		else:
			product_predict_assume_class_zero[i] = product_predict_assume_class_zero[i] * prob_features_is_one_given_y_is_zero[j]
		 

for i in range(n):
	for j in range(4):
		if dataset[i,j] == 0:
			product_predict_assume_class_one[i] = product_predict_assume_class_one[i] * prob_features_is_zero_given_y_is_one[j]
		else:
			product_predict_assume_class_one[i] = product_predict_assume_class_one[i] * prob_features_is_one_given_y_is_one[j]

#final prodcuts for continuous data

final_prob_zero = (np.prod(prob_features_given_zero, axis=1))
final_prob_one = (np.prod(prob_features_given_one, axis=1))


print "final_prob_zero", final_prob_zero
print "final_prob_one", final_prob_one

#final prodcuts for binary data

product_predict_assume_class_zero = product_predict_assume_class_zero
product_predict_assume_class_one = product_predict_assume_class_one


print "product_predict_assume_class_zero", product_predict_assume_class_zero
print "product_predict_assume_class_one", product_predict_assume_class_one


#calculated probabilties using all the features for both the classes

probabiltiy_zero = final_prob_zero * product_predict_assume_class_zero * prior_zero
probabiltiy_one = final_prob_one * product_predict_assume_class_one * prior_one

print probabiltiy_zero, probabiltiy_zero.shape
print probabiltiy_one, probabiltiy_one.shape

#finally predict the classes using the Naive Bayes Classifier for the training data

prediction_output = np.zeros((n))

for i in range(n):
	if probabiltiy_zero[i] > probabiltiy_one[i]:
		prediction_output[i] = 0
	else:
		prediction_output[i] = 1

print "final prediction", prediction_output
print prediction_output.shape

predicted_zero = 0
predicted_one = 0

for i in range(n):
	if prediction_output[i] == 0:
		predicted_zero = predicted_zero + 1
	else:
		predicted_one = predicted_one + 1


print "final prediction for class zero", predicted_zero
print "final prediction for class one", predicted_one

print(metrics.classification_report(dataset[:,-1], prediction_output))
print(metrics.confusion_matrix(dataset[:,-1], prediction_output))

#calculate accuracy
correct = 0
for i in range(n):
	if prediction_output[i] == dataset[i,-1]:
		correct +=1

print "accuracy", float(correct)/n

#evaluation metric

target = dataset[:,-1]
print target.shape
expected = target

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(expected, prediction_output)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
