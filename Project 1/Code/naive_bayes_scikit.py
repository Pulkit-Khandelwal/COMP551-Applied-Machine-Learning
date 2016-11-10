from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.qda import QDA

#read dataset from csv file and normalize features 

dataset = np.genfromtxt("classification.csv", delimiter=",", dtype= np.float64)
print dataset.shape
mean = np.mean(dataset[:,0])
std = np.std(dataset[:,0])
dataset[:,0] = (dataset[:,0] - mean) / std

for i in range(len(dataset)):
	if dataset[i,10] == 2:
		dataset[i, 10] = 0

target = dataset[:,-1]
data = dataset[:,:14]
expected = target

#QDA

clf = QDA()
clf.fit(data, target)
predicted = clf.predict(data)

#LDA

clf = LDA()
clf.fit(data, target)
predicted = clf.predict(data)

#Gaussian

model = GaussianNB()
model.fit(data, target)
print(model)

#data for prediction:

dataset_test = np.genfromtxt("classification_test.csv", delimiter=",", dtype= np.float64)
print dataset_test.shape
mean = np.mean(dataset_test[:,0])
std = np.std(dataset_test[:,0])
dataset_test[:,0] = (dataset_test[:,0] - mean) / std

for i in range(len(dataset_test)):
	if dataset_test[i,10] == 2:
		dataset_test[i, 10] = 0


predicted = model.predict(dataset_test)

print expected
print predicted


countervar = 0
for i in range(len(predicted)):
	if predicted[i] == 1:
		countervar +=1

print "number of ones in prediction are", countervar


# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# calculate accuracy
correct = 0
for i in range(len(dataset)):
	if predicted[i] == dataset[i,-1]:
		correct +=1

print "accuracy", float(correct)/len(dataset)

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(expected, predicted)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

# plot evaluation metrics
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

# save predicted output as a csv file
outs = open('predicted_scikit.csv', 'w')
for row in predicted:
    outs.write('%d' % row)
    outs.write('\n')
outs.close()

