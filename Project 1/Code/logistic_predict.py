#======================== Imports ===================================
import math
import csv
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve

#========================== Global parameters ==============================

theta = np.matrix([0] * 15)
AGE_MEAN = 38.4372632304 
AGE_SD = 11.2921665508
y_vector = None
feature_matrix = None
feature_matrixTest = None
learning_rate = 0.001
random_theta = list()
j_theta = list()
random_parameterNo = 2
iterations = 2000
#==============================Helper functions ======================================================

def hypothesis_sigmoid(z):
    return 1.0 / (1 + math.e**(-z))

def simoid_arg(feature_row):
    ans = theta * feature_row.transpose()
    return int(ans[0])
    
    
def convert_featureRow(raw_data):
    feature_row = list()
    normalised_age = (float(raw_data[0]) - AGE_MEAN)/AGE_SD
    feature_row.append(1)
    feature_row.append(normalised_age)
    
    for i in range(1,len(raw_data)):
        try:
            feature_row.append(float(raw_data[i]))
        except:
            print "error in:",raw_data[i]
            
    return feature_row
        
def cost_func():
    summation = 0
    for i in range(len(feature_matrix)):
        z = simoid_arg(feature_matrix[i])
        summation += ((int(y_vector[0,i]) * math.log(hypothesis_sigmoid(z))) + ((1 - int(y_vector[0,i])) * math.log(1 - hypothesis_sigmoid(z)))) 
    return summation/len(feature_matrix)
    
def gradient_descent():
    global theta, random_theta, j_theta
    
    diff_list = list()

    for i in range(len(feature_matrix)):
        z = simoid_arg(feature_matrix[i])
        diff_list.append(hypothesis_sigmoid(z) - int(y_vector[0,i]))
        
    diff_vec = np.matrix(diff_list)

    new_theta = theta - (learning_rate/len(feature_matrix)) * (diff_vec * feature_matrix)
    
    random_theta.append(new_theta[0,random_parameterNo])
    j_theta.append(cost_func())

    return new_theta

def learning_function():
    global theta
    for i in range(iterations):
        theta = gradient_descent()

def build_matrices():
    global feature_matrix, y_vector

    columns = defaultdict(list) 
    f_matrix = list(list())
    
    with open('classification.csv') as f:
        reader = csv.reader(f)
        
        for row in reader:
            for (i,v) in enumerate(row):
                columns[i].append(float(v))
        
            feature_vec = convert_featureRow(row)
            del feature_vec[-1]
            f_matrix.append(feature_vec)
            
    y_vector = np.matrix(columns[14])
    feature_matrix = np.matrix(f_matrix)
    
def build_matricesTest():
    global feature_matrixTest

    f_matrix = list(list())
    
    with open('classification_test.csv') as f:
        reader = csv.reader(f)
        
        for row in reader:
            feature_vec = convert_featureRow(row)
            f_matrix.append(feature_vec)
            
    feature_matrixTest = np.matrix(f_matrix)
    
    return feature_matrixTest
    
def calculate_accuracy():
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    predict_list = list()
    
    for i in range(len(feature_matrix)):
        z = simoid_arg(feature_matrix[i])
        
        if hypothesis_sigmoid(z) >= 0.5:
            predict_list.append(1.0)
    
            if int(y_vector[0,i]) == 1:
                true_positive += 1
            elif int(y_vector[0,i]) == 0:
                false_positive += 1
                
        elif hypothesis_sigmoid(z) < 0.5:
            predict_list.append(0.0)
            
            if int(y_vector[0,i]) == 0:
                true_negative += 1
            elif int(y_vector[0,i]) == 1:
                false_negative += 1
            
        else:
            print "Some BUG"
            
    predict_vec = np.matrix(predict_list)
    
    #print predict_list
    #print predict_vec
    
    #print type(y_vector.tolist()), type(predict_vec.tolist())
    #print y_vector.tolist()
    #print predict_vec.tolist()
    #print y_vector.tolist(), predict_vec.tolist()
    
    print "Precison Score: ", precision_score(y_vector.tolist()[0], predict_vec.tolist()[0])
    print "F1 Score: ", f1_score(y_vector.tolist()[0], predict_vec.tolist()[0])
    print "Recall Score: ", recall_score(y_vector.tolist()[0], predict_vec.tolist()[0])
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_vector.tolist()[0], predict_vec.tolist()[0])
    roc_auc = auc(false_positive_rate, true_positive_rate)

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

    print "True Positive, ", true_positive
    print "True Negative, ", true_negative
    print "False Positive, ", false_positive
    print "False Negative, ", false_negative  
    
    accuracy = float(true_positive + true_negative) / float(true_positive + true_negative + false_positive + false_negative) 
    
    print "Accuracy: ",accuracy
    
def calculate_accuracyTest():
    positive = 0
    negative = 0
    predict_list = list()
    
    for i in range(len(feature_matrixTest)):
        z = simoid_arg(feature_matrixTest[i])
                
        if hypothesis_sigmoid(z) >= 0.5:
            predict_list.append(1)
            positive += 1
            
        elif hypothesis_sigmoid(z) < 0.5:
            predict_list.append(0)
            negative += 1
        else:
            print "Some BUG"
            
    #predict_vec = np.matrix(predict_list)
    
    print "Positive, ", positive
    print "Negative, ", negative
    
    return predict_list
    
def plot_graph(x,y,xlable,ylable):
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    
    plt.plot(x,y,'ro')
    plt.show()
    
#======================= Main ======================================================
build_matrices()
build_matricesTest()

learning_function()
calculate_accuracy()

finalList = calculate_accuracyTest()


out = open('outFinal.csv', 'w+')

for ele in finalList:
        out.write('%d,' %ele)
        out.write('\n')
out.close()

#plot_graph(random_theta,j_theta,'Parameter value', 'Cost Value')

#================= Testing ============================
#main()
#print convert_featureRow(['52.0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '2', '0', '0', '0', '1'])
#print build_matrices()[0]
#build_matrices()
#
#start = time.time()
#learning_function()
#stop = time.time()
#
#print theta
#print "Time in Sec: ", (stop - start)
#print len(feature_matrix), len(y_vector)
#print type( y_vector[0])
#if '2' in y_vector:
#    print y_vector
#print simoid_arg(np.matrix([1] * 14))
#k_fold()