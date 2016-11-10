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
#from sklearn.metrics import confusion_matrix
#========================== Global parameters ==============================

theta = np.matrix([0] * 15)
AGE_MEAN = 38.4372632304 
AGE_SD = 11.2921665508
y_vector = None
feature_matrix = None
feature_matrixTest = None
learning_rate = 0.001
iterations = 2500
#==============================Helper functions ======================================================

def hypothesis_sigmoid(z):
    return 1.0 / (1 + math.e**(-z))

def simoid_arg(feature_row):
    ans = theta * feature_row.transpose()
    return int(ans[0])
    
def simoid_argKF(feature_row,para_vec):
    ans = para_vec * feature_row.transpose()
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
    pass

def gradient_descent():
    global theta
    diff_list = list()

    for i in range(len(feature_matrix)):
        z = simoid_arg(feature_matrix[i])
        diff_list.append(hypothesis_sigmoid(z) - int(y_vector[0,i]))
        
    diff_vec = np.matrix(diff_list)

    new_theta = theta - (learning_rate/len(feature_matrix)) * (diff_vec * feature_matrix)
    
    return new_theta

def gradient_descentKF(parameter_vec, feature_mat, out_vec):
    diff_list = list()

    for i in range(len(feature_mat)):
        z = simoid_arg(feature_mat[i])
        diff_list.append(hypothesis_sigmoid(z) - int(out_vec[0,i]))
        
    diff_vec = np.matrix(diff_list)

    new_theta = parameter_vec - (learning_rate/len(feature_mat)) * (diff_vec * feature_mat)
    
    return new_theta
    
def learning_function():
    global theta
    for i in range(iterations):
        theta = gradient_descent()

def learning_KF(parameter_vec,feature_mat,out_vec):
    for i in range(iterations):
        parameter_vec = gradient_descentKF(parameter_vec,feature_mat,out_vec)
    return parameter_vec

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
    
def calculate_accuracy():
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    
    for i in range(len(feature_matrix)):
        z = simoid_arg(feature_matrix[i])
        predict_list = list()
        
        if hypothesis_sigmoid(z) >= 0.5:
            predict_list.append(1)
    
            if int(y_vector[0,i]) == 1:
                true_positive += 1
            elif int(y_vector[0,i]) == 0:
                false_positive += 1
                
        elif hypothesis_sigmoid(z) < 0.5:
            predict_list.append(0)
            
            if int(y_vector[0,i]) == 0:
                true_negative += 1
            elif int(y_vector[0,i]) == 1:
                false_negative += 1
            
        else:
            print "Some BUG"
            
    predict_vec = np.matrix(predict_list)
    
    accuracy = float(true_positive + true_negative) / float(true_positive + true_negative + false_positive + false_negative) 
    print "True Positive, ", true_positive
    print "True Negative, ", true_negative
    print "False Positive, ", false_positive
    print "False Negative, ", false_negative    
    print accuracy

def calculate_accuracyKF(feature_mat,out_vec,para_vec):
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    predict_list = list()
    
    for i in range(len(feature_mat)):
        z = simoid_argKF(feature_mat[i],para_vec)
       
        
        if hypothesis_sigmoid(z) >= 0.5:
            predict_list.append(1.0)
    
            if int(out_vec[0,i]) == 1:
                true_positive += 1
            elif int(out_vec[0,i]) == 0:
                false_positive += 1
                
        elif hypothesis_sigmoid(z) < 0.5:
            predict_list.append(0.0)
            
            if int(out_vec[0,i]) == 0:
                true_negative += 1
            elif int(out_vec[0,i]) == 1:
                false_negative += 1 
        else:
            print "Some BUG"
            
    predict_vec = np.matrix(predict_list)
    
    #print out_vec.tolist()[0], predict_vec.tolist()[0]
    print len(out_vec.tolist()[0]), len(predict_vec.tolist()[0])
    
    #print "Precison Score: ", precision_score(out_vec.tolist()[0], predict_vec.tolist()[0])
#    print "F1 Score: ", f1_score(out_vec.tolist()[0], predict_vec.tolist()[0])
#    print "Recall Score: ", recall_score(out_vec.tolist()[0], predict_vec.tolist()[0])
#    
#    false_positive_rate, true_positive_rate, thresholds = roc_curve(out_vec.tolist()[0], predict_vec.tolist()[0])
#    roc_auc = auc(false_positive_rate, true_positive_rate)
#
#    plt.title('Receiver Operating Characteristic')
#    plt.plot(false_positive_rate, true_positive_rate, 'b',
#    label='AUC = %0.2f'% roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1],[0,1],'r--')
#    plt.xlim([-0.1,1.2])
#    plt.ylim([-0.1,1.2])
#    plt.ylabel('True Positive Rate')
#    plt.xlabel('False Positive Rate')
#    plt.show()
    
    accuracy = float(true_positive + true_negative) / float(true_positive + true_negative + false_positive + false_negative) 

    confusion_matrixFormat = [['true_positive','false_negative'],['false_positive','true_negative']]    

    confusion_matrix = [[true_positive,false_negative],[false_positive,true_negative]]
    
    print "Confusion matrix: ",confusion_matrixFormat, confusion_matrix

    print "Accuracy: ", accuracy

    
def k_fold():
    dataset_test1 = feature_matrix[6969:8710,:]
    dataset_test3 = feature_matrix[1742:3484,:]
    dataset_test4 = feature_matrix[3484:5227,:]
    dataset_test5 = feature_matrix[5527:6970,:]
    
    y_test1 = y_vector[0,6969:8710]
    y_test3 = y_vector[0,1742:3484]
    y_test4 = y_vector[0,3484:5227]
    y_test5 = y_vector[0,5527:6970]
    
    dataset1 = feature_matrix[:6970,:]
    dataset3 = np.concatenate((feature_matrix[:1742,:],feature_matrix[3484:,:]))
    dataset4 = np.concatenate((feature_matrix[:3484,:],feature_matrix[5227:,:]))
    dataset5 = np.concatenate((feature_matrix[:5227,:],feature_matrix[6970:,:]))

    dataY1 = y_vector[0,:6970]
    dataY3 = np.concatenate((y_vector[:1742,:],y_vector[3484:,:]))
    #dataY3 = np.concatenate((y_vector[0,:1742][1],y_vector[0,3484:][1]))
    dataY4 = np.concatenate((y_vector[:3484,:],y_vector[5227:,:]))
    dataY5 = np.concatenate((y_vector[:5227,:],y_vector[6970:,:]))
    
    #print '---------',len(y_vector[0,:1742])#.tolist()
    #print '---------',len(y_vector[0,3484:])#.tolist()
    
    para_vec1 = learning_KF(theta,dataset1,dataY1)
    calculate_accuracyKF(dataset1,dataY1,para_vec1)

    para_vec3 = learning_KF(theta,dataset3,dataY3)
    calculate_accuracyKF(dataset3,dataY3,para_vec3)

    para_vec4 = learning_KF(theta,dataset4,dataY4)
    calculate_accuracyKF(dataset4,dataY4,para_vec4)

    para_vec5 = learning_KF(theta,dataset5,dataY5)
    calculate_accuracyKF(dataset5,dataY5,para_vec5)
    
    calculate_accuracyKF(dataset_test1,y_test1,para_vec1)
    calculate_accuracyKF(dataset_test3,y_test3,para_vec3)
    calculate_accuracyKF(dataset_test4,y_test4,para_vec4)
    calculate_accuracyKF(dataset_test5,y_test5,para_vec5)    
#    
    #print [para_vec1, para_vec3, para_vec4, para_vec5]
    #print para_vec5
#======================= Main ======================================================
build_matrices()
k_fold()
#learning_function()
#print calculate_accuracy()
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
