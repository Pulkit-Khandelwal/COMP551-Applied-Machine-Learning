# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 20:58:57 2016

@author: pulkit
"""

import numpy as np

class knn():
        
    def __init__(self,k,p=None,sigma=None,dist='eu'):        
        self.k=k
        self.p=p
        self.sigma=sigma
        self.dist=dist
    
    def fit(self,X,Y):
        self.X=X
        self.Y=np.array([Y]).transpose()
        
    def predict(self,X_test):
        Y=self.Y
        X=self.X
        k=self.k
        p=self.p
        sigma=self.sigma
        dist=self.dist
        distance_matrix = np.zeros((len(X_test),len(X)))
        
        if(dist=='eu'):
            for i in range(len(X_test)):
                #dd[i,:] = np.sum(np.square(X-X_test[i,:]), axis = 1)
                distance_matrix[i,:] = np.sqrt(np.sum(np.square(X-X_test[i,:]),axis = 1))
        elif(dist=='mi'):
            for i in range(len(X_test)):
                distance_matrix[i,:] = np.power(np.sum(np.power(abs(X-X_test[i,:]),p),axis=1),float(1)/p)

                #dd[i,:] = np.sum(np.square(X-X_test[i,:]), axis = 1)

        elif(dist=='rbf'):
            #dd = np.zeros((len(X_test),len(X)))
            for i in range(len(X_test)):
                #dd[i,:] = np.sum(np.square(X-X_test[i,:]), axis = 1)
                distance_matrix[i,:] =  np.exp(-1*(np.sum(np.square(X-X_test[i,:]), axis = 1) / (2*sigma*sigma)))
        
        sorted_indices = np.argsort(distance_matrix)
        Y = Y.transpose()
        #print Y.shape
        
        top_k_indices = sorted_indices[:,0:k]
        #print top_k_indices
        
        
        top_k_Y_entries = np.zeros((len(X_test),k))
        #print top_k_Y_entries.shape
        
        for i in range(len(X_test)):
            for j in range(k):
                index = top_k_indices[i,j]
                top_k_Y_entries[i,j] = Y[0,index]
                
        #print top_k_Y_entries
        top_k_Y_entries  = top_k_Y_entries.astype(int)
        
        Y_predicted = np.zeros((len(X_test),1))
        for i in range(len(X_test)):
            Y_predicted[i,0] = np.argmax(np.bincount(top_k_Y_entries[i,:]))
	 
        return np.ravel(Y_predicted.astype(int))
