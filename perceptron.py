#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 22:07:02 2019

@author: Varunya Ilanghovan, Camilo Barrera
"""

import numpy as np
import pandas as pd
from random import random

#Import the dataset
dataset=pd.read_csv('classification.txt',names=['x','y','z','label','drop'])

#Create X matrix
X=dataset.iloc[:,0:3]
ones=np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

#Create y matrix
Y=dataset.iloc[:,3:4].values

#Create random Weights array
w=np.zeros([1,4])
for i in range(len(w.T)):
    thr=random()
    initial_w=random()
    if thr < 0.5:
        w[0,i]=-initial_w
    else:
        w[0,i]=initial_w      
print('Initial weights: ',w)

#Define alpha step for Gradient Descent algorithm
alpha=0.01
#Define number of iterations that algorithm will run
iterations=100

#Create Activation function to make predictions
def Activation(row,w):
    summation=np.dot(row,w.T)
    if summation>0:
        activation=1
    else:
        activation=-1
    return activation

#Create Train function to train the weights
def Train(X,Y,iterations,alpha):
    for i in range(iterations):
        sum_error=0
        for row,k in zip(X,range(len(X))):
            prediction=Activation(row,w)
            error=Y[k]-prediction
            sum_error += error**2
            for j in range(4):
                w[0,j]=w[0,j]+alpha*((Y[k]-prediction)*row[j])
    return w

#Create function to measure accuracy
def Accuracy(X,Y,w):
    error=0
    test=np.zeros([len(X),2])
    for k in range(len(X)):
        test[k,0]=Activation(X[k],w)
        test[k,1]=Y[k]
        if test[k,0] != test[k,1]:
            error+=1
        accuracy=100-((error/(k+1))*100)
    print('Accuracy: ',accuracy)
    return accuracy

w=Train(X,Y,iterations,alpha)
accuracy=Accuracy(X,Y,w)

print('Final weights: ',w)

    
        