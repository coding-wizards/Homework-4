#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 00:19:57 2019

@author: cami
"""

import numpy as np
import pandas as pd
from random import random
import matplotlib.pyplot as plt

#Import the dataset
dataset=pd.read_csv('classification.txt',names=['x','y','z','label','drop'])

#Create X matrix
X=dataset.iloc[:,0:3]
ones=np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

#Create y matrix
Y=dataset.iloc[:,4:5].values

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

#Initialize pocket
pocket=w

#Define alpha step for Gradient Descent algorithm
alpha=0.01
#Define number of iterations that algorithm will run
iterations=7000

#Create Activation function to make predictions
def Activation(row,w):
    summation=np.dot(row,w.T)
    if summation>0:
        activation=1
    else:
        activation=-1
    return activation

#Create Train function to train the weights
def Train(X,Y,iterations,alpha,pocket,pocket_error):
    error=np.zeros([iterations,2])
    for i in range(iterations):
        for row,k in zip(X,range(len(X))):
            prediction=Activation(row,w)
            for j in range(4):
                w[0,j]=w[0,j]+alpha*((Y[k]-prediction)*row[j])
        error[i,0]=Accuracy(X,Y,w)
        if error[i,0] <= pocket_error:
            pocket=w
            pocket_error=error[i,0]
            #print('pocket updated: ',pocket)
        error[i,1]=pocket_error
    #print('pocket to be returned: ',pocket)        
    return pocket,error

#Create function to measure accuracy
def Accuracy(X,Y,w):
    error=0
    test=np.zeros([len(X),2])
    for k in range(len(X)):
        test[k,0]=Activation(X[k],w)
        test[k,1]=Y[k]
        if test[k,0] != test[k,1]:
            error+=1
    return error

#Run the pocket algorithm
pocket_error=Accuracy(X,Y,pocket)
w,error=Train(X,Y,iterations,alpha,pocket,pocket_error)
accuracy=Accuracy(X,Y,w)

print('Final weights: ',w)

#Plot the nuber of missclassified points
g=np.zeros([iterations,2])
f=np.zeros([iterations,2])
for i in range(iterations):
    g[i,0]=i+1
    f[i,0]=i+1
    g[i,1]=error[i,0]
    f[i,1]=error[i,1]
    
plt.plot(g[:,0],g[:,1])
plt.plot(f[:,0],f[:,1])
plt.legend(('Actual Weights','Pocket'))
plt.ylabel('No. of errors')
plt.xlabel('Iteration')
    
        