#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:53:36 2019

@author: Varunya Ilanghovan, Camilo Barrera
"""

import numpy as np
import pandas as pd
from sklearn import linear_model


#Import the dataset
dataset=pd.read_csv('linear-regression.txt',names=['x','y','z'])


"""Personal Implementation"""
#Create X matrix
X=dataset.iloc[:,0:2]
ones=np.ones([X.shape[0],1])
X = np.concatenate((ones,X),axis=1)

#Create y matrix
Y=dataset.iloc[:,2:3].values

#Create theta array
theta=np.zeros([1,3])

#Define alpha step for Gradient Descent algorithm
alpha=0.001
#Define number of iterations that algorithm will run
iterations=75000

#Function to compute cost
def Cost(X,Y,theta):
    partial=np.power(((X @ theta.T)-2),2)
    c=np.sum(partial)/(2*len(X))
    return c

#Function for gradient descent
def GradDescent(X,Y,theta,alpha,iterations):
    cost=np.zeros(iterations)
    for i in range(iterations):
        theta=theta-(alpha/len(X)) * np.sum(X * (X @ theta.T - Y),axis=0)
        cost[i]=Cost(X,Y,theta)
    return theta,cost

weights,cost=GradDescent(X,Y,theta,alpha,iterations)
print ('Weights using our own implementation: ',weights)

finalcost=Cost(X,Y,weights)
print('final cost: ',finalcost)



"""Implementation using sklearn"""

#Define variables
X1=dataset.drop(columns='z')
Z1=dataset.z

#Create linear regression object
reg=linear_model.LinearRegression()

#Train the model
reg.fit(X1,Z1)

#Regression Coefficients:
print('Weights using SKlearn: ',reg.coef_)
print('Intercept using SKlearn: ',reg.intercept_)
