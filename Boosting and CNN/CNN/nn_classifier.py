import IPython
from numpy.random import uniform
import random
import time

import numpy as np
import glob
import os

import matplotlib.pyplot as plt


import sys

from  sklearn.neighbors import KNeighborsClassifier



class NN(): 


    def __init__(self,train_data,val_data,n_neighbors=5):

        self.train_data = train_data
        self.val_data = val_data

        self.sample_size = 400

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)


    def get_data(self, data):
        
        '''
        Get the data and observations from the dataset
        '''
        
        X = []
        y = []
        for i in data:
            X.append(i['features'].flatten())
            y.append(np.argmax(i['label']))
        return X, y
        
    
    def train_model(self): 

        '''
        Train Nearest Neighbors model
        '''
        
        X, y = self.get_data(self.train_data)
        self.model.fit(X, y)



    def get_validation_error(self):

        '''
        Compute validation error. Please only compute the error on the sample_size number 
        over randomly selected data points. To save computation. 

        '''
        
        X, y = self.get_data(self.val_data)
        X = X[:400]
        y = y[:400]
        acc = 0
        y_hat = self.model.predict(X)
        for i in range(400):
            if y[i] == y_hat[i]:
                acc += 1
                
        return acc / 400



    def get_train_error(self):

        '''
        Compute train error. Please only compute the error on the sample_size number 
        over randomly selected data points. To save computation. 
        '''
        
        X, y = self.get_data(self.train_data)
        X = X[:400]
        y = y[:400]
        acc = 0
        y_hat = self.model.predict(X)
        for i in range(400):
            if y[i] == y_hat[i]:
                acc += 1
                
        return acc / 400


