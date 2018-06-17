# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 17:40:17 2017

A basic wrapper for Keras to work with sklearn library functions and classes.

Can accept different model structures, i.e. number and type of hidden layers,
via 'structure' parameter in the constructor. Can be used with GridSearchCV,
cross_val_score, cross_val_predict etc.

Expects two-dimensional input and output, i.e.:
    _, input_dim = np.shape(X)
    _, output_dim = np.shape(y)

@author: Karolis
"""
import warnings
import numpy as np

from sklearn.base import BaseEstimator

from keras.models import Sequential
from keras.layers import Dense, Dropout

warnings.filterwarnings("ignore", category=DeprecationWarning)


default = [(Dense, {'units' : 512, 
                    'activation' : 'relu'}), 
           (Dropout, {'rate' : 0.5}), 
           (Dense, {'activation' : 'softmax'})]

class KerasClassifier(BaseEstimator):
    
    def __init__(self, 
                 structure=default,
                 batch_size=1,
                 epochs=100,
                 optimizer='sgd',
                 loss='categorical_crossentropy',
                 scoring='accuracy'):
        
        self.structure = default
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.scoring = scoring
    
    
    def __build_model(self, input_dim, output_dim):
        self.model = Sequential()
        
        # First hidden layer
        layer, params = self.structure[0]
        params.update({'input_shape' : (input_dim,)})
        self.model.add(layer(**params))
        
        for layer, params in self.structure[1:-1]:
            self.model.add(layer(**params))
          
        # Output layer
        layer, params = self.structure[-1]
        params.update({'units' : output_dim})
        self.model.add(layer(**params))
    
        self.model.compile(loss=self.loss, 
                           optimizer=self.optimizer, 
                           metrics=[self.scoring])
        
    
    def __fit(self, X, y):
        try:        
            _, input_dim = np.shape(X)
            _, output_dim = np.shape(y)
        except:
            X = np.array(X, ndmin=2)            
            y = np.array(y, ndmin=2)
            
        _, input_dim = np.shape(X)
        _, output_dim = np.shape(y)
            
        self.__build_model(input_dim, output_dim)
        
        self.model.fit(X, 
                       y,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=0)
    
    
    def fit(self, X, y=None):        
        self.__fit(X, y)
        return self
    
    
    def predict(self, X):
        return self.model.predict(X, self.batch_size)
    
    
    def score(self, X, y):
        return self.model.evaluate(X, y)[1]