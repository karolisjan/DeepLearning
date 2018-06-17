# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 17:28:47 2017

@author: Karolis
"""

import itertools
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


def plot_confusion_matrix(y_true, y_hat,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    confusion_matrix = metrics.confusion_matrix(y_true, y_hat)
    classes = np.unique(y_true)
    
    fig = plt.figure(figsize=(8, 6), dpi=90)
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        confusion_matrix = np.round((confusion_matrix.astype('float') 
                                     / confusion_matrix.sum(axis=1)[:, np.newaxis]), 2)

    thresh = confusion_matrix.max() / 2.
    for i, j in itertools.product(range(confusion_matrix.shape[0]), 
                                  range(confusion_matrix.shape[1])):
        
        plt.text(j, 
                 i, 
                 confusion_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusion_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
            
        
def visualise_predictions(X, y_true, y_hat):        
    reduced_X = TruncatedSVD(n_components=50).fit_transform(X)
    reduced_X = TSNE(n_components=2).fit_transform(reduced_X) 
    
    colors = ['green' if y else 'red' for y in y_hat]
    
    fig, ax = plt.subplots(ncols=2, figsize=(8, 6), dpi=90)
    ax[0].scatter(reduced_X[:, 0], reduced_X[:, 1], color=colors)
    ax[0].set_title('Predicted')
    
    colors = ['green' if y else 'red' for y in y_true]
    ax[1].scatter(reduced_X[:, 0], reduced_X[:, 1], color=colors)
    ax[1].set_title('True')
    
    plt.tight_layout()
    plt.show()
	
	
def report_results(cv_scores, train_scores=None): 
    print("Cross-Validation (CV) results:\n")
    if train_scores != None:
        print("Mean training score: %.2f%% +- %.2f%%" % (train_scores.mean() * 100, 
                                                         train_scores.std() * 200))
    print("Min test score: %.2f%% \t\t\t\t<- worst case" % (cv_scores.min() * 100))
    print("Max test score: %.2f%% \t\t\t\t<- best case" % (cv_scores.max() * 100))
    print("Mean test score: %.2f%% +- %.2f%% (95%% conf.) \t<- expected performance" % 
          (cv_scores.mean() * 100, cv_scores.std() * 200))
    
    print("\n95/100 times the test score will be in range %.2f-%.2f%%" %
         (((cv_scores.mean() - 2 * cv_scores.std()) * 100), 
          ((cv_scores.mean() + 2 * cv_scores.std()) * 100)))