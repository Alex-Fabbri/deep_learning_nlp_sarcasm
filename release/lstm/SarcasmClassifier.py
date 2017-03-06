# using the following for reference:
# https://github.com/umass-semeval/semeval16/blob/master/semeval/lstm_words.py 
import cPickle
import numpy as np
import collections
import theano
import theano.tensor as T
import re
import warnings
import sys
import pandas as pd
import logging
import math
import pickle
import os
import timeit
import time
import lasagne
from lasagne.layers import get_output_shape
from sklearn.base import BaseEstimator
from release.lstm.SarcasmLstm import SarcasmLstm
from release.preprocessing.utils import str_to_bool 
from sklearn.model_selection import train_test_split






class SarcasmClassifier(BaseEstimator):
    def __init__(self,
            W=None, 
            W_path=None, 
            K=300,
            num_hidden=256, 
            batch_size=16, 
            bidirectional=False, 
            grad_clip=100.0,
            max_seq_len=200, 
            num_classes=2, 
            num_epochs = 25, 
            **kwargs):
            #):
           # **kwargs):
            

        #self.W = W
        self.W = W
        self.K = int(K)
        self.num_hidden = int(num_hidden)
        self.bidirectional = str_to_bool(bidirectional)
        self.grad_clip = float(grad_clip)
        self.max_seq_len = int(max_seq_len)
        self.num_classes = int(num_classes)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.classifier = SarcasmLstm(W=W,batch_size=self.batch_size,max_seq_len=self.max_seq_len) 


    def fit(self, X, y):

        early_stopping_heldout = .9
        if early_stopping_heldout:
            X, X_heldout, y, y_heldout = train_test_split(X,
                                                          y,
                                                          train_size=early_stopping_heldout,
                                                          )
            print('Train Fold: {} Heldout: {}'.format(collections.Counter(y), collections.Counter(y_heldout)))

        data = zip(*X)
        X = np.array(data[0])
        num_batches = X.shape[0] // self.batch_size
        best = 0
        training = np.array(zip(*data))
        
        for epoch in range(self.num_epochs):
            epoch_cost = 0

            idxs = np.random.choice(X.shape[0], X.shape[0], False)
            #print('Unique', len(set(idxs)))
                
            for batch_num in range(num_batches+1):
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)

                batch = training[idxs[s:e]]
                inputs = zip(*batch)

                X_current = np.array(inputs[0])
                X_current_mask = np.array(inputs[1])
                y_current = y[idxs[s:e]]

                #print(X_current.shape)
                #print(X_current_mask.shape)
                #print(y_current.shape)

                #y_current = y[idxs[s:e]]
                cost = self.classifier.train(X_current, X_current_mask, y_current)
                print(epoch, batch_num, cost)
                #epoch_cost += cost

        #    if early_stopping_heldout:
        #        scores = self.decision_function(zip(zip(*X_heldout)[:-1]))
        #        auc_score = roc_auc_score(zip(*X_heldout)[:-1], scores)
        #        if self.verbose:
        #            print('{} ROC AUC: {}'.format(outputfile, auc_score))
        #        if auc_score > best:
        #            best = auc_score
        #            best_params = self.classifier.get_params()

        #     print(epoch_cost)

        #if best > 0:
        #    self.classifier.set_params(best_params)

        return self

    def predict(self, X):
        scores = self.decision_function(X)
        return scores > .5
    
    def decision_function(self, X):
        inputs = zip(*X)
        scores = self.classifier.predict(*inputs)
        return scores

    def save(self, outfilename):
        self.classifier.save(outfilename)
