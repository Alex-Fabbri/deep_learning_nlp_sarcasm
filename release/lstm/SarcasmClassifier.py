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
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import datetime





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

        time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file = "logs/log_file_{}".format(time_stamp)
        log_file = open(log_file, "w+")

        early_stopping_heldout = .9
        X, X_heldout, y, y_heldout = train_test_split(X,y, train_size=early_stopping_heldout, random_state = 123)


        #data = zip(*X)
        # data = tuples
        num_batches = len(X) // self.batch_size
        best = 0
        #training = np.array(zip(*data))
        #training = X
        #print(training[:,0,:].shape)
        
        for epoch in range(self.num_epochs):
            print("Epoch: {}\n".format(epoch))
            epoch_cost = 0

            idxs = np.random.choice(len(X), len(X), False)
            #print('Unique', len(set(idxs)))
                
            for batch_num in range(num_batches):
                print(batch_num)
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)

                batch = X[idxs[s:e]]
                #inputs = zip(*batch)
                inputs = np.array(zip(*batch))

                #X_current = np.array(inputs[0])
                #X_current_mask = np.array(inputs[1])
                y_current = y[idxs[s:e]]

                #cost = self.classifier.train(X_current, X_current_mask, y_current)
                cost = self.classifier.train(*inputs, y=y_current)
                log_file.write("Epoch: {}, batch_num: {}, cost: {}\n".format(epoch, batch_num, cost))
                log_file.flush()
                epoch_cost += cost
                

        best_params = self.classifier.get_params()
        self.classifier.set_params(best_params)

        log_file.close()
        return self

    
    def test(self, X, y):
        inputs = np.array(zip(*X))
        #X1 = data[0]
        #X1_mask = data[1]
        preds = self.classifier.pred(*inputs)
        precision, recall, fscore, support = score(y, preds)
        return preds,[precision, recall, fscore] 

    def save(self, outfilename):
        self.classifier.save(outfilename)
