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
from release.preprocessing.load_data import split_train_test
from release.preprocessing.load_data import get_batch
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

        print("starting training")
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file = "logs/log_file_{}".format(time_stamp)
        log_file = open(log_file, "w+")

        early_stopping_heldout = .9
        X, X_heldout, y, y_heldout = split_train_test(X, y, train_size=early_stopping_heldout, random_state=123)
        #num_batches = len(X) // self.batch_size
        #best = 0
        train_size = X[0].shape[0]
        #train_size = X[0].eval().shape[0]
        n_train_batches  = train_size  // self.batch_size
        best = 0

        patience = 1000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        #validation_frequency = min(n_train_batches, patience // 2)
        validation_frequency = n_train_batches //4 
                                      # go through this many
                                      # minibatches before checking the network
                                      # on the validation set.

        print("validation frequency: {}\n".format(validation_frequency))
        best_validation_accuracy = -np.inf
        best_iter = 0
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False

        while (epoch < self.num_epochs) and (not done_looping):
            epoch = epoch + 1
            start_time_epoch = timeit.default_timer()
            print("Epoch number: {}".format(epoch))
            log_file.write("Epoch number: {}".format(epoch))
            log_file.flush()
            idxs = np.random.choice(train_size, train_size, False)

            for batch_num in range(n_train_batches):

                print(batch_num)
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)
                batch_idxs = idxs[s:e]
                X_batch, y_batch = get_batch(X, y, batch_idxs) 
                cost = self.classifier.train(*X_batch, y=y_batch)
                log_file.write("batch num: {}, cost: {}\n".format(batch_num, cost))
                
                
            


                # iteration number
                iter = (epoch - 1) * n_train_batches + batch_num

                if (iter + 1) % validation_frequency == 0:
                    print("time to check validation! ")
                    log_file.write("time to check validation! ")
                    this_validation_cost, this_validation_accuracy,_ = self.classifier.val_fn(*X_heldout, y=y_heldout)
            	    log_file.write("this is the current validation lost {}".format(this_validation_cost))
            	    log_file.write("this is the current validation accuracy {}".format(this_validation_accuracy))
                    
                    # if we got the best validation score until now
                    if this_validation_accuracy > best_validation_accuracy:
                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_accuracy > best_validation_accuracy *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

            	        best_validation_accuracy = this_validation_accuracy

                        best_iter = iter
                        best_params = self.classifier.get_params()

                log_file.flush()
                        

                if patience <= iter:
                    done_looping = True
                    break

            end_time_epoch = timeit.default_timer()
            total_time = (end_time_epoch - start_time_epoch) /60.
            print("Total time for epoch: " + str(total_time))

        
        log_file.flush()
        log_file.close()
        self.classifier.set_params(best_params)
        end_time = timeit.default_timer()
        print("the code trained for {} ".format(((end_time-start_time)/60)))
        print("Optimization finished: the best validation accuracy of {} achieved at {}".format(best_validation_accuracy, best_iter))

        return self

    
    def predict(self, X, y):
        preds = self.classifier.pred(*X)
        precision, recall, fscore, support = score(y, preds)
        return preds,[precision, recall, fscore] 

    def save(self, outfilename):
        self.classifier.save(outfilename)
