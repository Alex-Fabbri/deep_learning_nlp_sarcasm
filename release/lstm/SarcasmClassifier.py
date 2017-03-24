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
from release.lstm.SarcasmLstmAttention import SarcasmLstmAttention
from release.lstm.SarcasmLstmAttentionSeparate import SarcasmLstmAttentionSeparate
from release.preprocessing.utils import str_to_bool 
from release.preprocessing.load_data import split_train_test
from release.preprocessing.load_data import get_batch
from sklearn.metrics import precision_recall_fscore_support as score
from datetime import datetime





class SarcasmClassifier(BaseEstimator):
    def __init__(self,**kwargs):

        self.max_seq_len = int(kwargs["max_sent_len"])
        self.num_epochs = int(kwargs["num_epochs"])
        self.batch_size = int(kwargs["batch_size"])
        if kwargs["separate"] == "False":
            if kwargs["attention"] == "True":
                self.classifier = SarcasmLstmAttention(**kwargs)
            else:
                self.classifier = SarcasmLstm(**kwargs) 
        else:
                self.classifier = SarcasmLstmAttentionSeparate(**kwargs) 
                print("Attention with separating context and response!\n")

    def fit(self, X, y, log_file):

        print("starting training")
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

        #print("validation frequency: {}\n".format(validation_frequency))
        best_validation_accuracy = -np.inf
        best_iter = 0
        start_time = timeit.default_timer()
        epoch = 0
        done_looping = False

        while (epoch < self.num_epochs) and (not done_looping):
            epoch = epoch + 1
            start_time_epoch = timeit.default_timer()
            print("Epoch number: {}\n".format(epoch))
            log_file.write("Epoch number: {}\n".format(epoch))
            log_file.flush()
            idxs = np.random.choice(train_size, train_size, False)

            for batch_num in range(n_train_batches):

                #print(batch_num)
                s = self.batch_size * batch_num
                e = self.batch_size * (batch_num+1)
                batch_idxs = idxs[s:e]
                X_batch, y_batch = get_batch(X, y, batch_idxs) 
                # use to check output, get numpy arrays
                #concat_output, fwd_output, bwd_output, l_mask = self.classifier.get_concat(*X_batch)
                #print(l_mask[0,:])
                #print(l_mask_concat[0,:])
                #print(concat_output.shape)
                #print(concat_output[0,:,0])
                #print(fwd_output[0,197,:])
                #print(bwd_output[0,:,0])
                #print("forward last")
                #print(fwd_output[0,190:197,0])
                #print("backward last")
                #print(bwd_output[0,190:197,0])


                cost = self.classifier.train(*X_batch, y=y_batch)
                log_file.write("batch num: {}, cost: {}\n".format(batch_num, cost))


                # iteration number
                iter = (epoch - 1) * n_train_batches + batch_num

                if (iter + 1) % validation_frequency == 0:
                    this_validation_cost, this_validation_accuracy,_ = self.classifier.val_fn(*X_heldout, y=y_heldout)
            	    log_file.write("this is the current validation lost {}\n".format(this_validation_cost))
            	    log_file.write("this is the current validation accuracy {}\n".format(this_validation_accuracy))
                    
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
        self.classifier.set_params(best_params)
        end_time = timeit.default_timer()
        print("the code trained for {}\n".format(((end_time-start_time)/60)))
        print("Optimization finished: the best validation accuracy of {} achieved at {}\n".format(best_validation_accuracy, best_iter))

        return self

    
    def predict(self, X, y):
        preds = self.classifier.pred(*X)
        precision, recall, fscore, support = score(y, preds)
        return preds,[precision, recall, fscore] 

    def save(self, outfilename):
        self.classifier.save(outfilename)
