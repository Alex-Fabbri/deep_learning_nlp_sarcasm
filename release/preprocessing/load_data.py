# using the following for reference:
# https://github.com/umass-semeval/semeval16/blob/master/semeval/lstm_words.py 
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
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

def load_data(target, path, hidden_units, both, top, batch_size):

    # get the train and validation data 
    if both == False:
        train_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.nocontext.TRAIN.' +   target  + '.pkl'
        test_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.nocontext.TEST.' + target +  '.pkl'
    
    if both == True:
        train_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.contextcat.TRAIN.' +   target  + '.pkl'
        test_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.contextcat.TEST.' + target +  '.pkl'

        
    if both == True and top == True:
        train_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.contexttop.TRAIN.' +   target  + '.pkl'
        test_file = path + '/pkl/1_cnn/w2v_300/' + 'ucsc.contexttop.TEST.' + target +  '.pkl'
    
    print "loading data...",
    logger.error("loading data...");
    
    x = cPickle.load(open(train_file,"rb"))
    train_data, W, word_idx_map, max_l = x[0], x[1], x[2], x[3]
    # print(word_idx_map['i'])
    # print(word_idx_map['the'])
    # print(train_data[0])
    #print(word_idx_map)
    X_train_indx, y_train = text_to_indx(train_data, word_idx_map)
    X_train, X_train_mask = pad_mask(X_train_indx, max_l)

    # print(X_train_indx[0])
    # print(len(X_train_indx[0]))
    
    # print(X_train[0])
    # print(X_train_mask[0])
    # print(sum(X_train_mask[0]))
    # print("SUMMED")
    train_data = np.asarray(train_data)


    print("\n train_data.shape: {}\n".format(train_data.shape))
    n_batches = int(math.ceil(train_data.shape[0]/float(batch_size)))
    n_train_batches = int(np.round(n_batches*0.9))
    # print(n_batches)
    # print(n_train_batches)
    # print 'n_batches: ', n_batches
    # print 'n_train_batches: ', n_train_batches
    train_set_x = X_train[:n_train_batches*batch_size,:]
    train_set_mask = X_train_mask[:n_train_batches*batch_size,:]
    train_set_y = y_train[:n_train_batches*batch_size]

    print 'train_set_x.shape: ', train_set_x.shape
    val_set_x = X_train[n_train_batches*batch_size:,:]
    val_set_mask = X_train_mask[n_train_batches*batch_size:,:]
    val_set_y = y_train[n_train_batches*batch_size:]

    print 'val_set_x: ', val_set_x.shape
    if val_set_x.shape[0] % batch_size > 0:
        #print("shape doesn't match\n")
        extra_data_num = batch_size - val_set_x.shape[0] % batch_size
        new_set = np.append(val_set_x, val_set_x[:extra_data_num], axis=0)
        new_set_mask = np.append(val_set_mask, val_set_mask[:extra_data_num], axis = 0)
        new_set_y = np.append(val_set_y, val_set_y[:extra_data_num], axis = 0)
        # might be possible that we still do not have the proper batch size - 
        # in that case - for remaining - add from "training" data
        val_set_x = new_set
        val_set_mask = new_set_mask
        val_set_y = new_set_y
        if val_set_x.shape[0] % batch_size > 0:
             extra_data_num = batch_size - val_set_x.shape[0] % batch_size
             new_set = np.append(val_set_x, train_set_x[:extra_data_num], axis=0)
             new_set_mask = np.append(val_set_mask, train_set_mask[:extra_data_num], axis = 0)
             new_set_y = np.append(val_set_y, train_set_y[:extra_data_num], axis = 0)

             val_set_x = new_set
             val_set_mask = new_set_mask
             val_set_y = new_set_y
    #print 'val_set_x after adjustment: ', val_set_x.shape
    print 'train size =', train_set_x.shape, ' val size after adjustment =', val_set_x.shape 

    # get the test data

    test_data = cPickle.load(open(test_file,'rb'))
    X_test_indx, y_test = text_to_indx(test_data, word_idx_map)
    X_test, X_test_mask = pad_mask(X_test_indx, max_l)
    # print(test_data[0]['text'])
    # print(X_test_indx[0])
    # print(X_test[0])
    # print(X_test_mask[0])

    # put into shared variables  -- only useful if using GPU
    train_set_x, train_set_mask, train_set_y = shared_dataset_mask(train_set_x, train_set_mask, train_set_y)
    val_set_x, val_set_mask, val_set_y  = shared_dataset_mask(val_set_x, val_set_mask, val_set_y)
    test_set_x, test_set_mask, test_set_y = shared_dataset_mask(X_test, X_test_mask, y_test)
    #test_set_x, test_set_mask, test_set_y = X_test, X_test_mask, y_test


    print "data loaded!"
    
    print "max length = " + str(max_l)

    return train_set_x, train_set_mask, train_set_y, val_set_x, val_set_mask, val_set_y, test_set_x, test_set_mask, test_set_y, word_idx_map, W, max_l
