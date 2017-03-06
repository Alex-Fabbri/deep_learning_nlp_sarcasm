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
from sklearn.base import BaseEstimator
from release.lstm.SarcasmLstm import SarcasmLstm
from release.preprocessing.utils import str_to_bool 






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
        self.classifier = SarcasmLstm(W=W) 


