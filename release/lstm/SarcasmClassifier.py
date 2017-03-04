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






class SarcasmClassifier(BaseEstimator):
    def __init__(self,
            W=None, 
            K=300,
            num_hidden=256, 
            batch_size=16, 
            bidirectional=False, 
            grad_clip=100.0,
            max_seq_len=200, 
            num_classes=2, 
            num_epochs = 25):
            

        self.W = W
        self.K = K
        self.num_hidden = num_hidden
        self.bidirectional = bidirectional
        self.grad_clip = grad_clip
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.classifier = SarcasmLstm(W, K, num_hidden, batch_size, bidirectional, grad_clip, max_seq_len) 


