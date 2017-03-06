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


warnings.filterwarnings("ignore")   

logger = logging.getLogger('myapp')
hdlr = logging.FileHandler('log_all')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.WARNING)

class SarcasmLstm:
    def __init__(self, 
                W=None, 
                W_path=None, 
                K=300, 
                num_hidden=256,
                batch_size=None,
                bidirectional=False, 
                grad_clip=100., 
                max_seq_len=200, 
                num_classes=2):

        W = W
        V = len(W)
        print(V)
        max_seq_len = 198 
        batch_size = 16
        num_hidden = 256
        K = 300
        num_classes = 2
        grad_clip = 100

        index = T.lscalar() 
        X = T.imatrix('X')
        M = T.imatrix('M')
        y = T.ivector('y')
        # Input Layer
        l_in = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=X)
        l_mask = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=M)

    
    
        # Embedding layer
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
    
        # add droput
        l_emb = lasagne.layers.DropoutLayer(l_emb, p=0.2)
    
        # Use orthogonal Initialization for LSTM gates
        gate_params = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            b=lasagne.init.Constant(0.)
        )
        cell_params = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
            W_cell=None, b=lasagne.init.Constant(0.),
            nonlinearity=lasagne.nonlinearities.tanh
        )
    
        l_fwd = lasagne.layers.LSTMLayer(
            l_emb, num_units=num_hidden, grad_clipping=grad_clip,
            nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
            ingate=gate_params, forgetgate=gate_params, cell=cell_params,
            outgate=gate_params, learn_init=True
        )
    
    
        network = lasagne.layers.DenseLayer(
            l_fwd,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )
        self.network = network
        output = lasagne.layers.get_output(network)

        # Define objective function (cost) to minimize, mean crossentropy error
        cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

        # Compute gradient updates
        params = lasagne.layers.get_all_params(network)
        # grad_updates = lasagne.updates.nesterov_momentum(cost, params,learn_rate)
        grad_updates = lasagne.updates.adam(cost, params)
        #learn_rate = .01
        #grad_updates = lasagne.updates.adadelta(cost, params, learn_rate)
        log_path = "logs/"
        test_output = lasagne.layers.get_output(network, deterministic=True)
        val_cost_fn = lasagne.objectives.categorical_crossentropy(
            test_output, y).mean()
        preds = T.argmax(test_output, axis=1)

        val_acc_fn = T.mean(T.eq(preds, y),
                            dtype=theano.config.floatX)
        val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds],
                                 allow_input_downcast=True)
        log_file = open(log_path + "training_log_" +
                        time.strftime('%m%d%Y_%H%M%S'), "w+")
        #print(y_train)
        # Compile train objective
        print "Compiling training functions"
        self.train = theano.function(inputs = [X,M,y], outputs = cost, updates = grad_updates, allow_input_downcast=True)
        #self.train_model = theano.function(
        #    inputs=[index],
        #    outputs=cost,
        #    updates=grad_updates,
        #    givens={
        #        X: train_set_x[index * batch_size: (index + 1) * batch_size],
        #        M: train_set_x_mask[index * batch_size: (index +1) * batch_size],
        #        y: train_set_y[index * batch_size: (index + 1) * batch_size]
        #    }
        #)
        #self.validate_model = theano.function(
        #    inputs=[index],
        #    outputs=[val_cost_fn, val_acc_fn],
        #    givens={
        #        X: valid_set_x[index * batch_size:(index + 1) * batch_size],
        #        M: valid_set_x_mask[index * batch_size: (index +1) * batch_size],
        #        y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        #    }
        #)
        #self.test_model = theano.function(
        #    inputs=[index],
        #    outputs=[val_acc_fn],
        #    givens={
        #        X: test_set_x[index * batch_size:(index + 1) * batch_size],
        #        M: test_set_x_mask[index * batch_size: (index +1) * batch_size],
        #        y: test_set_y[index * batch_size:(index + 1) * batch_size]
        #    }
        #)

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params)

    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)

