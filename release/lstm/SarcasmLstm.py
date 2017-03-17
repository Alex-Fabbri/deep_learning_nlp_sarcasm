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



class SarcasmLstm:
    def __init__(self, 
                W=None, 
                W_path=None, 
                K=300, 
                num_hidden=256,
                batch_size=None,
                grad_clip=100., 
                max_seq_len=200, 
                num_classes=2, 
                **kwargs):

        W = W
        V = len(W)
        K = int(K)
        num_hidden = int(num_hidden)
        batch_size = int(batch_size)
        grad_clip = int(grad_clip)
        max_seq_len = int(max_seq_len)
        num_classes = int(num_classes)    


        index = T.lscalar() 
        X = T.imatrix('X')
        M = T.imatrix('M')
        y = T.ivector('y')
        # Input Layer
        l_in = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=X)
        print(" l_in shape: {}\n".format(get_output_shape(l_in)))
        l_mask = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=M)
        #l_mask2 = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=M)
        #l_mask_concat = lasagne.layers.ConcatLayer([l_mask, l_mask2])

        print(" l_mask shape: {}\n".format(get_output_shape(l_mask)))
        #print(" l_mask shape: {}\n".format(get_output_shape(l_mask_concat)))

    
    
        # Embedding layer
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        print(" l_emb shape: {}\n".format(get_output_shape(l_emb)))
    
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
        l_fwd = lasagne.layers.DropoutLayer(l_fwd,p=0.5)
        print(" forward shape: {}\n".format(get_output_shape(l_fwd)))
        if kwargs["lstm"] == "bi":
            gate_params_bwd = lasagne.layers.recurrent.Gate(
                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                b=lasagne.init.Constant(0.)
            )
            cell_params_bwd = lasagne.layers.recurrent.Gate(
                W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
                W_cell=None, b=lasagne.init.Constant(0.),
                nonlinearity=lasagne.nonlinearities.tanh
            )
            l_bwd = lasagne.layers.LSTMLayer(
                     l_emb, num_units=num_hidden, grad_clipping=grad_clip,
                     nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
                     ingate=gate_params_bwd, forgetgate=gate_params_bwd, cell=cell_params_bwd,
                     outgate=gate_params_bwd, learn_init=True,
                     backwards=True
            )
            l_bwd = lasagne.layers.DropoutLayer(l_bwd,p=0.5)
            print(" backward shape: {}\n".format(get_output_shape(l_bwd)))

            # concat and dropout
            l_concat = lasagne.layers.ConcatLayer([l_fwd, l_bwd])
            #l_concat = lasagne.layers.ElemwiseSumLayer([l_fwd, l_bwd])
            l_concat_dropout = lasagne.layers.DropoutLayer(l_concat,p=0.5)
            print(" concat shape: {}\n".format(get_output_shape(l_concat)))
        else:
            l_concat_dropout = l_fwd
    
    
        network = lasagne.layers.DenseLayer(
            l_concat_dropout,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )
        #print(" network shape: {}\n".format(get_output_shape(network)))

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
        test_output = lasagne.layers.get_output(network, deterministic=True)
        val_cost_fn = lasagne.objectives.categorical_crossentropy(
            test_output, y).mean()
        preds = T.argmax(test_output, axis=1)

        val_acc_fn = T.mean(T.eq(preds, y),
                            dtype=theano.config.floatX)
        self.val_fn = theano.function([X, M, y], [val_cost_fn, val_acc_fn, preds],
                                 allow_input_downcast=True)
        concat_output = lasagne.layers.get_output(l_concat) 
        fwd_output = lasagne.layers.get_output(l_fwd) 
        bwd_output = lasagne.layers.get_output(l_bwd) 
        mask_output  = lasagne.layers.get_output(l_mask)
        #mask_concat_output  = lasagne.layers.get_output(l_mask_concat)

        self.get_concat = theano.function([X,M], [concat_output, fwd_output, bwd_output, mask_output]) #, mask_concat_output])
        #print(y_train)
        # Compile train objective
        print "Compiling training functions"
        self.train = theano.function(inputs = [X,M,y], outputs = cost, updates = grad_updates, allow_input_downcast=True)
        self.test = theano.function(inputs = [X,M,y], outputs = val_acc_fn)
        self.pred = theano.function(inputs = [X,M],outputs = preds)

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params)

    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)
