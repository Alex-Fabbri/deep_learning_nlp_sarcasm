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
from lasagne.regularization import apply_penalty, l2

from release.lstm.hidey_layers import AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer



class SarcasmLstmAttention:
    def __init__(self, 
                W=None, 
                W_path=None, 
                K=300, 
                num_hidden=256,
                batch_size=None,
                grad_clip=100., 
                max_sent_len=200, 
                num_classes=2, 
                **kwargs):

        W = W
        V = len(W)
        K = int(K)
        num_hidden = int(num_hidden)
        batch_size = int(batch_size)
        grad_clip = int(grad_clip)
        max_seq_len = int(max_sent_len)
        max_post_len = int(kwargs["max_post_len"])
        num_classes = int(num_classes)    

        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of posts
        idxs_post = T.itensor3('idxs_post') #imatrix
        #B x S x N matrix
        mask_post_words = T.itensor3('mask_post_words')
        #B x S matrix
        mask_post_sents = T.imatrix('mask_post_sents')
        #B-long vector
        y = T.ivector('y')
        # TODO
        # Add biases, other params? 
        #lambda_w = T.scalar('lambda_w')
        #p_dropout = T.scalar('p_dropout')

        #biases = T.matrix('biases')
        #weights = T.ivector('weights')
        
        inputs = [idxs_post, mask_post_words, mask_post_sents]
                
        #now use this as an input to an LSTM
        l_idxs_post = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),
                                            input_var=idxs_post)
        l_mask_post_words = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),input_var=mask_post_words)
        l_mask_post_sents = lasagne.layers.InputLayer(shape=(None, max_post_len),
                                                input_var=mask_post_sents)

        #if add_biases:
        #    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                 # input_var=biases)
        #now B x S x N x D
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        l_emb_rr_w = lasagne.layers.EmbeddingLayer(l_idxs_post, input_size=V, output_size=K,
                                                   W=W)
        l_emb_rr_w.params[l_emb_rr_w.W].remove('trainable')
#        l_hid = l_emb_rr_w
        #CBOW w/attn
        #now B x S x D
        #l_attention_words = AttentionWordLayer([l_emb_rr_w, l_mask_post_words], K)
        #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words)))
        #l_avg_rr_s_words = WeightedAverageWordLayer([l_emb_rr_w, l_attention_words])
        #l_attention_words = AttentionWordLayer([l_emb_rr_w, l_mask_post_words], K)
        #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words)))
        l_avg_rr_s_words = WeightedAverageWordLayer([l_emb_rr_w, l_mask_post_words])
        ##concats = l_avg_rr_s_words
        ##concats = [l_avg_rr_s_words]
        l_avg_rr_s = l_avg_rr_s_words

        # concats not relevant here, was just frames, sentiment etc for other task.
            
            
        #l_avg_rr_s = lasagne.layers.ConcatLayer(concats, axis=-1)

        # TODO
        # add highway ?
        #add MLP
        #if highway:
        #    l_avg_rr_s = HighwayLayer(l_avg_rr_s, num_units=l_avg_rr_s.output_shape[-1],
        #                              nonlinearity=lasagne.nonlinearities.rectify,
        #                              num_leading_axes=2)
        #    
        l_lstm_rr_s = lasagne.layers.LSTMLayer(l_avg_rr_s, num_hidden,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=grad_clip,
                                               mask_input=l_mask_post_sents)
        
        l_hid = l_lstm_rr_s
        #LSTM w/ attn
        #now B x D
        l_attn_rr_s = AttentionSentenceLayer([l_lstm_rr_s, l_mask_post_sents], num_hidden)        
        l_lstm_rr_avg = WeightedAverageSentenceLayer([l_lstm_rr_s, l_attn_rr_s])
        print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg)))
        l_hid = l_lstm_rr_avg

        # TODO
        # add more layers? biases? 
        #for num_layer in range(num_layers):
        #    l_hid = lasagne.layers.DenseLayer(l_hid, num_units=rd,
        #                                  nonlinearity=lasagne.nonlinearities.rectify)

        #    #now B x 1
        #    l_hid = lasagne.layers.DropoutLayer(l_hid, p_dropout)
        #    
        #if add_biases:
        #    l_hid = lasagne.layers.ConcatLayer([l_hid, l_biases], axis=-1)
        #    inputs.append(biases)
        #    
        #self.network = lasagne.layers.DenseLayer(l_hid, num_units=2,
        #                                         nonlinearity=T.nnet.sigmoid)
        #
        #predictions = lasagne.layers.get_output(self.network).ravel()
        #
        #xent = lasagne.objectives.binary_crossentropy(predictions, gold)
        #loss = lasagne.objectives.aggregate(xent, weights, mode='normalized_sum')
        #
        #params = lasagne.layers.get_all_params(self.network, trainable=True)
        #
        # TODO
        ##add regularization? different gradient technique?
        #loss += lambda_w*apply_penalty(params, l2)

        #updates = lasagne.updates.nesterov_momentum(loss, params,
        #                                            learning_rate=learning_rate, momentum=0.9)

        #print('compiling...')
        #train_outputs = loss
        #self.train = theano.function(inputs + [gold, lambda_w, p_dropout, weights],
        #                             train_outputs,
        #                              updates=updates,
        #                              allow_input_downcast=True,
        #                              on_unused_input='warn')
        #print('...')
        #test_predictions = lasagne.layers.get_output(self.network, deterministic=True).ravel()
        #
        #self.predict = theano.function(inputs,
        #                               test_predictions,
        #                               allow_input_downcast=True,
        #                              on_unused_input='warn')

        #test_acc = T.mean(T.eq(test_predictions > .5, gold),
        #                                    dtype=theano.config.floatX)
        #print('...')
        #test_loss = lasagne.objectives.binary_crossentropy(test_predictions,
        #                                                    gold).mean()        
        #self.validate = theano.function(inputs + [gold, lambda_w, p_dropout, weights],
        #                                [loss, test_acc],
        #                              on_unused_input='warn')

        print('...')
        #attention for words, B x S x N        
         # TODO
        #word_attention = lasagne.layers.get_output(AttentionWordLayer([l_emb_rr_w, l_mask_post_words], K,
        #                                                              W_w = l_attention_words.W_w,
        #                                                              u_w = l_attention_words.u_w,
        #                                                              #b_w = l_attention_words.b_w,
        #                                                              normalized=False))
        #self.word_attention = theano.function([idxs_post,
        #                                       mask_post_words],
        #                                       word_attention,
        #                                       allow_input_downcast=True,
        #                                       on_unused_input='warn')

        #attention for sentences, B x S
        # TODO
        #sentence_attention = lasagne.layers.get_output(l_attn_rr_s)
        ##if add_biases:
        ##    inputs = inputs[:-1]
        #self.sentence_attention = theano.function(inputs,
        #                                          sentence_attention,
        #                                          allow_input_downcast=True,
        #                                          on_unused_input='warn')
        print('finished compiling...')
    
    
        network = lasagne.layers.DenseLayer(
            l_hid,
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
        test_output = lasagne.layers.get_output(network, deterministic=True)
        val_cost_fn = lasagne.objectives.categorical_crossentropy(
            test_output, y).mean()
        preds = T.argmax(test_output, axis=1)

        val_acc_fn = T.mean(T.eq(preds, y),
                            dtype=theano.config.floatX)

        self.val_fn = theano.function([idxs_post, mask_post_words, mask_post_sents, y], [val_cost_fn, val_acc_fn, preds],
                                 allow_input_downcast=True,on_unused_input='warn')
        # Compile train objective
        print "Compiling training, testing, prediction functions"
        self.train = theano.function(inputs = [idxs_post, mask_post_words, mask_post_sents, y], outputs = cost, updates = grad_updates, allow_input_downcast=True,on_unused_input='warn')
        self.test = theano.function(inputs = [idxs_post, mask_post_words, mask_post_sents, y], outputs = val_acc_fn,allow_input_downcast=True,on_unused_input='warn')
        self.pred = theano.function(inputs = [idxs_post, mask_post_words, mask_post_sents],outputs = preds,allow_input_downcast=True,on_unused_input='warn')

    def get_params(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.network, params)

    def save(self, filename):
        params = self.get_params()
        np.savez_compressed(filename, *params)

def load(model, filename):
    params = np.load(filename)
    param_keys = map(lambda x: 'arr_' + str(x), sorted([int(i[4:]) for i in params.keys()]))
    param_values = [params[i] for i in param_keys]
    lasagne.layers.set_all_param_values(model.network, param_values)
        
