# using the following for reference:
# https://github.com/umass-semeval/semeval16/blob/master/semeval/lstm_words.py 
# https://github.com/junfenglx/reasoning_attention
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
from custom_layers import * 

from release.preprocessing.utils import str_to_bool
from release.lstm.hidey_layers import AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer



class deep_mind:
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
        print("this is the value of K: {}\n".format(K))
        num_hidden = int(num_hidden)
        batch_size = int(batch_size)
        grad_clip = int(grad_clip)
        max_seq_len = int(max_sent_len)
        max_post_len = int(kwargs["max_post_len"])
        num_classes = int(num_classes)    
        dropout = float(kwargs["dropout"])
        lambda_w = float(kwargs["lambda_w"])
        separate_attention_context = str_to_bool(kwargs["separate_attention_context"])
        separate_attention_response = str_to_bool(kwargs["separate_attention_response"])
        interaction = str_to_bool(kwargs["interaction"])
        separate_attention_context_words = str_to_bool(kwargs["separate_attention_context_words"])
        separate_attention_response_words = str_to_bool(kwargs["separate_attention_response_words"])

        print("this is the separate_attention_context: {}\n".format(separate_attention_context))

        print("this is the separate_attention_response: {}\n".format(separate_attention_response))
        print("this is the separate_attention_context_words: {}\n".format(separate_attention_context_words))

        print("this is the separate_attention_response_words: {}\n".format(separate_attention_response_words))
        print("this is the interaction: {}\n".format(interaction))


        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of responses
        idxs_context = T.itensor3('idxs_context') #imatrix
        #B x S x N matrix
        mask_context_words = T.itensor3('mask_context_words')
        #B x S matrix
        mask_context_sents = T.imatrix('mask_context_sents')
        #B x S x N tensor of batches of responses
        idxs_response = T.itensor3('idxs_response') #imatrix
        #B x S x N matrix
        mask_response_words = T.itensor3('mask_response_words')
        #B x S matrix
        mask_response_sents = T.imatrix('mask_response_sents')
        #B-long vector
        y = T.ivector('y')
        # TODO
        # Add biases, other params? 
        #lambda_w = T.scalar('lambda_w')
        #p_dropout = T.scalar('p_dropout')

        #biases = T.matrix('biases')
        #weights = T.ivector('weights')
        
        inputs = [idxs_response, mask_response_words, mask_response_sents]
        # TODO 
        # change inputs, function calls
        #idxs_context, mask_context_words, mask_context_sents
                
        #now use this as an input to an LSTM
        l_idxs_context = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),
                                            input_var=idxs_context)
        l_mask_context_words = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),input_var=mask_context_words)
        l_mask_context_sents = lasagne.layers.InputLayer(shape=(None, max_post_len),
                                                input_var=mask_context_sents)

        #if add_biases:
        #    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                 # input_var=biases)
        #now B x S x N x D
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        l_emb_rr_w_context = lasagne.layers.EmbeddingLayer(l_idxs_context, input_size=V, output_size=K,
                                                   W=W)
        l_emb_rr_w_context.params[l_emb_rr_w_context.W].remove('trainable')
#        l_hid_context = l_emb_rr_w
        #CBOW w/attn
        #now B x S x D
        if separate_attention_context_words:
            l_attention_words_context = AttentionWordLayer([l_emb_rr_w_context, l_mask_context_words], K)
            #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_context)))
            l_avg_rr_s_words_context = WeightedAverageWordLayer([l_emb_rr_w_context,l_attention_words_context])
        else:
            l_avg_rr_s_words_context = WeightedAverageWordLayer([l_emb_rr_w_context, l_mask_context_words])
        ##concats = l_avg_rr_s_words_context
        ##concats = [l_avg_rr_s_words_context]
        l_avg_rr_s_context = l_avg_rr_s_words_context

        # concats not relevant here, was just frames, sentiment etc for other task.
            
            
        #l_avg_rr_s_context = lasagne.layers.ConcatLayer(concats, axis=-1)

        # TODO
        # add highway ?
        #add MLP
        #if highway:
        #    l_avg_rr_s_context = HighwayLayer(l_avg_rr_s_context, num_units=l_avg_rr_s_context.output_shape[-1],
        #                              nonlinearity=lasagne.nonlinearities.rectify,
        #                              num_leading_axes=2)
        #    
        #l_lstm_rr_s_context = lasagne.layers.LSTMLayer(l_avg_rr_s_context, num_hidden,
        #                                       nonlinearity=lasagne.nonlinearities.tanh,
        #                                       grad_clipping=grad_clip,
        #                                       mask_input=l_mask_context_sents)
        #
        #l_lstm_rr_s_context = lasagne.layers.DropoutLayer(l_lstm_rr_s_context,p=dropout)
        l_lstm_rr_s_context = l_avg_rr_s_context
        if interaction:
            #l_hid_context = l_lstm_rr_s_context
            if separate_attention_context:
                print("separate attention context\n")
                l_attn_rr_s_context = AttentionSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents], num_hidden)        
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_attn_rr_s_context])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))
            else:
                print("just averaged context without attention\n")
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))

            l_hid_context = l_lstm_rr_avg_context
            print("interaction\n")
        else:
            print("no interaction!!! \n")
            #LSTM w/ attn
            #now B x D
            if separate_attention_context:
                print("separate attention context\n")
                l_attn_rr_s_context = AttentionSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents], num_hidden)        
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_attn_rr_s_context])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))
            else:
                print("just averaged context without attention\n")
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))

            l_hid_context = l_lstm_rr_avg_context

        # TODO 
        # change inputs, function calls
        #idxs_context, mask_context_words, mask_context_sents
                
        #now use this as an input to an LSTM
        l_idxs_response = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),
                                            input_var=idxs_response)
        l_mask_response_words = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),input_var=mask_response_words)
        l_mask_response_sents = lasagne.layers.InputLayer(shape=(None, max_post_len),
                                                input_var=mask_response_sents)

        #if add_biases:
        #    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                 # input_var=biases)
        #now B x S x N x D
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        l_emb_rr_w_response = lasagne.layers.EmbeddingLayer(l_idxs_response, input_size=V, output_size=K,
                                                   W=W)
        l_emb_rr_w_response.params[l_emb_rr_w_response.W].remove('trainable')
#        l_hid_response = l_emb_rr_w
        #CBOW w/attn
        #now B x S x D
        if separate_attention_response_words:
            l_attention_words_response = AttentionWordLayer([l_emb_rr_w_response, l_mask_response_words], K)
            #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_response)))
            l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response,l_attention_words_response])
        else:
            l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response, l_mask_response_words])
        #l_attention_words_response = AttentionWordLayer([l_emb_rr_w_response, l_mask_response_words], K)
        #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_response)))
        #l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response, l_mask_response_words])
        ##concats = l_avg_rr_s_words_response
        ##concats = [l_avg_rr_s_words_response]
        l_avg_rr_s_response = l_avg_rr_s_words_response

        # concats not relevant here, was just frames, sentiment etc for other task.
            
            
        #l_avg_rr_s_response = lasagne.layers.ConcatLayer(concats, axis=-1)

        # TODO
        # add highway ?
        #add MLP
        #if highway:
        #    l_avg_rr_s_response = HighwayLayer(l_avg_rr_s_response, num_units=l_avg_rr_s_response.output_shape[-1],
        #                              nonlinearity=lasagne.nonlinearities.rectify,
        #                              num_leading_axes=2)
        #    
        #if interaction:
        #    print("interaction\n")
        #    # add some cell init
        #    l_lstm_rr_s_response = lasagne.layers.LSTMLayer(l_avg_rr_s_response, num_hidden,
        #                                           nonlinearity=lasagne.nonlinearities.tanh,
        #                                           grad_clipping=grad_clip,cell_init=l_hid_context,
        #                                           mask_input=l_mask_response_sents)
        #else:
        #    l_lstm_rr_s_response = lasagne.layers.LSTMLayer(l_avg_rr_s_response, num_hidden,
        #                                           nonlinearity=lasagne.nonlinearities.tanh,
        #                                           grad_clipping=grad_clip,
        #                                           mask_input=l_mask_response_sents)
        #    
        #l_lstm_rr_s_response = lasagne.layers.DropoutLayer(l_lstm_rr_s_response,p=dropout)
        l_lstm_rr_s_response = l_avg_rr_s_response
        #LSTM w/ attn
        #now B x D
        if separate_attention_response:
            print("separate attention on the response\n")
            l_attn_rr_s_response = AttentionSentenceLayer([l_lstm_rr_s_response, l_mask_response_sents], num_hidden)        
            l_lstm_rr_avg_response = WeightedAverageSentenceLayer([l_lstm_rr_s_response, l_attn_rr_s_response])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_response)))
        else:
            print("just average response without attention\n")
            l_lstm_rr_avg_response = WeightedAverageSentenceLayer([l_lstm_rr_s_response, l_mask_response_sents])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_response)))

        l_hid_response = l_lstm_rr_avg_response

        # TODO
        # add more layers? biases? 
        #for num_layer in range(num_layers):
        #    l_hid_response = lasagne.layers.DenseLayer(l_hid_response, num_units=rd,
        #                                  nonlinearity=lasagne.nonlinearities.rectify)

        #    #now B x 1
        #    l_hid_response = lasagne.layers.DropoutLayer(l_hid_response, p_dropout)
        #    
        #if add_biases:
        #    l_hid_response = lasagne.layers.ConcatLayer([l_hid_response, l_biases], axis=-1)
        #    inputs.append(biases)
        #    
        #self.network = lasagne.layers.DenseLayer(l_hid_response, num_units=2,
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

        ##attention for sentences, B x S
        print('finished compiling...')
        #l_premise_linear = CustomDense(l_hid_context, K,
        #                                               nonlinearity=lasagne.nonlinearities.linear)
        #l_hypo_linear = CustomDense(l_hid_response, K,
        #                                                W=l_premise_linear.W, b=l_premise_linear.b,
        #                                                                                nonlinearity=lasagne.nonlinearities.linear)
        encoder = CustomLSTMEncoder(l_hid_context, int(K), peepholes=False, mask_input=l_mask_context_sents)
        decoder = CustomLSTMDecoder(l_hid_response, K, cell_init=encoder, peepholes=False, mask_input=l_mask_response_sents,encoder_mask_input=l_mask_context_sents,
                                                                                    attention=True, word_by_word=True)
    
        print(" custom shape: {}\n".format(get_output_shape(decoder)))
        #if interaction:
        #    l_concat = l_hid_response
        #else:
        #    l_concat = lasagne.layers.ConcatLayer([l_hid_context,l_hid_response])
        network = lasagne.layers.DenseLayer(
            decoder,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )

        self.network = network
        output = lasagne.layers.get_output(network)

        # Define objective function (cost) to minimize, mean crossentropy error
        cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

        # Compute gradient updates
        params = lasagne.layers.get_all_params(network)
        cost += lambda_w*apply_penalty(params, l2)
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
        self.val_fn = theano.function([idxs_context, mask_context_words, mask_context_sents, idxs_response, mask_response_words, mask_response_sents, y], [val_cost_fn, val_acc_fn, preds],
                                 allow_input_downcast=True,on_unused_input='warn')
        # Compile train objective
        print "Compiling training, testing, prediction functions"
        self.train = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents, y], outputs = cost, updates = grad_updates, allow_input_downcast=True,on_unused_input='warn')
        self.test = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents, y], outputs = val_acc_fn,allow_input_downcast=True,on_unused_input='warn')
        self.pred = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents, idxs_response, mask_response_words, mask_response_sents],outputs = preds,allow_input_downcast=True,on_unused_input='warn')
        if separate_attention_response:
            sentence_attention = lasagne.layers.get_output(l_attn_rr_s_response, deterministic=True)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_response = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      [sentence_attention, preds],
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_context:
            sentence_attention_context = lasagne.layers.get_output(l_attn_rr_s_context, deterministic=True)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_context = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      [sentence_attention_context,preds],
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_response_words:
            word_attention = lasagne.layers.get_output(l_attention_words_response, deterministic=True) 
            self.sentence_attention_response_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],[word_attention,preds], 
                    allow_input_downcast=True,
                    on_unused_input='warn')
        if separate_attention_context_words:
            word_attention_context = lasagne.layers.get_output(l_attention_words_context, deterministic = True) 
            self.sentence_attention_context_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],[word_attention_context,preds], 
                    allow_input_downcast=True,
                    on_unused_input='warn')

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

from release.preprocessing.utils import str_to_bool
from release.lstm.hidey_layers import AttentionWordLayer, AttentionSentenceLayer, WeightedAverageWordLayer, WeightedAverageSentenceLayer, HighwayLayer



class SarcasmLstmAttentionSeparate:
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
        print("this is the value of K: {}\n".format(K))
        num_hidden = int(num_hidden)
        batch_size = int(batch_size)
        grad_clip = int(grad_clip)
        max_seq_len = int(max_sent_len)
        max_post_len = int(kwargs["max_post_len"])
        num_classes = int(num_classes)    
        dropout = float(kwargs["dropout"])
        lambda_w = float(kwargs["lambda_w"])
        separate_attention_context = str_to_bool(kwargs["separate_attention_context"])
        separate_attention_response = str_to_bool(kwargs["separate_attention_response"])
        interaction = str_to_bool(kwargs["interaction"])
        separate_attention_context_words = str_to_bool(kwargs["separate_attention_context_words"])
        separate_attention_response_words = str_to_bool(kwargs["separate_attention_response_words"])

        print("this is the separate_attention_context: {}\n".format(separate_attention_context))

        print("this is the separate_attention_response: {}\n".format(separate_attention_response))
        print("this is the separate_attention_context_words: {}\n".format(separate_attention_context_words))

        print("this is the separate_attention_response_words: {}\n".format(separate_attention_response_words))
        print("this is the interaction: {}\n".format(interaction))


        #S x N matrix of sentences (aka list of word indices)
        #B x S x N tensor of batches of responses
        idxs_context = T.itensor3('idxs_context') #imatrix
        #B x S x N matrix
        mask_context_words = T.itensor3('mask_context_words')
        #B x S matrix
        mask_context_sents = T.imatrix('mask_context_sents')
        #B x S x N tensor of batches of responses
        idxs_response = T.itensor3('idxs_response') #imatrix
        #B x S x N matrix
        mask_response_words = T.itensor3('mask_response_words')
        #B x S matrix
        mask_response_sents = T.imatrix('mask_response_sents')
        #B-long vector
        y = T.ivector('y')
        # TODO
        # Add biases, other params? 
        #lambda_w = T.scalar('lambda_w')
        #p_dropout = T.scalar('p_dropout')

        #biases = T.matrix('biases')
        #weights = T.ivector('weights')
        
        inputs = [idxs_response, mask_response_words, mask_response_sents]
        # TODO 
        # change inputs, function calls
        #idxs_context, mask_context_words, mask_context_sents
                
        #now use this as an input to an LSTM
        l_idxs_context = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),
                                            input_var=idxs_context)
        l_mask_context_words = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),input_var=mask_context_words)
        l_mask_context_sents = lasagne.layers.InputLayer(shape=(None, max_post_len),
                                                input_var=mask_context_sents)

        #if add_biases:
        #    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                 # input_var=biases)
        #now B x S x N x D
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        l_emb_rr_w_context = lasagne.layers.EmbeddingLayer(l_idxs_context, input_size=V, output_size=K,
                                                   W=W)
        l_emb_rr_w_context.params[l_emb_rr_w_context.W].remove('trainable')
#        l_hid_context = l_emb_rr_w
        #CBOW w/attn
        #now B x S x D
        if separate_attention_context_words:
            l_attention_words_context = AttentionWordLayer([l_emb_rr_w_context, l_mask_context_words], K)
            #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_context)))
            l_avg_rr_s_words_context = WeightedAverageWordLayer([l_emb_rr_w_context,l_attention_words_context])
        else:
            l_avg_rr_s_words_context = WeightedAverageWordLayer([l_emb_rr_w_context, l_mask_context_words])
        ##concats = l_avg_rr_s_words_context
        ##concats = [l_avg_rr_s_words_context]
        l_avg_rr_s_context = l_avg_rr_s_words_context

        # concats not relevant here, was just frames, sentiment etc for other task.
            
            
        #l_avg_rr_s_context = lasagne.layers.ConcatLayer(concats, axis=-1)

        # TODO
        # add highway ?
        #add MLP
        #if highway:
        #    l_avg_rr_s_context = HighwayLayer(l_avg_rr_s_context, num_units=l_avg_rr_s_context.output_shape[-1],
        #                              nonlinearity=lasagne.nonlinearities.rectify,
        #                              num_leading_axes=2)
        #    
        l_lstm_rr_s_context = lasagne.layers.LSTMLayer(l_avg_rr_s_context, num_hidden,
                                               nonlinearity=lasagne.nonlinearities.tanh,
                                               grad_clipping=grad_clip,
                                               mask_input=l_mask_context_sents)
        
        l_lstm_rr_s_context = lasagne.layers.DropoutLayer(l_lstm_rr_s_context,p=dropout)
        if interaction:
            #l_hid_context = l_lstm_rr_s_context
            if separate_attention_context:
                print("separate attention context\n")
                l_attn_rr_s_context = AttentionSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents], num_hidden)        
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_attn_rr_s_context])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))
            else:
                print("just averaged context without attention\n")
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))

            l_hid_context = l_lstm_rr_avg_context
            print("interaction\n")
        else:
            print("no interaction!!! \n")
            #LSTM w/ attn
            #now B x D
            if separate_attention_context:
                print("separate attention context\n")
                l_attn_rr_s_context = AttentionSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents], num_hidden)        
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_attn_rr_s_context])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))
            else:
                print("just averaged context without attention\n")
                l_lstm_rr_avg_context = WeightedAverageSentenceLayer([l_lstm_rr_s_context, l_mask_context_sents])
                print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_context)))

            l_hid_context = l_lstm_rr_avg_context

        # TODO 
        # change inputs, function calls
        #idxs_context, mask_context_words, mask_context_sents
                
        #now use this as an input to an LSTM
        l_idxs_response = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),
                                            input_var=idxs_response)
        l_mask_response_words = lasagne.layers.InputLayer(shape=(None, max_post_len, max_sent_len),input_var=mask_response_words)
        l_mask_response_sents = lasagne.layers.InputLayer(shape=(None, max_post_len),
                                                input_var=mask_response_sents)

        #if add_biases:
        #    l_biases = lasagne.layers.InputLayer(shape=(None,1),
                                                 # input_var=biases)
        #now B x S x N x D
        #l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
        l_emb_rr_w_response = lasagne.layers.EmbeddingLayer(l_idxs_response, input_size=V, output_size=K,
                                                   W=W)
        l_emb_rr_w_response.params[l_emb_rr_w_response.W].remove('trainable')
#        l_hid_response = l_emb_rr_w
        #CBOW w/attn
        #now B x S x D
        if separate_attention_response_words:
            l_attention_words_response = AttentionWordLayer([l_emb_rr_w_response, l_mask_response_words], K)
            #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_response)))
            l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response,l_attention_words_response])
        else:
            l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response, l_mask_response_words])
        #l_attention_words_response = AttentionWordLayer([l_emb_rr_w_response, l_mask_response_words], K)
        #print(" attention word layer shape: {}\n".format(get_output_shape(l_attention_words_response)))
        #l_avg_rr_s_words_response = WeightedAverageWordLayer([l_emb_rr_w_response, l_mask_response_words])
        ##concats = l_avg_rr_s_words_response
        ##concats = [l_avg_rr_s_words_response]
        l_avg_rr_s_response = l_avg_rr_s_words_response

        # concats not relevant here, was just frames, sentiment etc for other task.
            
            
        #l_avg_rr_s_response = lasagne.layers.ConcatLayer(concats, axis=-1)

        # TODO
        # add highway ?
        #add MLP
        #if highway:
        #    l_avg_rr_s_response = HighwayLayer(l_avg_rr_s_response, num_units=l_avg_rr_s_response.output_shape[-1],
        #                              nonlinearity=lasagne.nonlinearities.rectify,
        #                              num_leading_axes=2)
        #    
        if interaction:
            print("interaction\n")
            # add some cell init
            l_lstm_rr_s_response = lasagne.layers.LSTMLayer(l_avg_rr_s_response, num_hidden,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=grad_clip,cell_init=l_hid_context,
                                                   mask_input=l_mask_response_sents)
        else:
            l_lstm_rr_s_response = lasagne.layers.LSTMLayer(l_avg_rr_s_response, num_hidden,
                                                   nonlinearity=lasagne.nonlinearities.tanh,
                                                   grad_clipping=grad_clip,
                                                   mask_input=l_mask_response_sents)
            
        l_lstm_rr_s_response = lasagne.layers.DropoutLayer(l_lstm_rr_s_response,p=dropout)
        #LSTM w/ attn
        #now B x D
        if separate_attention_response:
            print("separate attention on the response\n")
            l_attn_rr_s_response = AttentionSentenceLayer([l_lstm_rr_s_response, l_mask_response_sents], num_hidden)        
            l_lstm_rr_avg_response = WeightedAverageSentenceLayer([l_lstm_rr_s_response, l_attn_rr_s_response])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_response)))
        else:
            print("just average response without attention\n")
            l_lstm_rr_avg_response = WeightedAverageSentenceLayer([l_lstm_rr_s_response, l_mask_response_sents])
            print(" attention weighted average sentence layer shape: {}\n".format(get_output_shape(l_lstm_rr_avg_response)))

        l_hid_response = l_lstm_rr_avg_response

        # TODO
        # add more layers? biases? 
        #for num_layer in range(num_layers):
        #    l_hid_response = lasagne.layers.DenseLayer(l_hid_response, num_units=rd,
        #                                  nonlinearity=lasagne.nonlinearities.rectify)

        #    #now B x 1
        #    l_hid_response = lasagne.layers.DropoutLayer(l_hid_response, p_dropout)
        #    
        #if add_biases:
        #    l_hid_response = lasagne.layers.ConcatLayer([l_hid_response, l_biases], axis=-1)
        #    inputs.append(biases)
        #    
        #self.network = lasagne.layers.DenseLayer(l_hid_response, num_units=2,
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

        ##attention for sentences, B x S
        print('finished compiling...')
    
    
        if interaction:
            l_concat = l_hid_response
        else:
            l_concat = lasagne.layers.ConcatLayer([l_hid_context,l_hid_response])
        network = lasagne.layers.DenseLayer(
            l_concat,
            num_units=num_classes,
            nonlinearity=lasagne.nonlinearities.softmax
        )

        self.network = network
        output = lasagne.layers.get_output(network)

        # Define objective function (cost) to minimize, mean crossentropy error
        cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

        # Compute gradient updates
        params = lasagne.layers.get_all_params(network)
        cost += lambda_w*apply_penalty(params, l2)
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
        self.val_fn = theano.function([idxs_context, mask_context_words, mask_context_sents, idxs_response, mask_response_words, mask_response_sents, y], [val_cost_fn, val_acc_fn, preds],
                                 allow_input_downcast=True,on_unused_input='warn')
        # Compile train objective
        print "Compiling training, testing, prediction functions"
        self.train = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents, y], outputs = cost, updates = grad_updates, allow_input_downcast=True,on_unused_input='warn')
        self.test = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents, y], outputs = val_acc_fn,allow_input_downcast=True,on_unused_input='warn')
        self.pred = theano.function(inputs = [idxs_context, mask_context_words, mask_context_sents, idxs_response, mask_response_words, mask_response_sents],outputs = preds,allow_input_downcast=True,on_unused_input='warn')
        if separate_attention_response:
            sentence_attention = lasagne.layers.get_output(l_attn_rr_s_response, deterministic=True)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_response = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      [sentence_attention, preds],
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_context:
            sentence_attention_context = lasagne.layers.get_output(l_attn_rr_s_context, deterministic=True)
            #if add_biases:
            #    inputs = inputs[:-1]
            self.sentence_attention_context = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],
                                                      [sentence_attention_context,preds],
                                                      allow_input_downcast=True,
                                                      on_unused_input='warn')
        if separate_attention_response_words:
            word_attention = lasagne.layers.get_output(l_attention_words_response, deterministic=True) 
            self.sentence_attention_response_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],[word_attention,preds], 
                    allow_input_downcast=True,
                    on_unused_input='warn')
        if separate_attention_context_words:
            word_attention_context = lasagne.layers.get_output(l_attention_words_context, deterministic = True) 
            self.sentence_attention_context_words = theano.function([idxs_context, mask_context_words, mask_context_sents,idxs_response, mask_response_words, mask_response_sents],[word_attention_context,preds], 
                    allow_input_downcast=True,
                    on_unused_input='warn')

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
        
