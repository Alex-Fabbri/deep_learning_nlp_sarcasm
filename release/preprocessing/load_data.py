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
import nltk
import logging
import math
import pickle
import os
import timeit
import time
import lasagne
from lasagne.layers import get_output_shape
from release.preprocessing.process_properties import PreProcessor
from release.preprocessing.utils import str_to_bool 

def load_data(processor):

    return_dict = processor.__dict__
    path = processor.output
    both = str_to_bool(processor.both)
    top = str_to_bool(processor.topSim)
    lastSent = str_to_bool(processor.lastSent)
    attention = str_to_bool(processor.attention)
    separate = str_to_bool(processor.separate)
    batch_size = int(processor.batch_size)
    return_dict["lstm"] = processor.lstm
    data_type = processor.data_type

    target = processor.target
    print(target)
    print("The path to the input is: {}\n".format(path))

    # get the train and validation data 
    if both == False:
        print("loading just response text\n")
        train_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.nocontext.TRAIN.' +   target  + '.pkl'
        test_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.nocontext.TEST.' + target +  '.pkl'
        max_l = 200
    
    if both == True and top == False:

        if lastSent == True:
            print("loading last sentence context\n")
            train_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contextlast.TRAIN.' +   target  + '.pkl'
            test_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contextlast.TEST.' + target +  '.pkl'
            max_l = 400
        else:
            print("loading fully concatenated context\n")

            train_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contextcat.TRAIN.' +   target  + '.pkl'
            test_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contextcat.TEST.' + target +  '.pkl'
            max_l = 400

        
    if both == True and top == True:
        print("Loading top similar sentence context\n")
        max_l = 400
        if lastSent == True:
            print("Make up your mind! :P \n")
            quit()
        train_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contexttop.TRAIN.' +   target  + '.pkl'
        test_file = path + '/pkl/1_cnn/w2v_300/' + data_type + '.contexttop.TEST.' + target +  '.pkl'
        max_l = 400

    if separate == True:
         
        print("Separate loading,this overrides any previous booleans. Is this what you want?\n")
        if top == False:
            if lastSent == True:
                print("loading separate last sentence context\n")
                train_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextseplast'  +'.TRAIN.' + target + '.pkl'
                test_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextseplast'  + '.TEST.' + target + '.pkl'
                max_l = 400
            else:
                print("loading separate fully context\n")
                train_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextsep'  +'.TRAIN.' + target + '.pkl'
                test_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextsep'  + '.TEST.' + target + '.pkl'
                max_l = 400
        else:
            print("loading separate top sim context\n")
            train_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextsepsim'  +'.TRAIN.' + target + '.pkl'
            test_file = path +  '/pkl/2_cnn/w2v_300/' + data_type + '.contextsepsim'  + '.TEST.' + target + '.pkl'
            max_l = 400
            
    print "loading data...",
    #logger.error("loading data...");
    
    print("the train_file is: {}\n".format(train_file))
    x = cPickle.load(open(train_file,"rb"))
    train_data, W, word_idx_map, max_sent_len = x[0], x[1], x[2], x[3]
    return_dict["W"] = W


    max_sent_len = 50
#    max_post_len = 5 
#
#    if both == True:
#        if top == True or lastSent == True:
#            max_post_len = 6 
#            print("why are we doing this??\n")
#        else:
#            max_post_len = 10 
#            print("we shouldn't be in this for the current tests\n")
#
    max_post_len = int(processor.max_sentences_response)
    max_context_len = int(processor.max_sentences_context)
    
    #if separate == True:
    #    max_post_len = 30
    print("max post len: {}\n".format(max_post_len))            
        
    return_dict["max_sent_len_basic"] = max_l
    return_dict["max_sent_len"] = max_sent_len
    return_dict["max_post_len"] = max_post_len
    print("this is the test file: {}\n".format(test_file))
    test_data = cPickle.load(open(test_file,'rb'))
    if(separate == True):
        print("This is separate loading\n")

        # The separate model currently only deals with the word and sentence heirarchy in the attention model.
        # 
        X_train_indx_context,X_train_indx_response,y_train = text_to_indx_sentence_separate(train_data, word_idx_map, max_context_len, data_type)
        X_test_indx_context, X_test_indx_response, y_test = text_to_indx_sentence_separate(test_data, word_idx_map, max_context_len, data_type)

        # use max_post_context variable
        #max_post_len = 15
        X_train_indx_pad_context, X_train_indices_mask_sents_context, X_train_indices_mask_posts_context = text_to_indx_mask(X_train_indx_context, max_sent_len, max_context_len)
        X_train_indx_pad_response, X_train_indices_mask_sents_response, X_train_indices_mask_posts_response = text_to_indx_mask(X_train_indx_response, max_sent_len, max_post_len)
        print(X_train_indx_pad_context.shape)
        print(X_train_indx_pad_response.shape)


        X_test_indx_pad_context, X_test_indices_mask_sents_context, X_test_indices_mask_posts_context = text_to_indx_mask(X_test_indx_context, max_sent_len, max_context_len)
        X_test_indx_pad_response, X_test_indices_mask_sents_response, X_test_indices_mask_posts_response = text_to_indx_mask(X_test_indx_response, max_sent_len, max_post_len)



        training = X_train_indx_pad_context, X_train_indices_mask_sents_context, X_train_indices_mask_posts_context,X_train_indx_pad_response, X_train_indices_mask_sents_response, X_train_indices_mask_posts_response
        train_set_y = np.asarray(y_train)

        testing = X_test_indx_pad_context, X_test_indices_mask_sents_context, X_test_indices_mask_posts_context,X_test_indx_pad_response, X_test_indices_mask_sents_response, X_test_indices_mask_posts_response
        test_set_y = np.asarray(y_test)

    else:
        if (attention == False):
            print("basic model without attention\n")

            X_train_indx, y_train = text_to_indx(train_data, word_idx_map, max_l)
            X_train, X_train_mask = pad_mask(X_train_indx, max_l)
            #print("shapes\n")
            #print(X_train.shape)
            #print(X_train_mask.shape)


            # get the test data

            X_test_indx, y_test = text_to_indx(test_data, word_idx_map, max_l)
            X_test, X_test_mask = pad_mask(X_test_indx, max_l)

            #print("shapes\n")
            #print(X_test.shape)
            #print(X_test_mask.shape)

            # put into shared variables  -- only useful if using GPU
            #train_set_x, train_set_mask, train_set_y = shared_dataset_mask(X_train, X_train_mask, y_train)
            #test_set_x, test_set_mask, test_set_y = shared_dataset_mask(X_test, X_test_mask, y_test)
            train_set_x, train_set_mask, train_set_y  = X_train, X_train_mask, y_train
            test_set_x, test_set_mask, test_set_y = X_test, X_test_mask, y_test

            #train_set_x, train_set_mask, train_set_y = X_train, X_train_mask, y_train
            #test_set_x, test_set_mask, test_set_y = X_test, X_test_mask, y_test


            print "data loaded!"
            
            print "max length = " + str(max_sent_len)
            #training = np.array(zip(*[train_set_x, train_set_mask]))
            training = train_set_x, train_set_mask
            train_set_y = np.asarray(train_set_y)

            #testing = np.array(zip(*[test_set_x, test_set_mask]))
            testing = test_set_x, test_set_mask
            test_set_y = np.asarray(test_set_y)
        else:
            print("testing attention! \n")
            X_train_indx, y_train = text_to_indx_sentence(train_data, word_idx_map, max_post_len)
            X_train_indx_pad, X_train_indices_mask_sents, X_train_indices_mask_posts = text_to_indx_mask(X_train_indx, max_sent_len, max_post_len)

            X_test_indx, y_test = text_to_indx_sentence(test_data, word_idx_map, max_post_len)
            X_test_indx_pad, X_test_indices_mask_sents, X_test_indices_mask_posts = text_to_indx_mask(X_test_indx, max_sent_len, max_post_len)

            training = X_train_indx_pad, X_train_indices_mask_sents, X_train_indices_mask_posts
            train_set_y = np.asarray(y_train)

            testing = X_test_indx_pad, X_test_indices_mask_sents, X_test_indices_mask_posts
            test_set_y = np.asarray(y_test)
            #print X_train_indx[0]
            #print X_train_indx_pad[0]
            #print(indices_mask_sents[0])
            #print(indices_mask_posts[0])

    return training,train_set_y,testing,test_set_y,return_dict, test_data
    
def text_to_indx_mask(X_train_indx, max_sent_len, max_post_len):
    X_train_indx_pad = []
    indices_mask_sents = []
    indices_mask_posts = []
    
    for post in X_train_indx:
        post_length = len(post)
        sentences_length = [len(x) for x in post]
        curr_indx = pad_post(post, max_sent_len, max_post_len)
        curr_mask_sents = make_sentence_mask(post_length, max_post_len, sentences_length, max_sent_len)
        curr_mask_posts = make_mask(post_length, max_post_len)


        X_train_indx_pad.append(curr_indx)
        indices_mask_sents.append(curr_mask_sents)
        indices_mask_posts.append(curr_mask_posts)

    return np.asarray(X_train_indx_pad, dtype=np.int32), np.asarray(indices_mask_sents,dtype=np.int32), np.asarray(indices_mask_posts, dtype=np.int32)


def make_mask(post_length, max_post_len):
    return [1]*min(max_post_len, post_length) + [0]*max(0,max_post_len-post_length)

def make_sentence_mask(post_length, max_post_len, sentence_lengths, max_sent_len):
    ret = []
    for i in range(min(post_length, max_post_len)):
        ret.append([1]*min(sentence_lengths[i], max_sent_len) + [0]*max(0,max_sent_len - sentence_lengths[i]))
    for i in range(max_post_len - post_length):
        ret.append([0]*max_sent_len)
    return ret    

def pad_post(X_train_indx, max_l, max_s):
    padded_post = []
    for sentence in X_train_indx:
        padded_post.append(sentence[:max_l] + [0]*max(0, max_l-len(sentence)))
    return padded_post[:max_s] + [[0]*max_l]*max(0,(max_s-len(padded_post)))

def text_to_indx(train_data, word_idx_map, max_l):
    X = []
    y = []
    for query in train_data:
        #text = query["text"].split()
        text = query["text"]
        text = text.strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        text = nltk.word_tokenize(text)
        text = text[:max_l]
        y_val = query["y"]
        out = []
        for word in text:
            if word in word_idx_map:
                out.append(word_idx_map[word])
            else:
                # unknown word
                out.append(1)
        X.append(out)
        y.append(y_val)
    return X,y
def text_to_indx_sentence(train_data, word_idx_map, max_post_length):
    X = []
    y = []
    max_sentence_length = 50
    for query in train_data:
        text = query["text"]
        y_val = query["y"]
        sentences_arr = []
       
        try:
            text = text.decode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            text = text.decode('latin').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')

        sentences = nltk.sent_tokenize(text)
        if len(sentences) > max_post_length:
            continue
        for sentence in sentences:
            out = []
            words = nltk.word_tokenize(sentence)
            words = words[:max_sentence_length]
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)   

        X.append(sentences_arr)
        y.append(y_val)
    return X,y

def text_to_indx_sentence_separate(train_data, word_idx_map, max_context_len, data_type):
    X = []
    X_context = []
    y = []
    max_sentence_length = 50
    for query in train_data:
        context = query["x1"]
        response = query["x2"]
        y_val = query["y"]
        sentences_arr = []
        if "tweet" in data_type:
            sentences = context.split(' ||| ')
        else:

            try:
                context = context.decode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
            except:
                context = context.strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
            sentences = nltk.sent_tokenize(context)
        if len(sentences) > max_context_len:
            sentences = sentences[-max_context_len:]
        for sentence in sentences:
            out = []
            words = nltk.word_tokenize(sentence)
            words = words[:max_sentence_length]
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)   
        X_context.append(sentences_arr)
        sentences_arr = []
        try:
            response = response.decode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            response = response.strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')

        if "tweet" in data_type:
            words = nltk.word_tokenize(response)
            out = []
            for word in words:
                if word in word_idx_map:
                    out.append(word_idx_map[word])
                else:
                    out.append(1)
            sentences_arr.append(out)   
        else:
            sentences = nltk.sent_tokenize(response)
            # TODO, how to deal with long context
            #if len(sentences) > max_post_length:
            #    continue
            for sentence in sentences:
                out = []
                words = nltk.word_tokenize(sentence)
                words = words[:max_sentence_length]
                for word in words:
                    if word in word_idx_map:
                        out.append(word_idx_map[word])
                    else:
                        out.append(1)
                sentences_arr.append(out)   

        X.append(sentences_arr)
        y.append(y_val)
    return X_context, X, y

def pad_mask(X_train_indx, max_l):

    N = len(X_train_indx)
    X = np.zeros((N, max_l), dtype=np.int32)
    X_mask = np.zeros((N,max_l), dtype = np.int32)
    for i, x in enumerate(X_train_indx):
        n = len(x)
        if n < max_l:
            X[i, :n] = x
            X_mask[i, :n] = 1
        else:
            X[i, :] = x[:max_l]
            X_mask[i, :] = 1

    return X,X_mask


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)

def shared_dataset_mask(data_x,data_y, data_z, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x,
                                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z,
                                                 dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        return shared_x, shared_y, T.cast(shared_z, 'int32')

def iterate_minibatches(inputs,inputs2, targets, batch_size, shuffle=False):
    ''' Taken from the mnist.py example of Lasagne'''

    targets = np.asarray(targets)
    assert inputs.shape[0] == targets.size
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], inputs2[excerpt], targets[excerpt]

def split_train_test(X_tuple, y, train_size=.9, random_state=123):
    np.random.seed(random_state)
    tuple_len = len(X_tuple)
    x_shape = X_tuple[0].shape[0]
    #x_shape = X_tuple[0].eval().shape[0]
    train_len= int(math.floor(x_shape * train_size))
    #print("train_len: {}\n".format(train_len))
    indxs = np.random.choice(x_shape, x_shape, False)

    x_train = []
    x_holdout = []
    for i in range(tuple_len):
        #test = X_tuple[i][indxs[0:train_len]]
        #print("shape of test: {}\n".format(test.eval().shape))
        x_train.append(X_tuple[i][indxs[0:train_len]])
        x_holdout.append(X_tuple[i][indxs[train_len:]])

    #print(y)
    #print(y.shape)
    y_train = y[indxs[0:train_len]]
    y_holdout = y[indxs[train_len:]]

    return x_train, x_holdout, y_train, y_holdout 

def get_batch(X, y, batch_idxs):
    x_arr = []
    for i in range(len(X)):
        x_arr.append(X[i][batch_idxs]) # .eval())
    y = y[batch_idxs] # .eval()

    return x_arr, y

def read_model_data(model, filename):
    """Unpickles and loads parameters into a Lasagne model."""
    filename = os.path.join('./', '%s.%s' % (filename, 'params'))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)


def write_model_data(model, filename):
    """Pickels the parameters within a Lasagne model."""
    data = lasagne.layers.get_all_param_values(model)
    filename = os.path.join('./', filename)
    filename = '%s.%s' % (filename, 'params')
    with open(filename, 'w+') as f:
        pickle.dump(data, f)


