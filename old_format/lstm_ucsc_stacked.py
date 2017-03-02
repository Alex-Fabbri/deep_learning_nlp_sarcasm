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



def iterate_minibatches2(inputs,inputs2, inputs1_mask, inputs2_mask, targets, batch_size, shuffle=False):
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
        yield inputs[excerpt], inputs2[excerpt], inputs1_mask[excerpt], inputs2_mask[excerpt],targets[excerpt]

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
                                               dtype='int32'),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        return shared_x, shared_y

def shared_dataset_mask(data_x,data_y, data_z, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """

        shared_x = theano.shared(np.asarray(data_x,
                                                 dtype='int32'),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype='int32'),
                                 borrow=borrow)
        shared_z = theano.shared(np.asarray(data_z,
                                                 dtype='int32'),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue

        return shared_x, shared_y,shared_z


def text_to_indx2(train_data, word_idx_map):
    X1 = []
    X2 = []
    y = []
    for query in train_data:
        x1 = query["x1"].split()
        x2 = query["x2"].split()
        y_val = query["y"]
        out1 = []
        out2 = []
        for word in x1:
            if word in word_idx_map:
                out1.append(word_idx_map[word])
            else:
                # unknown word
                out1.append(1)
        for word in x2:
            if word in word_idx_map:
                out2.append(word_idx_map[word])
            else:
                out2.append(1)
        X1.append(out1)
        X2.append(out2)
        y.append(y_val)
    return X1,X2,y

def pad_mask2(X1_train_indx, X2_train_indx, max_x1, max_x2):

    N = len(X1_train_indx)
    X1 = np.zeros((N, max_x1), dtype=np.int32)
    X2 = np.zeros((N, max_x2), dtype = np.int32)
    X1_mask = np.zeros((N,max_x1), dtype = np.int32)
    X2_mask = np.zeros((N, max_x2), dtype = np.int32)
    for i, x in enumerate(X1_train_indx):
        n = len(x)
        if n < max_x1:
            X1[i, :n] = x
            X1_mask[i, :n] = 1
        else:
            X1[i, :] = x[:max_x1]
            X1_mask[i, :] = 1
    for i, x in enumerate(X2_train_indx):
        n = len(x)
        if n < max_x2:
            X2[i,:n] = x
            X2_mask[i,:n] = 1
        else:
            X2[i,:] = x[:max_x2]
            X2_mask[i,:] = 1


    return X1,X1_mask, X2, X2_mask

def build_model(W = None, # word embeddings
                K=300,  # dimensionality of embeddings
                num_hidden=256,  # number of hidden_units
                batch_size=None,  # size of each batch (None for variable size)
                input_var1=None,  # theano variable for input
                input_var2=None,
                mask_var1=None,  # theano variable for input mask
                mask_var2=None,
                bidirectional=False,  # whether to use bi-directional LSTM
                mean_pooling=False,
                grad_clip=100.,  # gradients above this will be clipped
                max_seq_len1=200,  # maximum lenght of a sequence 
                max_seq_len2=200,
                num_classes = 2):

    V = len(W)
    # Input layer for the response 
    l_in1 = lasagne.layers.InputLayer((batch_size, max_seq_len1), input_var=input_var1)
    l_mask1 = lasagne.layers.InputLayer((batch_size, max_seq_len1), input_var=mask_var1)

    # Embedding layer
    l_emb1 = lasagne.layers.EmbeddingLayer(l_in1, input_size=V, output_size=K, W=W)

    # Use orthogonal Initialization for LSTM gates
    gate_params1 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )
    cell_params1 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )
    l_fwd1 = lasagne.layers.LSTMLayer(
        l_emb1, num_units=num_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask1,
        ingate=gate_params1, forgetgate=gate_params1, cell=cell_params1,
        outgate=gate_params1, learn_init=True
    )
    l_fwd1 = lasagne.layers.DropoutLayer(l_fwd1, p=0.5)

    # input layer for the context 
    l_in2 = lasagne.layers.InputLayer((batch_size, max_seq_len2), input_var=input_var2)
    l_mask2 = lasagne.layers.InputLayer((batch_size, max_seq_len2), input_var=mask_var2)
    # Embedding layer
    l_emb2 = lasagne.layers.EmbeddingLayer(l_in2, input_size=V, output_size=K, W=W)

    # Use orthogonal Initialization for LSTM gates
    gate_params2 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )
    cell_params2 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )
    l_fwd2 = lasagne.layers.LSTMLayer(
        l_emb2, num_units=num_hidden, grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask2,
        ingate=gate_params2, forgetgate=gate_params2, cell=cell_params2,
        outgate=gate_params2, learn_init=True
    )
    l_fwd2 = lasagne.layers.DropoutLayer(l_fwd2, p=0.5)


    l_concat = lasagne.layers.ConcatLayer([l_fwd2,l_fwd1])
    concat_mask = lasagne.layers.ConcatLayer([l_mask2, l_mask1])
    print 'output:', get_output_shape(l_concat)
    # add droput
    #l_concat = lasagne.layers.DropoutLayer(l_concat, p=0.5)

    gate_params3 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.)
    )
    cell_params3 = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        W_cell=None, b=lasagne.init.Constant(0.),
        nonlinearity=lasagne.nonlinearities.tanh
    )

    l_lstm2 = lasagne.layers.LSTMLayer(
        l_concat,
        num_units=num_hidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        ingate=gate_params3,
        mask_input=concat_mask,
        forgetgate=gate_params3,
        cell=cell_params3,
        outgate=gate_params3,
        learn_init=True,
        only_return_final=True
    )


    # add dropout
    l_lstm2 = lasagne.layers.DropoutLayer(l_lstm2, p=0.6)
    l_pool = l_lstm2
    #Mean Pooling Layer

    # Check pool_size ? 
    # pool_size = 1
    # l_pool = lasagne.layers.FeaturePoolLayer(l_concat, pool_size)
    # with GlobalPoolLayer you don't need to check input shape
    #l_pool  = lasagne.layers.GlobalPoolLayer(l_lstm2)

    # print('Mean Pool Layer Shape:')
    # print 'input:', get_output_shape(l_lstm2, HYPOTHETICALLY)
    # print 'output:', get_output_shape(l_pool, HYPOTHETICALLY)
    # print

    # Dense Layer
    network = lasagne.layers.DenseLayer(
        l_pool,
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    # print('Dense Layer Shape:')
    # print 'input:', get_output_shape(l_concat, HYPOTHETICALLY)
    # print 'output:', get_output_shape(network, HYPOTHETICALLY)

    return network
    
def load_data(target, path, hidden_units, both, top, batch_size):
 


    train_file = path +  '/pkl/2_cnn/w2v_300/ucsc.contextsep.TRAIN.' + target + '.pkl'
    test_file = path + '/pkl/2_cnn/w2v_300/ucsc.contextsep.TEST.' + target + '.pkl'
    

    
    batch_size = 25
    x1_filter_hs = [1,2,3]
    x2_filter_hs = [1,2,3]
    hidden_units = [100,2]
    non_static = True
    
    print "loading data...",
    logger.error("loading data...");
    

    x = cPickle.load(open(train_file,"rb"))
    train_data, W, word_idx_map, max_x1, max_x2 = x[0], x[1], x[2], x[3], x[4]
    print 'size=', len(W), len(W[0])
    print "train_data loaded!"
    
    print "max x1 length = " + str(max_x1)
    print "max x2 length = " + str(max_x2)
    # print(train_data[0]['x1'])
    # print("***********************")
    # print(train_data[0]['x2'])
    X1_train_indx, X2_train_indx, y_train = text_to_indx2(train_data, word_idx_map)
    # print(X1_train_indx[0])
    # print(len(X1_train_indx[0]))
    # print(X2_train_indx[0])
    # print(len(X2_train_indx[0]))
    # print(word_idx_map['that'])
    X1_train, X1_train_mask, X2_train, X2_train_mask = pad_mask2(X1_train_indx, X2_train_indx, max_x1, max_x2)
    # print(X1_train[0])
    # print(X1_train_mask[0])
    # print(sum(X1_train_mask[0]))
    # print(X2_train[0])
    # print(X2_train_mask[0])
    # print(sum(X2_train_mask[0]))

    train_data = np.asarray(train_data)
    # print(train_data.shape)
    n_batches = int(math.ceil(train_data.shape[0]/float(batch_size)))
    n_train_batches = int(np.round(n_batches*0.9))
    # print(n_batches)
    # print(n_train_batches)
    # print 'n_batches: ', n_batches
    # print 'n_train_batches: ', n_train_batches
    train_set_x1 = X1_train[:n_train_batches*batch_size,:]
    train_set_mask1 = X1_train_mask[:n_train_batches*batch_size,:]
    train_set_x2 = X2_train[:n_train_batches*batch_size,:]
    train_set_mask2 = X2_train_mask[:n_train_batches*batch_size,:]
    train_set_y = y_train[:n_train_batches*batch_size]


    val_set_x1 = X1_train[n_train_batches*batch_size:,:]
    val_set_mask1 = X1_train_mask[n_train_batches*batch_size:,:]
    val_set_x2 = X2_train[n_train_batches*batch_size:,:]
    val_set_mask2 = X2_train_mask[n_train_batches*batch_size:,:]
    val_set_y = y_train[n_train_batches*batch_size:]


    if val_set_x1.shape[0] % batch_size > 0:
        extra_data_num = batch_size - val_set_x1.shape[0] % batch_size
        new_set1 = np.append(val_set_x1, val_set_x1[:extra_data_num], axis=0)
        new_set_mask1 = np.append(val_set_mask1, val_set_mask1[:extra_data_num], axis = 0)
        new_set2 = np.append(val_set_x2, val_set_x2[:extra_data_num], axis=0)
        new_set_mask2 = np.append(val_set_mask2, val_set_mask2[:extra_data_num], axis = 0)
        new_set_y = np.append(val_set_y, val_set_y[:extra_data_num], axis = 0)
        # might be possible that we still do not have the proper batch size - 
        # in that case - for remaining - add from "training" data
        val_set_x1 = new_set1
        val_set_mask1 = new_set_mask1
        val_set_x2 = new_set2
        val_set_mask2 = new_set_mask2
        val_set_y = new_set_y
        if val_set_x1.shape[0] % batch_size > 0:
             extra_data_num = batch_size - val_set_x1.shape[0] % batch_size
             new_set1 = np.append(val_set_x1, train_set_x1[:extra_data_num], axis=0)
             new_set_mask1 = np.append(val_set_mask1, train_set_mask1[:extra_data_num], axis = 0)
             new_set2 = np.append(val_set_x2, train_set_x2[:extra_data_num], axis=0)
             new_set_mask2 = np.append(val_set_mask2, train_set_mask2[:extra_data_num], axis = 0)
             new_set_y = np.append(val_set_y, train_set_y[:extra_data_num], axis = 0)

             val_set_x1 = new_set1
             val_set_mask1 = new_set_mask1
             val_set_x2 = new_set2
             val_set_mask2 = new_set_mask2
             val_set_y = new_set_y

    print 'train size =', train_set_x1.shape, ' val size =', val_set_x1.shape 

    # get the test data

    test_data = cPickle.load(open(test_file,'rb'))
    X1_test_indx,X2_test_indx, y_test = text_to_indx2(test_data, word_idx_map)
    test_set_x1,test_set_mask1,  test_set_x2, test_set_mask2 = pad_mask2(X1_test_indx, X2_test_indx, max_x1, max_x2)


    # put into shared variables  -- only useful if using GPU
    #train_set_x, train_set_mask, train_set_y = shared_dataset_mask(train_set_x, train_set_mask, train_set_y)
    #val_set_x, val_set_mask, val_set_y  = shared_dataset_mask(val_set_x, val_set_mask, val_set_y)
    #test_set_x, test_set_mask, test_set_y = shared_dataset_mask(X_test, X_test_mask, y_test)
    #test_set_x, test_set_mask, test_set_y = X_test, X_test_mask, y_test


    print "data loaded!"
    
    print "max length  1 = " + str(max_x1)
    print "max length  1 = " + str(max_x2)

    return train_set_x1, train_set_mask1, train_set_x2, train_set_mask2, train_set_y, val_set_x1, val_set_mask1, val_set_x2, val_set_mask2,\
        val_set_y, test_set_x1, test_set_mask1, test_set_x2, test_set_mask2, y_test, word_idx_map, W, max_x1,max_x2


def train_test_model(target, path, hidden_units, both, top, batch_size,num_epochs = 25):
    train_set_x1, train_set_mask1, train_set_x2, train_set_mask2, train_set_y, val_set_x1, val_set_mask1, val_set_x2, \
    val_set_mask2,val_set_y, test_set_x1, test_set_mask1, test_set_x2, test_set_mask2, y_test, word_idx_map, W, max_x1,max_x2 = load_data(target, path, hidden_units, both,top, batch_size = batch_size)


    V = len(word_idx_map)
    print "Vocab size:", V

    # Initialize theano variables for input and output
    index = T.lscalar() 
    X1 = T.imatrix('X1')
    X2 = T.imatrix('X2')
    M1 = T.matrix('M1')
    M2 = T.matrix('M2')
    y = T.ivector('y')
    
    network = build_model(W = W, input_var1=X1, input_var2 = X2, mask_var1=M1, mask_var2 = M2, batch_size = batch_size, max_seq_len1 = max_x1,max_seq_len2 = max_x2)
    output = lasagne.layers.get_output(network)

    # Define objective function (cost) to minimize, mean crossentropy error
    cost = lasagne.objectives.categorical_crossentropy(output, y).mean()

    # Compute gradient updates 
    params = lasagne.layers.get_all_params(network)
    # grad_updates = lasagne.updates.nesterov_momentum(cost, params,learn_rate)
    grad_updates = lasagne.updates.adam(cost, params)
    # grad_updates = lasagne.updates.adadelta(cost, params, learn_rate)
    #print(y_train)
    # Compile train objective
    print "Compiling training functions"
    train = theano.function([X1, X2, M1, M2, y], cost,
                            updates=grad_updates,
                            allow_input_downcast=True)

    log_path = "logs/"
    test_output = lasagne.layers.get_output(network, deterministic=True)
    val_cost_fn = lasagne.objectives.categorical_crossentropy(
        test_output, y).mean()
    preds = T.argmax(test_output, axis=1)

    val_acc_fn = T.mean(T.eq(preds, y),
                        dtype=theano.config.floatX)
    val_fn = theano.function([X1, X2, M1, M2, y], [val_cost_fn, val_acc_fn, preds],
                             allow_input_downcast=True)
    log_file = open(log_path + "training_log_" +
                    time.strftime('%m%d%Y_%H%M%S'), "w+")


    def compute_val_error(log_file=log_file, val_set_x1=val_set_x1,val_set_x2=val_set_x2, val_set_mask1= val_set_mask1,val_set_mask2=val_set_mask2, val_set_y=val_set_y):
        val_loss = 0.
        val_acc = 0.
        val_batches = 0
        for batch in iterate_minibatches2(val_set_x1,val_set_x2,val_set_mask1, val_set_mask2,val_set_y,
                                         batch_size, shuffle=False):
            x_val_mini1,x_val_mini2, x_val_mini_mask1,x_val_mini_mask2,y_val_mini = batch
            v_loss, v_acc, _ = val_fn(x_val_mini1,x_val_mini2,
                                      x_val_mini_mask1, x_val_mini_mask2,
                                      y_val_mini)
            val_loss += v_loss
            val_acc += v_acc
            val_batches += 1

        try:
            val_loss /= val_batches
            val_acc /= val_batches
            log_file.write("\t  validation loss:\t\t{:.6f}\n".format(val_loss))
            log_file.write("\t  validation accuracy:\t\t{:.2f} %\n".format(val_acc * 100.))
        except ZeroDivisionError:
            print('WARNING: val_batches == 0')

        return val_loss, val_acc

    print "Starting Training"
    begin_time = time.time()
    best_val_acc = -np.inf
    for epoch in xrange(num_epochs):
        train_err = 0.
        train_batches = 0
        start_time = time.time()
        # if epoch > 5:
        #     learn_rate /= 2
        for batch in iterate_minibatches2(train_set_x1,train_set_x2,train_set_mask1,train_set_mask2,train_set_y,
                                         batch_size, shuffle=True):
            x_mini1,x_mini2, x_mini_mask1,x_mini_mask2,y_mini = batch
            # print x_train.shape, y_train.shape
            train_err += train(x_mini1,x_mini2,
                                      x_mini_mask1, x_mini_mask2,
                                      y_mini)

            train_batches += 1
            # print "Batch {} : cost {:.6f}".format(
            #     train_batches, train_err / train_batches)

            if train_batches % 512 == 0:
                log_file.write("\tBatch {} of epoch {} took {:.3f}s\n".format(
                    train_batches, epoch+1, time.time() - start_time))
                log_file.write("\t  training loss:\t\t{:.6f}\n".format(train_err / train_batches))

                val_loss, val_acc = compute_val_error(val_set_x1=val_set_x1,val_set_x2=val_set_x2, val_set_mask1= val_set_mask1,val_set_mask2=val_set_mask2, val_set_y=val_set_y)

                if val_acc >= best_val_acc:
                    best_val_acc = val_acc
                    write_model_data(network, log_path + '/best_lstm_model_' + target)

                log_file.write(
                    "\tCurrent best validation accuracy:\t\t{:.2f}\n".format(
                        best_val_acc * 100.))
                log_file.flush()

        disp_msg = "Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time)
        print disp_msg
        log_file.write(disp_msg)
        log_file.write("\t  training loss:\t\t{:.6f}\n".format(
            train_err / train_batches))
        val_loss, val_acc = compute_val_error(val_set_x1=val_set_x1,val_set_x2=val_set_x2, val_set_mask1= val_set_mask1,val_set_mask2=val_set_mask2, val_set_y=val_set_y)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            write_model_data(network, log_path + '/best_lstm_model_' + target)

        log_file.write("Current best validation accuracy:\t\t{:.2f}\n".format(
            best_val_acc * 100.))

        if (epoch) % 1 == 0:
            test_loss, test_acc, _ = val_fn(test_set_x1,test_set_x2, test_set_mask1, test_set_mask2, y_test)
            log_file.write("Test accuracy:\t\t{:.2f}\n".format(test_acc * 100.))

        log_file.flush()

    log_file.write("Training took {:.3f}s\n".format(time.time() - begin_time))

    network = read_model_data(network, log_path + '/best_lstm_model_' + target)
    test_loss, test_acc, preds =val_fn(test_set_x1,test_set_x2, test_set_mask1, test_set_mask2, y_test)
    #print(preds)
    log_file.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(test_acc * 100.))

    log_file.close()

    return network

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

def main(args):
    
    path  = args[0]
    hidden_units = int(args[1])
    both = str_to_bool(args[2])
    top = str_to_bool(args[3])
    batch_size = int(args[4])
    targets = ['one', 'two','three', 'four', 'five']
    #targets = ['one']
    for target in targets:
        print 'working on folder: ' + target
        train_test_model(target, path, hidden_units, both, top, batch_size)

if __name__=="__main__":
    main(sys.argv[1:])