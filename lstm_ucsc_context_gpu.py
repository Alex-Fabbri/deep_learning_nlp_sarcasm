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
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return T.cast(shared_x, 'int32'), T.cast(shared_y, 'int32')

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

        return T.cast(shared_x, 'int32'), T.cast(shared_y, 'int32'),T.cast(shared_z, 'int32')


def text_to_indx(train_data, word_idx_map):
    X = []
    y = []
    for query in train_data:
        text = query["text"].split()
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

def build_model(W = None, # word embeddings
                K=300,  # dimensionality of embeddings
                num_hidden=256,  # number of hidden_units
                batch_size=None,  # size of each batch (None for variable size)
                input_var=None,  # theano variable for input
                mask_var=None,  # theano variable for input mask
                bidirectional=False,  # whether to use bi-directional LSTM
                mean_pooling=False,
                grad_clip=100.,  # gradients above this will be clipped
                max_seq_len=200,  # maximum lenght of a sequence 
                num_classes = 2):

    V = len(W)
    # Input Layer
    l_in = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=input_var)
    l_mask = lasagne.layers.InputLayer((batch_size, max_seq_len), input_var=mask_var)

    #HYPOTHETICALLY = {l_in: (200, 140), l_mask: (200, 140)}

    # print('Input Layer Shape:')
    #LIN = get_output_shape(l_in, HYPOTHETICALLY)
    # print 'input:', HYPOTHETICALLY
    # print 'output:', LIN
    # print

    # Embedding layer
    l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=V, output_size=K, W=W)
    # print('Embedding Layer Shape:')
    # print 'input:', LIN
    # print 'output:', get_output_shape(l_emb, HYPOTHETICALLY)
    # print

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

    # print('Forward LSTM Shape:')
    # print 'input:', get_output_shape(l_emb, HYPOTHETICALLY)
    # print 'output:', get_output_shape(l_fwd, HYPOTHETICALLY)

    # add droput
    #l_fwd = lasagne.layers.DropoutLayer(l_fwd, p=0.5)

    # if bidirectional:
    #     # add a backwards LSTM layer for bi-directional
    #     l_bwd = lasagne.layers.LSTMLayer(
    #         l_emb, num_units=num_hidden, grad_clipping=grad_clip,
    #         nonlinearity=lasagne.nonlinearities.tanh, mask_input=l_mask,
    #         ingate=gate_params, forgetgate=gate_params, cell=cell_params,
    #         outgate=gate_params, learn_init=True,
    #         backwards=True
    #     )
    #     print('Backward LSTM Shape:')
    #     print 'input:', get_output_shape(l_emb, HYPOTHETICALLY)
    #     print 'output:', get_output_shape(l_bwd, HYPOTHETICALLY)
    #     print

    #     # print "backward layer:", lasagne.layers.get_output_shape(
    #     #     l_bwd, {l_in: (200, 140), l_mask: (200, 140)})

    #     # concatenate forward and backward LSTM
    #     l_concat = lasagne.layers.ConcatLayer([l_fwd, l_bwd])
    #     print('Concat Layer Shape:')
    #     print 'input:', get_output_shape(l_fwd, HYPOTHETICALLY), get_output_shape(l_bwd, HYPOTHETICALLY)
    #     print 'output:', get_output_shape(l_concat, HYPOTHETICALLY)
    #     print
    # else:
    #     l_concat = l_fwd
    #     print('Concat Layer Shape:')
    #     print 'input:', get_output_shape(l_fwd, HYPOTHETICALLY)
    #     print 'output:', get_output_shape(l_concat, HYPOTHETICALLY)
    #     print

    # add droput
    l_concat = lasagne.layers.DropoutLayer(l_fwd, p=0.5)
    #l_pool = l_concat
    l_lstm2 = lasagne.layers.LSTMLayer(
        l_concat,
        num_units=num_hidden,
        grad_clipping=grad_clip,
        nonlinearity=lasagne.nonlinearities.tanh,
        mask_input=l_mask,
        ingate=gate_params,
        forgetgate=gate_params,
        cell=cell_params,
        outgate=gate_params,
        learn_init=True,
        only_return_final=True
    )

    # print('LSTM Layer #2 Shape:')
    # print 'input:', get_output_shape(l_concat, HYPOTHETICALLY)
    # print 'output:', get_output_shape(l_lstm2, HYPOTHETICALLY)
    # print

    # add dropout
    l_lstm2 = lasagne.layers.DropoutLayer(l_lstm2, p=0.6)

    #Mean Pooling Layer

    # Check pool_size ? 
    # pool_size = 2
    # l_pool = lasagne.layers.FeaturePoolLayer(l_concat, pool_size)
    # with GlobalPoolLayer you don't need to check input shape
    #l_pool  = lasagne.layers.GlobalPoolLayer(l_lstm2)

    # print('Mean Pool Layer Shape:')
    # print 'input:', get_output_shape(l_lstm2, HYPOTHETICALLY)
    # print 'output:', get_output_shape(l_pool, HYPOTHETICALLY)
    # print

    # Dense Layer
    l_pool = l_lstm2
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


def train_test_model(target, path, hidden_units, both, top, batch_size,num_epochs = 25):
    train_set_x, train_set_x_mask, train_set_y, valid_set_x, valid_set_x_mask, valid_set_y, test_set_x, test_set_x_mask, test_set_y, word_idx_map,W, max_l = load_data(target, path, hidden_units, both,top, batch_size = batch_size)

    n_train_batches = train_set_x.eval().shape[0] // batch_size
    n_valid_batches = valid_set_x.eval().shape[0] // batch_size
    n_test_batches = test_set_x.eval().shape[0] // batch_size

    V = len(word_idx_map)
    n_classes = 2
    print "Vocab size:", V
    print "Number of classes", n_classes
    # print "Classes", set(y_train)

    index = T.lscalar() 
    X = T.imatrix('X')
    M = T.imatrix('M')
    y = T.ivector('y')
    
    network = build_model(W = W, input_var=X, mask_var=M, batch_size = batch_size, max_seq_len = max_l)
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
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=grad_updates,
        givens={
            X: train_set_x[index * batch_size: (index + 1) * batch_size],
	    M: train_set_x_mask[index * batch_size: (index +1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    validate_model = theano.function(
        inputs=[index],
        outputs=[val_cost_fn, val_acc_fn],
        givens={
            X: valid_set_x[index * batch_size:(index + 1) * batch_size],
	    M: valid_set_x_mask[index * batch_size: (index +1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        inputs=[index],
        outputs=[val_acc_fn],
        givens={
            X: test_set_x[index * batch_size:(index + 1) * batch_size],
	    M: test_set_x_mask[index * batch_size: (index +1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    print "Starting Training"
    begin_time = time.time()
    print('... training')

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    #validation_frequency = min(n_train_batches, patience // 2)
    validation_frequency = n_train_batches //4 
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False
    print("the validation frequency is {}".format(validation_frequency))

    while (epoch < num_epochs) and (not done_looping):
        epoch = epoch + 1
        start_time_epoch = timeit.default_timer()
	print("Epoch number: {}".format(epoch))
	#print("number of training batches: {}".format(n_train_batches))
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            print("the minibatch cost from training on batch {} is: {}".format(minibatch_index,minibatch_avg_cost))
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)[0] for i
                                     in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                validation_accuracies = [validate_model(i)[1] for i
                                     in range(n_valid_batches)]
                this_validation_accuracy = np.mean(validation_accuracies)
		print("this is the current validation lost {}".format(this_validation_loss))
		print("this is the current validation accuracy {}".format(this_validation_accuracy))
                
                log_file.write(
                       "epoch {}, validation accuracy:  {}".format( epoch,this_validation_accuracy *1.0 )
                    )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    write_model_data(network, log_path + '/best_lstm_model')

                    # test it on the test set
                    test_accuracies = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = np.mean(test_accuracies)
		    print("the current test score for the best validation set accuracy is: {}".format(test_score))
                    log_file.write("\t  test accuracy:\t\t{}\n".format(test_score))

            if patience <= iter:
                done_looping = True
                break
	end_time_epoch = timeit.default_timer()
	total_time = (end_time_epoch - start_time_epoch) /60.
	print("Total time for epoch: " + str(total_time))
	

    end_time = timeit.default_timer()
    log_file.write("the code ran for {} ".format(((end_time-start_time)/60)))
    return [best_validation_loss * 100., best_iter +1, test_score*100.]
#        for batch in iterate_minibatches(X_train,X_train_mask, y_train,
#                                         batch_size, shuffle=True):
#            x_mini,x_mini_mask, y_mini = batch
#            # print x_train.shape, y_train.shape
#            train_err += train(x_mini, x_mini_mask, y_mini)
#            train_batches += 1
#            # print "Batch {} : cost {:.6f}".format(
#            #     train_batches, train_err / train_batches)
#
#            if train_batches % batch_size == 0:
#                log_file.write("\tBatch {} of epoch {} took {:.3f}s\n".format(
#                    train_batches, epoch+1, time.time() - start_time))
#
#                val_loss, val_acc = compute_val_error(X_val=X_val,X_val_mask= X_val_mask, y_val=y_val)
#
#                if val_acc >= best_val_acc:
#                    best_val_acc = val_acc
#                    write_model_data(network, log_path + '/best_lstm_model')
#
#                log_file.write(
#                    "\tCurrent best validation accuracy:\t\t{:.2f}\n".format(
#                        best_val_acc * 100.))
#                log_file.flush()
#
#        disp_msg = "Epoch {} of {} took {:.3f}s\n".format(epoch + 1, num_epochs, time.time() - start_time)
#        print disp_msg
#        log_file.write(disp_msg)
#        log_file.write("\t  training loss:\t\t{:.6f}\n".format(
#            train_err / train_batches))
#        val_loss, val_acc = compute_val_error(X_val=X_val,X_val_mask = X_val_mask, y_val=y_val)
#        if val_acc >= best_val_acc:
#            best_val_acc = val_acc
#            write_model_data(network, log_path + '/best_lstm_model')
#
#        log_file.write("Current best validation accuracy:\t\t{:.2f}\n".format(
#            best_val_acc * 100.))
#
#        if (epoch) % 1 == 0:
#            test_loss, test_acc, _ = val_fn(X_test,X_test_mask, y_test)
#            log_file.write("Test accuracy:\t\t{:.2f}\n".format(test_acc * 100.))
#
#        log_file.flush()
#
#    log_file.write("Training took {:.3f}s\n".format(time.time() - begin_time))
#
#    network = read_model_data(network, log_path + '/best_lstm_model')
#    test_loss, test_acc, _ = val_fn(X_test, X_test_mask, y_test)
#    log_file.write("Best Model Test accuracy:\t\t{:.2f}%\n".format(test_acc * 100.))
#
#    log_file.close()
#
#    return network
#
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
