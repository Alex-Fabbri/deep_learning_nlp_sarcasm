import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd

from collections import Counter
import nltk

def getLastSentence(context):
   
    context = context.decode('utf-8').strip()
 
    sentences = nltk.sent_tokenize(context)
    return sentences[len(sentences)-1]

def convert(label):
    if label.strip() == 'sarc':
        return "1.0"
    elif label.strip() == 'notsarc':
        return "0.0"

def get_W(word_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')            

    np.random.seed(4321)
    W[0] = np.zeros(k) # 1st word is all zeros (for padding)
    W[1] = np.random.normal(0,0.17,k) # 2nd word is unknown word
    i = 2
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        print 'vocab size =', vocab_size, ' k =', layer1_size
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')  
            else:
                f.read(binary_len)
    return word_vecs

def load_txt_vec(fname, vocab):
    """
    Loads Xx1 word vecs from Google (Mikolov) word2vec or GloVe
    """
    word_vector = {}
    f = open(fname)
    for line in f:
        line = line.strip().lower()
        features = line.split()
        word = features[0]
        word_vector[word]  = np.array(features[1:],dtype="float32")
   
    return word_vector

def create_vocab(path):
    
    cutoff = 2
    vocab = defaultdict(float)
    train_file = path + 'sarcasm_v2_tab.txt'
    
    lines = open(train_file).readlines()
    raw_lines = [process_line(l) for l in lines ]
    cntx = Counter( [word for raw_line in raw_lines for word in raw_line] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    vocab['@touser']= 100 # just a default
    return vocab # (this is a dictionary of [word] = [position] which is 
    #fine since we are only bothered about the key of the dict.
   

def process_line(line):
   
    line = line.decode('utf-8').strip()
    [x,y,z,context, irony] = line.split('\t')
    '''
         we only need to build unigram model
     '''
    context_words = nltk.word_tokenize(context.lower())
    irony_words = nltk.word_tokenize(irony.lower())
    
    context_words.extend(irony_words) 
    return context_words


def loadVectors(vocab,w2v_file,w2v_type='bin'):
    
    print 'loading word2vec vectors...'
    if w2v_type == 'bin':
        w2v = load_bin_vec(w2v_file, vocab)
    if w2v_type == 'txt' :
        w2v = load_txt_vec(w2v_file, vocab)

    
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    return w2v

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was
