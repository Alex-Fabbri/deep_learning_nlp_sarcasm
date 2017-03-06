import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
from collections import Counter
import nltk


def loadVectors(vocab,w2v_file,w2v_type='bin'):
    
    print 'loading word2vec vectors...'
    if w2v_type == 'bin':
        w2v = load_bin_vec(w2v_file, vocab)
    if w2v_type == 'txt' :
        w2v = load_txt_vec(w2v_file, vocab)

    
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    
    return w2v

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


def str_to_bool(s):
    if(type(s) is not bool):
        if s == 'True':
             return True
        elif s == 'False':
             return False
        else:
             print("{} has this type: {}".format(s, type(s)))
             raise ValueError
    else:
        return s

"""
for input file, assume 1st column is label (int), and 2nd column is query (string)
"""
def read_data_file(data_file, max_l, is_train,both=False,topSim=False):
    queries = []
    ids = []
    change_count = 0
    
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            line = line.lower()
            [label,text,context] = line.split('\t')
            # initially - just do with the original msg only 
            # later do both 
            if both == True and topSim == False:
                text = getLastSentence(context) + ' ' + text 
            if both == True and topSim == True:
                text = context + ' ' + text 
            
            text = text.lower()
            words = nltk.word_tokenize(text)
            # we dont keep the target word in the text - just to make it similar to the EMNLP2015 approach
            removes = []
            newWords = []
            for index, word in enumerate(words):
                newWords.append(word)
                
            if len(newWords) > max_l:
                words = words[:max_l]
                change_count += 1
            datum = {"y": int(label),
                    "text": " ".join(words),
                    "num_words": len(words)}
            queries.append(datum)
            ids.append(label)
    
    print ("length more than %i: %i" % (max_l, change_count)) 
    
    
    return queries,ids

"""
The input file format is: 1st column is label (int), 2rd column is keyword, and 4nd column is query (string)
"""
def read_data_file_separate(data_file, max_x1, max_x2, is_train):
    queries = []
    all_ids = []
    change_count = 0
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            [label, text,context] = line.split('\t')
            current = text.lower()
            previous = getLastSentence(context).lower()

            x1_words = []
            x2_words = []
            for index1, word1 in enumerate(nltk.word_tokenize(current)):
                x1_words.append(word1)
            for index2, word2 in enumerate(nltk.word_tokenize(previous)):
                x2_words.append(word2)
             
            if len(x1_words) > max_x1:
                x1_words = x1_words[:max_x1]
                change_count += 1
            if len(x2_words) > max_x2:
                x2_words = x2_words[:max_x2]
                change_count += 1
            
            datum = {"y": int(label),
                    "x1": " ".join(x1_words),
                    "x2": " ".join(x2_words),
                    "x1_len": len(x1_words),
                    "x2_len": len(x2_words)}
            queries.append(datum)
            all_ids.append(label)
     
    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    return queries,all_ids

def getLastSentence(context):
   
    context = context.decode('utf-8').strip()
 
    sentences = nltk.sent_tokenize(context)
    return sentences[len(sentences)-1]

def convert(label):
    if label.strip() == 'sarc':
        return "1.0"
    elif label.strip() == 'notsarc':
        return "0.0"

def train_test(target,vocab,w2v,input,output,both=False,topSim=True):
    path = input + '/5folds/' + target + '/'
    if topSim == False:
    	train_file = path + 'ucsc.txt.train' + target
    	test_file = path + 'ucsc.txt.test' + target
    elif topSim == True:
    	train_file = path + 'ucsc.txt.train' + target +  '.topcontext' 
    	test_file = path + 'ucsc.txt.test' + target + '.topcontext'
    max_l = 200

    if both == False:
        output_train_file = output + '/pkl/1_cnn/w2v_300/' + 'ucsc.nocontext'  +'.TRAIN.' + target + '.pkl'
        output_test_file = output + 'pkl/1_cnn/w2v_300/' + 'ucsc.nocontext'  + '.TEST.' + target + '.pkl'
        output_train_id_file = output  + '/ids/1_cnn/w2v_300/' + 'ucsc.nocontext'  + '.TRAIN.' + target + '.id'
        output_test_id_file = output + '/ids/1_cnn/w2v_300/' + 'ucsc.nocontext'  + '.TEST.' + target + '.id'

    if both == True and topSim == False:
        max_l = 400

        output_train_file = output + '/pkl/1_cnn/w2v_300/' + 'ucsc.contextcat'  +'.TRAIN.' + target + '.pkl'
        output_test_file = output + '/pkl/1_cnn/w2v_300/' + 'ucsc.contextcat'  + '.TEST.' + target + '.pkl'
        output_train_id_file = output + '/ids/1_cnn/w2v_300/' + 'ucsc.contextcat'  + '.TRAIN.' + target + '.id'
        output_test_id_file = output + '/ids/1_cnn/w2v_300/' + 'ucsc.contextcat'  + '.TEST.' + target + '.id'

    if both == True and topSim == True:
        max_l = 400

        output_train_file = output + '/pkl/1_cnn/w2v_300/' + 'ucsc.contexttop'  +'.TRAIN.' + target + '.pkl'
        output_test_file = output + '/pkl/1_cnn/w2v_300/' + 'ucsc.contexttop'  + '.TEST.' + target + '.pkl'
        output_train_id_file = output + '/ids/1_cnn/w2v_300/' + 'ucsc.contexttop'  + '.TRAIN.' + target + '.id'
        output_test_id_file = output + '/ids/1_cnn/w2v_300/' + 'ucsc.contexttop'  + '.TEST.' + target + '.id'
    
    np.random.seed(4321)
    
    print "loading training data for fold: " + target
    
    train_data,train_id = read_data_file(train_file,max_l, 1,both,topSim)
    test_data,test_id = read_data_file(test_file,max_l, 0,both,topSim)
    cPickle.dump(test_data, open(output_test_file, "wb"))

    
    max_l = np.max(pd.DataFrame(train_data)["num_words"])
    print max_l
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    
    W, word_idx_map = get_W(w2v)
    np.random.shuffle(train_data)
    cPickle.dump([train_data, W, word_idx_map, max_l], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    print "dataset created!"

def train_test_separate(fold,vocab,w2v,input,output):

    train_file = input + '/5folds/' + fold + '/' +  'ucsc.txt.train' + fold 
    test_file = input + '/5folds/' + fold + '/' +  'ucsc.txt.test' + fold 

    output_train_file = output +  '/pkl/2_cnn/w2v_300/' + 'ucsc.contextsep'  +'.TRAIN.' + fold + '.pkl'
    output_test_file = output +  '/pkl/2_cnn/w2v_300/' + 'ucsc.contextsep'  + '.TEST.' + fold + '.pkl'






    output_train_id_file = output +  '/ids/2_cnn/w2v_300/' + 'ucsc.contextsep'  + '.TRAIN.' + fold + '.id'
    output_test_id_file = output +  '/ids/2_cnn/w2v_300/' + 'ucsc.contextsep'  + '.TEST.' + fold + '.id'

    
    max_x1 = 200
    max_x2 = 200
#    x1_filter_h = [1,2,3]
#    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)

    
    print "loading training data...",
    train_data,train_id = read_data_file_separate(train_file, max_x1, max_x2, 1)
    
    test_data,test_id = read_data_file_separate(test_file, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_file, "wb"))
    
    writer_perf = open(output_test_file + '.txt' , 'w')
    for data in test_data:
        writer_perf.write(str(data))
        writer_perf.write('\n')
    
    writer_perf.close()
    
 
    
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)

 #   print "max sentence length: " + str(max_x2)
    W, word_idx_map = get_W(w2v)
    np.random.shuffle(train_data)

    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    
    print "dataset created!"
