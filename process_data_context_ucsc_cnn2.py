import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import math

import nltk

from collections import Counter


windowSize = 1

def range(index1,index2):
    
    if math.fabs(index1-index2) <=windowSize:
        return True
    else:
        return False


"""
The input file format is: 1st column is label (int), 2rd column is keyword, and 4nd column is query (string)
"""

def getLastSentence(context):
    
    context = context.decode('utf-8').strip()
    sentences = nltk.sent_tokenize(context)
    return sentences[len(sentences)-1]

def read_data_file_cross(data_file, max_x1, max_x2, is_train):
    queries = []
    all_ids = []
    change_count = 0
    with open(data_file, "r") as fin:
        for line in fin:
            line = line.strip()
            [label, text] = line.split('\t')
            text = text.lower()
            current = text.split('|||')[0]
            previous = text.split('|||')[1]
            # delete the keywords from the text
            #text = re.sub(r'\s*\b%s\b\s*' % kw, ' ', text, 1).strip()

            #x2_words = text.split()
            #x1_words = kw.split()
            
            # we create an one word window for the hashtag
            
            x1_words = []
            for index1, word1 in enumerate(previous.split()):
                for index2, word2 in enumerate(current.split()):
                    word = word1 + '|||' + word2
                    x1_words.append(word)
             
            
            
            # update vocab only when it is training data
            '''
            if is_train == 1:
                for word in set(x1_words):
                    vocab[word] += 1
                for word in set(x2_words):
                    vocab[word] += 1
            '''
                    
            if len(x1_words) > max_x1:
                x1_words = x1_words[:max_x1]
                change_count += 1
           
            datum = {"y": int(label),
                    "text": " ".join(x1_words),
                    "num_words": len(x1_words)}
                    
            queries.append(datum)
            all_ids.append(label)
     
#    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    return queries,all_ids

"""
The input file format is: 1st column is label (int), 2rd column is keyword, and 4nd column is query (string)
"""
def read_data_file(data_file, max_x1, max_x2, is_train):
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


def read_all_data(allData, output_id_file,target, max_x1, max_x2, is_train):
    queries = []
    change_count = 0
    writer = open(output_id_file,'w')
    for line in allData:
        line = line.strip()
        [label, kw, _, text] = line.split('\t');
        text = text.lower()
                    
        # we create an one word window for the hashtag
        kwPosn = -1
        for index, word in enumerate(text.split()):
            if word.lower() == kw.lower() or word.lower() == '#' + kw.lower():
                kwPosn = index
                break
        if kwPosn == -1:
            print 'the target: ' + target + ' is absent - returning'
            continue
            
        x1_words = []
        x2_words = []
        for index, word in enumerate(text.split()):
            if range(index,kwPosn):
                x1_words.append(word)
            else:
                x2_words.append(word)
             
            
            
            # update vocab only when it is training data
            '''
            if is_train == 1:
                for word in set(x1_words):
                    vocab[word] += 1
                for word in set(x2_words):
                    vocab[word] += 1
            '''
                    
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
        writer.write((label))
        writer.write('\n')
    
    print ("length more than %i: %i" % (max_x2, change_count)) 
    
    writer.close()
    return queries
    

"""
input word_vecs is a dict from string to vector.
this function will generate two dicts: from string to index, from index to vector.
Also, a random vector is initialized for all unknown words with index 0.
""" 
def get_W(word_vecs):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    k = len(word_vecs.values()[0])
    print 'k =', k
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+2, k), dtype='float32')
    #W[0] = np.zeros(k)
    np.random.seed(4321) # first word is UNK
    W[0] = np.random.normal(0,0.17,k)
    W[1] = np.random.normal(0,0.17,k)
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



def train_senti(target):
    
    path = './data/input/sarcasm_senti_training/'
    
    train_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN'
    output_train_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN'   + '.pkl'
    output_train_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TRAIN' + '.id'
    
    max_x1 = 1
    max_x2 = 100
    x1_filter_h = [3]
    x2_filter_h = [1,2,3]
    
    np.random.seed(4321)
    print "loading training data...",
    train_data,train_id = read_data_file(train_file,target, max_x1, max_x2, 1)
    
    max_x1 = np.max(pd.DataFrame(train_data)["x1_len"])
    max_x2 = np.max(pd.DataFrame(train_data)["x2_len"])
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max_x1)
    print "max sentence length: " + str(max_x2)
    
    #w2v_file = "/Users/wguo/projects/nn/data/w2v/GoogleNews-vectors-negative300.bin"
    w2v_file = "../data/w2v/sdata-vectors.bin"
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"

    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    
    cPickle.dump([train_data, W, word_idx_map, max_x1, max_x2], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
   
    
def create_all_vocab(targets):
    cutoff = 5
    vocab = defaultdict(float)
    path = '/Users/dg513/work/eclipse-workspace/sarcasm-workspace/SarcasmDetection/data/twitter_corpus/wsd/sentiment/samelm2/weiwei/'

    allLines = []
    for target in targets:
        train_file = path + 'tweet.' + target + '.target.TRAIN'
        lines = open(train_file).readlines()
        allLines.extend(lines)
    
    raw = [process_line(l) for l in allLines ]
    cntx = Counter( [ w for e in raw for w in e ] )
    lst = [ x for x, y in cntx.iteritems() if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
    return vocab # (this is a dictionary of [word] = [position] which is fine since we are only bothered about the key of the dict.
   

    
def create_vocab():
    
    cutoff = 0
    vocab = defaultdict(float)
    path = './data/config/'
    train_file = path + 'only_context.unigrams.txt'
    
    lines = open(train_file).readlines()
    raw = [process_line(l) for l in lines ]
    lst = [ x for x, y in raw if y > cutoff ] + ["## UNK ##"]
    vocab = dict([ (y,x) for x,y in enumerate(lst) ])
    
        
    return vocab # (this is a dictionary of [word] = [position] which is fine since we are only bothered about the key of the dict.

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

def train_test_allData(targets,vocab):
    
    path = '/Users/dg513/work/eclipse-workspace/sarcasm-workspace/SarcasmDetection/data/twitter_corpus/wsd/sentiment/samelm2/weiwei/'
    
    allTraining = []
    allTesting = []
    
    output_train_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TRAIN' + '.pkl'
    output_test_file = './data/output/samelm/pkl/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TEST'   + '.pkl'
    
    output_train_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TRAIN' + '.id'
    output_test_id_file = './data/output/samelm/ids/2_cnn/' + 'tweet.' + 'ALLTARGETS' + '.target.TEST' + '.id'

    
    max_x1 = 3
    max_x2 = 100
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"
    print "loading word2vec vectors..."

    all_train_data = []
    all_test_data = []
    
    all_train_id = []
    all_test_id = []
    
    print "loading training data...",
  
    for target in targets:
        train_file = path + 'tweet.' + target + '.target.TRAIN'
        test_file = path + 'tweet.' + target + '.target.TEST'
        
        train_data,train_id = read_data_file(train_file,target, max_x1,max_x2, 1)
        all_train_data.extend(train_data)
        all_train_id.extend(train_id)
        
        test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
        all_test_data.extend(test_data)
        all_test_id.extend(test_id)
        
    
    
    np.random.seed(4321)
 #   target = 'ALLTARGETS'
    cPickle.dump(all_test_data, open(output_test_file, "wb"))
    
   
    max_x1_train = np.max(pd.DataFrame(all_train_data)["x1_len"])
    max_x2_train = np.max(pd.DataFrame(all_train_data)["x2_len"])
    
    max_x1_test = np.max(pd.DataFrame(all_test_data)["x1_len"])
    max_x2_test = np.max(pd.DataFrame(all_test_data)["x2_len"])
    
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "max keyword length: " + str(max(max_x1_train,max_x1_test))
    print "max sentence length: " + str(max(max_x2_train,max_x2_test))
    
    #w2v_file = "/Users/wguo/projects/nn/data/w2v/GoogleNews-vectors-negative300.bin"
    w2v_file = "../data/w2v/sdata-vectors.bin"
    w2v_file = "./data/config/tweet.all.05032015.sg.model.bin"

    print "loading word2vec vectors...",
    
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    #add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    
    cPickle.dump([all_train_data, W, word_idx_map,max(max_x1_train,max_x1_test), max(max_x2_train,max_x2_test)], open(output_train_file, "wb"))
    
    writer1 = open(output_train_id_file,'w')
    for id in all_train_id:
        writer1.write(id)
        writer1.write('\n')
        
    writer1.close()
    
    writer2 = open(output_test_id_file,'w')
    for id in all_test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    
    print "dataset created!"

def train_test(fold,vocab,w2v,input,output,w2v_type='bin'):

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
    train_data,train_id = read_data_file(train_file, max_x1, max_x2, 1)
    
    test_data,test_id = read_data_file(test_file, max_x1,max_x2, 0)
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


def test_senti(target):
 
    path = './data/input/sarcasm_senti_testing/'
    test_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TEST'
    output_test_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST'   + '.pkl'
    output_test_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST' + '.id'

    max_x1 = 3
    max_x2 = 100

    test_data,test_id = read_data_file(test_file,target, max_x1,max_x2, 0)
    cPickle.dump(test_data, open(output_test_file, "wb"))

    writer2 = open(output_test_id_file,'w')
    for id in test_id:
        writer2.write(id)
        writer2.write('\n')
        
    writer2.close()
    
    
def test(target):
    
    path = './data/input/sarcasm_senti_training/'
    
    test_file = path + 'tweet.' + target + '.target.SENTI.BOTH.TEST'
    output_test_file = './data/output/samelm/pkl/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST'   + '.pkl'
    output_test_id_file = './data/output/samelm/ids/senti/2_cnn/' + 'tweet.' + target + '.target.SENTI.BOTH.TEST' + '.id'

    
    max_l = 50
    vocab = defaultdict(float) # empty 
    test_data = read_data_file(test_file, vocab, max_l, 0)
    cPickle.dump(test_data, open(output_file, "wb"))
    
    
def loadVectors(vocab,w2v_file,w2v_type = 'bin'):
    
    print 'loading word2vec vectors...'
    if w2v_type == 'bin':
        w2v = load_bin_vec(w2v_file, vocab)
    if w2v_type == 'txt' :
        w2v = load_txt_vec(w2v_file, vocab)

    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))    
    return w2v

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


def main(args):    
    
    input = args[0]#'./data/ucsc/input/'
    output = args[1]#'./data/ucsc/output/'
    w2v_file = args[2]#'/export/projects/argumentation/data_large_ngrams/googlengram/GoogleNews-vectors-negative300.bin'
    vocab = create_vocab(input)
    w2v_type = 'bin'
    w2v = loadVectors(vocab,w2v_file,w2v_type='bin')

    
    folds = ['one', 'two', 'three', 'four', 'five']
    for fold in folds:
        train_test(fold,vocab,w2v,input,output,w2v_type='bin')
      
if __name__=="__main__":
    main(sys.argv[1:])
