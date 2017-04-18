# structure based off of https://github.com/chridey/cmv/blob/master/cmv/bin/cmv_predict_rnn.py
import sys
import pickle
from release.lstm.SarcasmClassifier import SarcasmClassifier
from release.preprocessing.load_data import load_data
from release.preprocessing.process_properties import PreProcessor
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support as score


if __name__ == "__main__":

    targets = ["one", "two", "three", "four", "five"]
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_file = open("output/logs/log_file_{}".format(time_stamp), "w+")
    recurrent_dimension = 100
    patience = 10
    dropout = 0.5
    lambda_w = .000001
    filename = sys.argv[1]
    processor = PreProcessor(sys.argv[2])

    processor.set_target("")
    log_file.write("working on train_val_test at {} with lambda_w: {} dropout: {} recurrent_dimension: {} patience: {}\n".format(time_stamp,lambda_w, dropout, recurrent_dimension, patience))
    training, y, testing, test_y, kwargs, test_data = load_data(processor)
    kwargs.update({'lambda_w' : lambda_w, 'dropout': dropout, "num_hidden": recurrent_dimension, "patience": patience})
    classifier = SarcasmClassifier(**kwargs)
    attention_classifier = classifier.classifier
    with np.load(filename) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    attention_classifier.set_params(param_values)
    test_sentence_attention = attention_classifier.sentence_attention_response(*testing)
    test_sentence_context,preds = attention_classifier.sentence_attention_context(*testing)
    precision, recall, fscore, support = score(test_y, preds)

    log_file.write("precision : {}".format(precision))
    log_file.write("recall : {}".format(recall))
    log_file.write("fscore : {}".format(fscore))
    types = pickle.load(open("data_final/type2/types.pkl", "rb")) 
    context_texts = []
    response_texts = []
    context_lens = []
    response_lens = []
    for query in test_data:
        context = query["x1"]
        response = query["x2"]
        try:
            context = context.decode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            context = context.strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        try:
            response = response.decode('utf-8').strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
        except:
            response = response.strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')
    
        

        context_sents = nltk.sent_tokenize(context)
        response_sents = nltk.sent_tokenize(response)

        context_lens.append(len(context_sents))
        response_lens.append(len(response_sents))
        context_texts.append(context_sents)
        response_texts.append(response_sents)

    with open("attention_output.txt", "w") as output:
        for i in range(test_sentence_context.shape[0]):
            output.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(i, test_y[i], preds[i],types[i], context_lens[i], context_texts[i], str(test_sentence_context[i]).strip().replace('\n',' ').replace('\r',' ').replace('\t',' '), response_lens[i], response_texts[i] , str(test_sentence_attention[i]).strip().replace('\n',' ').replace('\r',' ').replace('\t',' ')))
            #print("example: {} \n".format(i))
            #print("gold label: {} \n".format(test_y[i]))
            #print("predicted label: {} \n".format(preds[i]))
            #print(test_sentence_context[i])
            #print(test_sentence_attention[i])
            #print("finished example: {} \n".format(i))


    log_file.close()        
        
