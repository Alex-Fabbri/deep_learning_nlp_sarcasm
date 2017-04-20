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
    preds, _ = classifier.predict(testing, y=test_y)

    with open("preds_output.txt", "w") as output:
        for i in range(len(preds)):
            output.write("{} {} {}\n".format(i, test_y[i], preds[i]))
