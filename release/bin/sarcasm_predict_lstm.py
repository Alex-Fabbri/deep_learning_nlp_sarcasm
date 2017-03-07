# structure based off of https://github.com/chridey/cmv/blob/master/cmv/bin/cmv_predict_rnn.py
import sys
from release.lstm.SarcasmClassifier import SarcasmClassifier
from release.preprocessing.load_data import load_data
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == "__main__":

    targets = ["one", "two", "three", "four", "five"]
    for target in targets:
        print("working on target: {}\n".format(target))
        training, y, testing, test_y, kwargs = load_data(target, sys.argv[1])
        classifier = SarcasmClassifier(**kwargs)
        classifier.fit(training, y)
        classifier.save('{}_classifier_{}'.format(kwargs["output"], target))
        preds,scores = classifier.test(testing, test_y)
        np.save('{}_classifier_{}_preds'.format(kwargs["output"],target), preds)
