import sys
from release.lstm.SarcasmClassifier import SarcasmClassifier
from release.preprocessing.load_data import load_data
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    targets = ["one", "two", "three", "four", "five"]
    taget = ["one"]
    for target in targets:
        print("working on target: {}\n".format(target))
        training, y, validation, val_y, kwargs = load_data(target, sys.argv[1])
        classifier = SarcasmClassifier(**kwargs)
        X, X_heldout, y, y_heldout = train_test_split(training, y, test_size = .9)
        #classifier.fit(training, y)
        #classifier.save('{}.{}.{}'.format(outputfile, recurrent_dimension, target))
        #scores = classifier.decision_function(validation, val_y)
        #np.save('{}.{}.{}.scores'.format(outputfile, recurrent_dimension, target))
