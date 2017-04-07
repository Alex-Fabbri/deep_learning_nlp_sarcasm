# structure based off of https://github.com/chridey/cmv/blob/master/cmv/bin/cmv_predict_rnn.py
import sys
from release.lstm.SarcasmClassifier import SarcasmClassifier
from release.preprocessing.load_data import load_data
from release.preprocessing.process_properties import PreProcessor
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import datetime


if __name__ == "__main__":

    targets = ["one", "two", "three", "four", "five"]
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log_file = open("output/logs/log_file_{}".format(time_stamp), "w+")

    lambda_ws = [0, .0000001, .000001, .00001, .0001]
    dropouts = [0.25, 0, 0.5, 0.75]
    recurrent_dimensions = [100, 50, 200, 300]
    patiences = [2,5,10]
    processor = PreProcessor(sys.argv[1])

    targets = ["one", "two", "three", "four", "five"]
    for lambda_w in lambda_ws:
        for dropout in dropouts:
            for recurrent_dimension in recurrent_dimensions:
                for patience in patiences:
                    if processor.test_type  == "5fold":
                        for target in targets:
                            print("working on target: {}\n".format(target))
                            log_file.write("working on target: {}\n".format(target))

                            processor.set_target("target", target)
                            kwargs.update({'lambda_w' : lambda_w, 'dropout': dropout, "num_hidden": recurrent_dimension, "patience": patience})
                            training, y, testing, test_y, kwargs = load_data(processor)
                            classifier = SarcasmClassifier(**kwargs)
                            classifier.fit(training, y, log_file)
                            classifier.save('output/models/classifier_{}_{}_{}_{}_{}_{}'.format(target, time_stamp,lambda_w, dropout, recurrent_dimension, patience))
                            preds,scores = classifier.predict(testing, test_y)
                            precision, recall, fscore = scores[0], scores[1], scores[2]

                            log_file.write("precision for target {} : {}".format(target, precision))
                            log_file.write("recall for target {} : {}".format(target, recall))
                            log_file.write("fscore for target {} : {}".format(target, fscore))
                            log_file.flush()
                            np.save('output/preds/preds_{}_{}_{}_{}_{}_{}'.format(target, time_stamp, lambda_w, dropout, recurrent_dimension, patience), preds)
                            print("finished target: {}\n".format(target))

                    if processor.test_type == "train_val_test":
                        print("train_val_test!\n")
                        processor.set_target("")
                        training, y, testing, test_y, kwargs = load_data(processor)
                        kwargs.update({'lambda_w' : lambda_w, 'dropout': dropout, "num_hidden": recurrent_dimension, "patience": patience})
                        classifier = SarcasmClassifier(**kwargs)
                        classifier.fit(training, y, log_file)
                        classifier.save('output/models/classifier_{}_{}_{}_{}_{}_{}'.format(target, time_stamp,lambda_w, dropout, recurrent_dimension, patience))
                        preds,scores = classifier.predict(testing, test_y)
                        precision, recall, fscore = scores[0], scores[1], scores[2]

                        log_file.write("precision for target {} : {}".format(target, precision))
                        log_file.write("recall for target {} : {}".format(target, recall))
                        log_file.write("fscore for target {} : {}".format(target, fscore))
                        log_file.flush()
                        np.save('output/preds/preds_{}_{}_{}_{}_{}_{}'.format(target, time_stamp, lambda_w, dropout, recurrent_dimension, patience), preds)
                        print("finished target: {}\n".format(target))
    

    log_file.close()        
        
