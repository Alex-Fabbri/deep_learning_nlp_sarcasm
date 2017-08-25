from release.preprocessing.process_properties import PreProcessor
from release.preprocessing.utils import * 
import sys


if __name__=="__main__":
    np.random.seed(1234)
    processor = PreProcessor(sys.argv[1])
    vocab = create_vocab(processor.vocab_file)
    w2v = loadVectors(vocab,processor.w2v_file,processor.w2v_type)

    folds = ['one', 'two', 'three', 'four', 'five']
    if processor.test_type == "5fold":
        print("5 fold cross validation\n")
        for fold in folds:
            print("working on: {}\n".format(fold))
            processor.set_target(fold)
            if(processor.separate == "True"):
                print("separate processing on fold\n")
                train_test_separate(processor, vocab, w2v)
            else:
                print("train_test on fold\n")
                # is the same whether or not using attention
                train_test(processor, vocab, w2v)
    if processor.test_type == "train_val_test":
        processor.set_target("")
        if(processor.separate == "True"):
            print("separate processing\n")
            #only separate is used for final results!!!
            train_test_separate(processor, vocab, w2v)
        else:
            print("train_test\n")
            # is the same whether or not using attention
            train_test(processor, vocab, w2v)
        print("train_val_test\n")
