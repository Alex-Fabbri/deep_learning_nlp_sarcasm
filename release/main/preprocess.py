from release.preprocessing.process_properties import PreProcessor
from release.preprocessing.utils import * 
import sys


if __name__=="__main__":
    processor = PreProcessor(sys.argv[1])
    vocab = create_vocab(processor.vocab_file)
    w2v = loadVectors(vocab,processor.w2v_file,processor.w2v_type)

    folds = ['one', 'two', 'three', 'four', 'five']
    if processor.test_type == "5fold":
        print("5 fold cross validation\n")
        for fold in folds:
            if(processor.separate == "True"):
                train_test_separate(fold, vocab, w2v, processor.input, processor.output, processor.topSim, processor.lastSent, processor.data_type)
            else:
                print("train_test")
                #print(processor.lastSent)
                # is the same whether or not using attention
                train_test(fold, vocab, w2v, processor.input, processor.output, processor.both, processor.topSim, processor.lastSent,processor.data_type)
    else:
        print("train_val_test\n")


            



