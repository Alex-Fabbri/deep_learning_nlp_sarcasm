from release.preprocessing.process_properties import PreProcessor
from release.preprocessing.utils import * 
import sys


if __name__=="__main__":
    processor = PreProcessor(sys.argv[1])
    test = processor.__dict__ 
    print(test)
    print(type(test))
            



