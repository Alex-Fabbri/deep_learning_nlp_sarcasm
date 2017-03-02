import sys
import ConfigParser

class PreProcessor:
    def __init__(self, config_file):
	config = ConfigParser.RawConfigParser()
	config.read(config_file)
	header = "PropertySection"
	self.input = config.get(header, "input")
	self.output = config.get(header, "output")
	self.both = config.get(header,"both")
	self.topSim  = config.get(header,"topSim")
	self.w2v_file = config.get(header,"w2v_file")
	















if __name__=="__main__":
    config_file = sys.argv[1]
    preprocessor = PreProcessor(config_file)
