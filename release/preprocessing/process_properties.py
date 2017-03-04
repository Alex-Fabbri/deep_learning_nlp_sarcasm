import sys
import ConfigParser

class PreProcessor:
    def __init__(self, config_file):
	config = ConfigParser.RawConfigParser()
	config.read(config_file)
        try:
	    header = "PropertySection"
	    self.input = config.get(header, "input")
	    self.output = config.get(header, "output")
	    self.both = config.get(header,"both")
	    self.topSim  = config.get(header,"topSim")
            self.separate = config.get(header,"separate")
	    self.w2v_file = config.get(header,"w2v_file")
            self.w2v_type = config.get(header,"w2v_type")
        except:
            print("make sure to include all parameters")
            exit()
        
