import sys
import ConfigParser

class PreProcessor:
    def __init__(self,config_file):
	config = ConfigParser.RawConfigParser()
	config.read(config_file)
        try:
	    header = "PropertySection"
	    self.input = config.get(header, "input")
	    self.output = config.get(header, "output")
            self.vocab_file = config.get(header, "vocab_file")
            self.data_type = config.get(header, "data_type")
            self.test_type = config.get(header, "test_type")
	    self.both = config.get(header,"both")
	    self.topSim  = config.get(header,"topSim")
            self.separate = config.get(header,"separate")
            self.separate_attention_context = config.get(header, "separate_attention_context")
            self.separate_attention_response = config.get(header, "separate_attention_response")
            self.separate_attention_context_words = config.get(header, "separate_attention_context_words")
            self.separate_attention_response_words = config.get(header, "separate_attention_response_words")
            self.interaction = config.get(header, "interaction")
            self.lastSent = config.get(header,"lastSent")
	    self.w2v_file = config.get(header,"w2v_file")
            self.w2v_type = config.get(header,"w2v_type")
            self.K = config.get(header,"K")
            self.num_hidden = config.get(header,"num_hidden")
            self.batch_size = config.get(header,"batch_size")
            self.num_epochs = config.get(header,"num_epochs")
            self.lstm = config.get(header,"lstm")
            self.attention = config.get(header,"attention")
            self.attention_words = config.get(header,"attention_words")
            self.attention_sentences = config.get(header, "attention_sentences")
        except:
            print("check the parameters that you entered in the config file")
            exit()
    def set_target(self,target):
        self.target = target
