# Jordan Boyd_Graber
# 2023
#
# File to define default command line parameters and to instantiate objects
# based on those parameters.  Included in most files in this project.

import logging
import argparse
import json
import gzip

from pandas import read_csv



def add_general_params(parser):
    parser.add_argument('--no_cuda', action='store_true')
    parser.set_defaults(feature=True)
    parser.add_argument('--logging_level', type=int, default=logging.INFO)
    parser.add_argument('--logging_file', type=str, default='qanta.log')
    parser.add_argument('--load', type=bool, default=True)
    print("Setting up logging")

def add_question_params(parser):
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--question_source', type=str, default='gzjson')
    parser.add_argument('--questions', default = "../data/qanta.guesstrain.json.gz",type=str)
    parser.add_argument('--secondary_questions', default = "../data/qanta.guessdev.json.gz",type=str)
    parser.add_argument('--expo_output_root', default="expo/expo", type=str) 

def add_buzzer_params(parser):
    from logistic_buzzer import LogisticParameters
    
    params = {}
    params["logistic"] = LogisticParameters()

    for ii in params:
        params[ii].add_command_line_params(parser)
    
    parser.add_argument('--buzzer_guessers', nargs='+', default = ['Tfidf'], help='Guessers to feed into Buzzer', type=str)
    parser.add_argument('--buzzer_history_length', type=int, default=0, help="How many time steps to retain guesser history")
    parser.add_argument('--buzzer_history_depth', type=int, default=0, help="How many old guesses per time step to keep")    
    parser.add_argument('--features', nargs='+', help='Features to feed into Buzzer', type=str,  default=[])  
    parser.add_argument('--buzzer_type', type=str, default="logistic")
    parser.add_argument('--run_length', type=int, default=100)
    parser.add_argument('--primary_guesser', type=str, default='Tfidf', help="What guesser does buzzer depend on?")

    return params
    
def add_guesser_params(parser):
    params = {}

    params["dan"] = DanParameters()
    params["gpr"] = GprParameters()
    
    for ii in params:
        params[ii].add_command_line_params(parser)
        
    parser.add_argument('--guesser_type', type=str, default="Tfidf")
    # TODO (jbg): This is more general than tfidf, make more general (currently being used by DAN guesser as well)
    parser.add_argument('--guesser_min_length', type=int, help="How long (in characters) must text be before it is indexed?", default=50)
    parser.add_argument('--guesser_max_vocab', type=int, help="How big features/vocab set to use", default=10000)
    parser.add_argument('--guesser_answer_field', type=str, default="page", help="Where is the cannonical answer")    
    parser.add_argument('--guesser_max_length', type=int, help="How long (in characters) must text be to be removed?", default=500)    
    parser.add_argument('--guesser_split_sentence', type=bool, default=True, help="Index sentences rather than paragraphs")
    parser.add_argument('--wiki_min_frequency', type=int, help="How often must wiki page be an answer before it is used", default=10)
    parser.add_argument('--TfidfGuesser_filename', type=str, default="models/TfidfGuesser")
    parser.add_argument('--WikiGuesser_filename', type=str, default="models/WikiGuesser")    
    parser.add_argument('--wiki_zim_filename', type=str, default="data/wikipedia.zim")
    parser.add_argument('--num_guesses', type=int, default=25)

    return params

def setup_logging(flags):
    logging.basicConfig(level=flags.logging_level, force=True)
    
def load_questions(flags, secondary=False):
    questions = None
    
    question_filename = flags.questions
    if secondary:
        question_filename = flags.secondary_questions

    if question_filename == 'tiny':
        from guesser import kTOY_DATA
        questions = kTOY_DATA['tiny'] + kTOY_DATA['tiny']
        return questions
        
    if question_filename == 'mini-train':
        from guesser import kTOY_DATA
        questions = kTOY_DATA['mini-train']
        return questions
    
    if question_filename == 'mini-dev':
        from guesser import kTOY_DATA        
        questions = kTOY_DATA['mini-dev']
        return questions
        
    if flags.questions == 'presidents':
        from president_guesser import kPRESIDENT_DATA
        questions = kPRESIDENT_DATA['dev']
        
    if flags.question_source == 'gzjson':
        logging.info("Loading questions from %s" % question_filename)
        with gzip.open(question_filename) as infile:
            questions = json.load(infile)
    
    if flags.question_source == 'json':
        with open(question_filename) as infile:
            try:
                questions = json.load(infile)
            except UnicodeDecodeError:
                logging.error("Got a Unicode decode error while reading json questions.  This can mean: 1) your data are corrupt (redownload them), 2) you're trying to use json question source on 'gzjson' type data (change the question_source flag)")
            
    if flags.question_source == 'csv':
        questions = read_csv(question_filename)

    if flags.question_source == 'expo':
        questions = ExpoQuestions()
        if flags.questions:
            questions.load_questions(question_filename)
        else:
            questions.debug()
        
    assert questions is not None, "Did not load %s of type %s" % (flags.questions, flags.question_source)

    if flags.limit > 0:
        questions = questions[:flags.limit]

    logging.info("Read %i questions" % len(questions))
        
    return questions

def instantiate_guesser(guesser_type, flags, params, load):
    import torch
    
    cuda = not flags.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    logging.info("Using device '%s' (cuda flag=%s)" % (device, str(flags.no_cuda)))
    
    guesser = None
    logging.info("Initializing guesser of type %s" % guesser_type)
    if guesser_type == "gpr":
        from gpr_guesser import GprGuesser
        logging.info("Loading %s guesser" % guesser_type)
        guesser = GprGuesser(flags.gpr_guesser_filename)
        if load:
            guesser.load()
    if guesser_type == "toy_tfidf":
        from toytfidf_guesser import ToyTfIdfGuesser
        guesser = ToyTfIdfGuesser(flags.TfidfGuesser_filename)
        if load:
            guesser.load()
        
    if guesser_type == "tfidf":
        from tfidf_guesser import TfidfGuesser        
        guesser = TfidfGuesser(flags.TfidfGuesser_filename)  
        if load:                                             
            guesser.load()
    if guesser_type == "Dan":                                
        from dan_guesser import DanGuesser
        dan_params = params["dan"]
        guesser = DanGuesser(dan_params)
        if load:                                                    
            guesser.load()
        else:
            guesser.initialize_model()
    if guesser_type == "president":
        from president_guesser import PresidentGuesser, kPRESIDENT_DATA        
        guesser = PresidentGuesser()
        guesser.train(kPRESIDENT_DATA['train'])
            
    assert guesser is not None, "Guesser (type=%s) not initialized" % guesser_type

    return guesser

def load_guesser(flags, guesser_params, load=False):
    """
    Given command line flags, load a guesser.  Essentially a wrapper for instantiate_guesser because we don't know the type.
    """

    return instantiate_guesser(flags.guesser_type, flags, guesser_params, load)

def load_buzzer(flags, guesser_params, load=False):
    """
    Create the buzzer and its features.
    """
    
    print("Loading buzzer")
    buzzer = None
    if flags.buzzer_type == "logistic":
        from logistic_buzzer import LogisticBuzzer
        buzzer = LogisticBuzzer(flags.logistic_buzzer_filename, flags.run_length, flags.num_guesses)

    if load:
        buzzer.load()

    assert buzzer is not None, "Buzzer (type=%s) not initialized" % flags.buzzer_type

    primary_loaded = 0
    for gg in flags.buzzer_guessers:
        guesser = instantiate_guesser(gg, flags, guesser_params, load=True)
        guesser.load()
        logging.info("Adding %s to Buzzer (total guessers=%i)" % (gg, len(flags.buzzer_guessers)))
        primary = (gg == flags.primary_guesser or len(flags.buzzer_guessers)==1)
        buzzer.add_guesser(gg, guesser, primary_guesser=primary)
        if primary:
            primary_loaded += 1
    assert primary_loaded == 1 or (primary_loaded == 0 and flags.primary_guesser=='consensus'), "There must be one primary guesser"

    print("Initializing features: %s" % str(flags.features))
    print("dataset: %s" % str(flags.questions))

    ######################################################################
    ######################################################################
    ######################################################################
    ######
    ######
    ######  For the feature engineering homework, here's where you need
    ######  to add your features to the buzzer.
    ######
    ######
    ######################################################################
    ######################################################################
    ######################################################################    

    features_added = set()

    for ff in flags.features:
        if ff == "Length":
            from features import LengthFeature
            feature = LengthFeature(ff)
            buzzer.add_feature(feature)
            features_added.add(ff)

    if len(flags.features) != len(features_added):
        error_message = "%i features on command line (%s), but only added %i (%s).  "
        error_message += "Did you add code to params.py's load_buzzer "
        error_message += "to actually add the feature to "
        error_message += "the buzzer?  Or did you forget to increment features_added "
        error_message += "in that function?"
        logging.error(error_message % (len(flags.features), str(flags.features),
                                           len(features_added), str(features_added)))
    return buzzer




class Parameters:
    def __init__(self):
        self.params = []

    def add_command_line_params(self, parser):
        for parameter, param_type, default, description in self.params:
            parser.add_argument("--%s_%s" % (self.name, parameter),
                                type=param_type, default=default,
                                help=description)
            
    def load_command_line_params(self, flags):
        for parameter, _, _, _ in self.params:
            print("Adding param %s" % parameter)
            name =  "%s_%s" % (self.name, parameter)
            value = getattr(flags, name)
            setattr(self, name, value)

    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
            
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)


class GuesserParameters(Parameters):
    guesser_params = [("filename", str, "models/guesser",
                       "Where we save guesser"),
                      ("vocab_size", int, 20000,
                       "Maximum number of words in vocabulary"),
                      ]
    
    def __init__(self):
        Parameters.__init__(self)
        self.params += self.guesser_params

            
class DanParameters(GuesserParameters):
    def __init__(self, customized_params=None):
        GuesserParameters.__init__(self)
        self.name = "dan_guesser"
        if customized_params:
            self.params += customized_params
        else:
            dan_params = [("embed_dim", int, 300, "How many dimensions in word embedding layer"),
                          ("batch_size", int, 120, "How many examples per batch"),
                          ("num_workers", int, 8, "How many workers to serve examples"),
                          ("hidden_units", int, 100, "Number of dimensions of hidden state"),
                          ("max_classes", int, 1000, "Maximum number of answers"),
                          ("ans_min_freq", int, 1, "Frequency of answer count must be above this to be counted"),
                          ("nn_dropout", float, 0.5, "How much dropout we use"),
                          ("device", str, "cuda", "Where we run pytorch inference"),
                          ("num_epochs", int, 20, "How many training epochs"),
                          ("neg_samp", int, 5, "Number of negative training examples"),
                          ("plot_viz", str, "", "Where to plot the state (only works for 2D)"),
                          ("plot_every", int, 10, "After how many epochs do we plot visualization"),
                          ("unk_drop", bool, True, "Do we drop unknown tokens or use UNK symbol"),
                          ("grad_clipping", float, 5.0, "How much we clip the gradients")]
            self.params += dan_params

    # TODO: These should be inherited from base class, remove 
    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
           
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)
            
class GprParameters(GuesserParameters):
    def __init__(self, customized_params=None):
        GuesserParameters.__init__(self)
        self.name = "gpr_guesser"
        if customized_params:
            self.params += customized_params  

