# Author: Jordan Boyd-Graber
# 2013

# File to take guesses and decide if they're correct

import argparse
import logging
import pickle

from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm

from collections import Counter
from collections import defaultdict

from guesser import add_guesser_params
from features import LengthFeature
from parameters import add_buzzer_params, add_question_params, load_guesser, load_buzzer, load_questions, add_general_params, setup_logging, Parameters

def runs(text, run_length):
    """
    Given a quiz bowl questions, generate runs---subsegments that simulate
    reading the question out loud.

    These are then fed into the rest of the system.

    """
    words = text.split()
    assert len(words) > 0
    current_word = 0
    last_run = 0

    for idx in range(run_length, len(text), run_length):
        current_run = text.find(" ", idx)
        if current_run > last_run and current_run < idx + run_length:
            yield text[:current_run]
            last_run = current_run

    yield text

def sentence_runs(sentences, run_length):
    """
    Generate runs, but do it per sentence (always stopping at sentence boundaries).
    """
    
    previous = ""
    for sentence in sentences:
        for run in runs(sentence, run_length):
            yield previous + run
        previous += sentence
        previous += "  "

class BuzzerParameters(Parameters):
    buzzer_params = [("filename", str, "models/buzzer", "Where we save buzzer")]

    def __init__(self):
        Parameters.__init__(self)
        self.params += self.buzzer_params
                     
        
class Buzzer:
    """
    Base class for any system that can decide if a guess is correct or not.
    """
    
    def __init__(self, filename, run_length, num_guesses=1):
        self.filename = filename
        self.num_guesses = num_guesses
        self.run_length=run_length
        
        self._runs = []
        self._questions = []
        self._answers = []
        self._training = []
        self._correct = []
        self._features = []
        self._metadata = []
        self._feature_generators = []
        self._guessers = {}

        logging.info("Buzzer using run length %i" % self.run_length)
        
        self._finalized = False
        self._primary_guesser = None
        self._classifier = None
        self._featurizer = None

    def add_guesser(self, guesser_name, guesser, primary_guesser=False):
        """
        Add a guesser identified by guesser_name to the set of guessers.

        If it is designated as the primary_guesser, then its guess will be
        chosen in the case of a tie.

        """

        assert not self._finalized, "Trying to add guesser after finalized"
        assert guesser_name != "consensus"
        assert guesser_name is not None
        assert guesser_name not in self._guessers
        self._guessers[guesser_name] = guesser
        if primary_guesser:
            self._primary_guesser = guesser_name

    def add_feature(self, feature_extractor):
        """
        Add a feature that the buzzer will use to decide to trust a guess.
        """

        assert not self._finalized, "Trying to add feature after finalized"
        assert feature_extractor.name not in [x.name for x in self._feature_generators]
        assert feature_extractor.name not in self._guessers
        self._feature_generators.append(feature_extractor)
        logging.info("Adding feature %s" % feature_extractor.name)
        
    def featurize(self, question, run_text, guess_history,
                  guesses=None, guess_count=None):
        """
        Turn a question's run into features.

        guesses -- A dictionary of all the guesses.  If None, will regenerate the guesses.
        guess_count -- A count of all of the other guesses
        """
        
        features = {}
        guess = None

        # If we didn't cache the guesses, compute them now
        if guesses is None:
            guesses = {}            
            for gg in self._guessers:
                guesses[gg] = self._guessers[gg](run_text)

        for gg in self._guessers:
            assert gg in guesses, "Missing guess result from %s" % gg
            result = list(guesses[gg])[0]
            if gg == self._primary_guesser:
                guess = result["guess"]

            # This feature could be useful, but makes the formatting messy
            # features["%s_guess" % gg] = result["guess"]
            features["%s_confidence" % gg] = result["confidence"]

            for other_guesses in guesses[gg]:                         
                all_guesses[other_guesses["guess"]] += 1              

        if len(all_guesses) > 1:                                            
            consensus_guess, consensus_count = all_guesses.most_common(1)[0]
            if consensus_guess == guess:                                    
                logging.debug("Consensus guess matches to guess %s" % guess)
                features["consensus_count"] = consensus_count - 1
                features["consensus_match"] = 1
            else:                                                           
                features["consensus_count"] = all_guesses[guess] - 1
                features["consensus_match"] = 0

        for ff in self._feature_generators:
            for feat, val in ff(question, run_text, guess, guess_history, guesses):
                features["%s_%s" % (ff.name, feat)] = val

        assert guess is not None or guesses[self._primary_guesser][0]["guess"] is None, \
          "Guess was not set (Primary=%s, others=%s) Guesses=%s" % \
          (self._primary_guesser, str(set(guesses)), str(guesses))
        return guess, features

    def finalize(self):
        """
        Set the guessers (will prevent future addition of features and guessers)
        """
        
        self._finalized = True
        if self._primary_guesser is None:
            self._primary_guesser = "consensus"
        
    def add_data(self, questions, answer_field="page"):
        """
        Add data and store them so you can later create features for them
        """
        
        self.finalize()
        
        num_questions = 0
        logging.info("Generating runs of length %i" % self.run_length)        
        for qq in tqdm(questions):
            answer = qq[answer_field]
            text = qq["text"]
            # Delete these fields so you can't inadvertently cheat while
            # creating features.  However, we need the answer for the labels.
            del qq[answer_field]
            if "answer" in qq:
                del qq["answer"]
            if "page" in qq:
                del qq["page"]
            del qq["first_sentence"]
            del qq["text"]

            for rr in runs(text, self.run_length):
                self._answers.append(answer)
                self._runs.append(rr)
                self._questions.append(qq)

    def build_features(self, history_length=0, history_depth=0):
        """
        After all of the data has been added, build features from the guesses and questions.
        """
        from eval import rough_compare

        all_guesses = {}
        logging.info("Building guesses from %s" % str(self._guessers.keys()))
        for guesser in self._guessers:
            all_guesses[guesser] = self._guessers[guesser].batch_guess(self._runs, self.num_guesses)
            logging.info("%10i guesses from %s" % (len(all_guesses[guesser]), guesser))
            assert len(all_guesses[guesser]) == len(self._runs), "Guesser %s wrong size" % guesser
            
        assert len(self._questions) == len(self._answers)
        assert len(self._questions) == len(self._runs)        
            
        num_runs = len(self._runs)

        logging.info("Generating all features")
        for question_index in tqdm(range(num_runs)):
            question_guesses = dict((x, all_guesses[x][question_index]) for x in self._guessers)
            guess_history = defaultdict(dict)
            for guesser in question_guesses:
                # print("Building history with depth %i and length %i" % (history_depth, history_length))
                # TODO(jbg): I think this is inefficient, shouldn't this be using question_guesses, not all_guesses?
                guess_history[guesser] = dict((time, guess[:history_depth]) for time, guess in enumerate(all_guesses[guesser]) if time < question_index and time > question_index - history_length)

            # print(guess_history)
            question = self._questions[question_index]
            run = self._runs[question_index]
            answer = self._answers[question_index]
            guess, features = self.featurize(question, run, guess_history, question_guesses)
            
            self._features.append(features)
            self._metadata.append({"guess": guess, "answer": answer, "id": question["qanta_id"], "text": run})

            correct = rough_compare(guess, answer)
            logging.debug(str((correct, guess, answer)))
                
            self._correct.append(correct)

                
            assert len(self._correct) == len(self._features)
            assert len(self._correct) == len(self._metadata)
        
        assert len(self._answers) == len(self._correct), \
            "Answers (%i) does not match correct (%i)" % (len(self._answers), len(self._features))
        assert len(self._answers) == len(self._features)        

        if "GprGuesser" in self._guessers:
            self._guessers["GprGuesser"].save()
            
        return self._features
    
    def single_predict(self, run):
        """
        Make a prediction from a single example ... this us useful when the code
        is run in real-time.

        """
        
        guess, features = self.featurize(None, run)

        X = self._featurizer.transform([features])

        return self._classifier.predict(X), guess, features
    
           
    def predict(self, questions, online=False):
        """
        Predict from a large set of questions whether you should buzz or not.
        """
        
        assert self._classifier, "Classifier not trained"
        assert self._featurizer, "Featurizer not defined"
        assert len(self._features) == len(self._questions), "Features not built.  Did you run build_features?"
        X = self._featurizer.transform(self._features)

        return self._classifier.predict(X), X, self._features, self._correct, self._metadata

    def write_json(self, output_filename):
        import json
        
        vocab = set()
        with open(output_filename, 'w') as outfile:
            for features, correct, meta in zip(self._features, self._correct, self._metadata):
                assert "label" not in features
                new_features = {}

                new_features['guess:%s' % meta['guess']] = 1                
                for key in features:
                    if isinstance(features[key], str):
                        new_features["%s:%s" % (key, features[key])] = 1
                    else:
                        new_features[key] = features[key]
                for key in new_features:
                    vocab.add(key)

                new_features['label'] = correct
                    
                outfile.write("%s\n" % json.dumps(new_features))
        vocab = list(vocab)
        vocab.sort()
        return ['BIAS_CONSTANT'] + vocab
    
    def load(self):
        """
        Load the buzzer state from disk
        """
        
        with open("%s.featurizer.pkl" % self.filename, 'rb') as infile:
            self._featurizer = pickle.load(infile)        
    
    def save(self):
        """
        Save the buzzer state to disck
        """
        
        for gg in self._guessers:
            self._guessers[gg].save()
        with open("%s.featurizer.pkl" % self.filename, 'wb') as outfile:
            pickle.dump(self._featurizer, outfile)  
    
    def train(self):
        """
        Learn classifier parameters from the data loaded into the buzzer.
        """

        assert len(self._features) == len(self._correct)        
        self._featurizer = DictVectorizer(sparse=True)
        X = self._featurizer.fit_transform(self._features)
        return X

if __name__ == "__main__":
    # Train a simple model on QB data, save it to a file
    import argparse
    parser = argparse.ArgumentParser()

    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)
    question_params = add_question_params(parser)
    flags = parser.parse_args()
    setup_logging(flags)    

    guesser = load_guesser(flags, guesser_params)    
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags, question_params)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    buzzer.train()
    buzzer.save()

    if flags.limit == -1:
        print("Ran on %i questions" % len(questions))
    else:
        print("Ran on %i questions of %i" % (flags.limit, len(questions)))
    
