# Jordan Boyd-Graber
# 2023

# Base class for our guessers

import os
import re
import json
import pickle
import logging

from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple

from nltk.tokenize import sent_tokenize, word_tokenize

alphanum = re.compile('[^a-zA-Z0-9]')

from parameters import load_guesser, load_questions, setup_logging
from parameters import add_general_params, add_guesser_params, add_general_params, add_question_params

kTOY_DATA = {"tiny": [{"text": "currency england", "page": "Pound"},
                      {"text": "currency russia", "page": "Rouble"},
                      {"text": "capital russia", "page": "Moscow"},                      
                      {"text": "capital england", "page": "London"}],
             "train": [{'page': 'Maine', 'text': 'For 10 points, name this New England state with capital at Augusta.'},
                       {'page': 'Massachusetts', 'text': 'For ten points, identify this New England state with capital at Boston.'},
                       {'page': 'Boston', 'text': 'For 10 points, name this city in New England, the capital of Massachusetts.'},
                       {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Pride and Prejudice.'},
                       {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Emma and Pride and Prejudice.'},
                       {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'For 10 points, name this composer of Magic Flute and Don Giovanni.'},
                       {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'Name this composer who wrote a famous requiem and The Magic Flute.'},
                       {'page': "Gresham's_law", 'text': 'For 10 points, name this economic principle which states that bad money drives good money out of circulation.'},
                       {'page': "Gresham's_law", 'text': "This is an example -- for 10 points \\-- of what Scotsman's economic law, which states that bad money drives out good?"},
                       {'page': "Gresham's_law", 'text': 'FTP name this economic law which, in simplest terms, states that bad money drives out the good.'},
                       {'page': 'Rhode_Island', 'text': "This colony's Touro Synagogue is the oldest in the United States."},
                       {'page': 'Lima', 'text': 'It is the site of the National University of San Marcos, the oldest university in South America.'},
                       {'page': 'College_of_William_&_Mary', 'text': 'For 10 points, identify this oldest public university in the United States, a college in Virginia named for two monarchs.'}],
              "dev": [{'text': "This capital of England", "top": 'Maine', "second": 'Boston'},
                      {'text': "The author of Pride and Prejudice", "top": 'Jane_Austen',
                           "second": 'Jane_Austen'},
                      {'text': "The composer of the Magic Flute", "top": 'Wolfgang_Amadeus_Mozart',
                           "second": 'Wolfgang_Amadeus_Mozart'},
                      {'text': "The economic law that says 'good money drives out bad'",
                           "top": "Gresham's_law", "second": "Gresham's_law"},
                      {'text': "located outside Boston, the oldest University in the United States",
                           "top": 'College_of_William_&_Mary', "second": 'Rhode_Island'}]
                }

def word_overlap(query, page):
    """
    Checks overlap between two strings, used in checking if an answer is a match.
    """
    
    query_words = set(alphanum.split(query))
    page_words = set(alphanum.split(page))

    return len(query_words.intersection(page_words)) / len(query_words)


def print_guess(guess, max_char=20):
    """
    Utility function for printing out snippets (up to max_char) of top guesses.
    """
    
    standard = ["guess", "confidence", "question"]
    output = ""

    for ii in standard:
        if ii in guess:
            if isinstance(guess[ii], float):
                short = "%0.2f" % guess[ii]
            else:
                short = str(guess[ii])[:max_char]
            output += "%s:%s\t" % (ii, short)
            
    return output
    
class Guesser:
    """
    Base class for guessers.  If it itself is instantiated, it will only guess
    one thing (the default guess).  This is useful for unit testing.
    """
    
    def __init__(self, default_guess="Les Misérables (musical)"):
        self._default_guess = default_guess
        self.phrase_model = None
        None

    @staticmethod
    def split_examples(training_data, answer_field, split_by_sentence=True, min_length=-1,
                        max_length=-1):
        """
        Given training data, create a mapping of of answers to the question with that answer.
        What qualifies as the answer is specified by the "answer_field".  

        If split_by_sentence is true, it creates individual questions
        for each of the sentences in the original question.
        """
        from collections import defaultdict
        from tqdm import tqdm
        
        answers_to_questions = defaultdict(set)
        if split_by_sentence:
            for qq in tqdm(training_data):
                for ss in sent_tokenize(qq["text"]):
                    if (min_length < 0 or len(ss) > min_length) and \
                        (max_length < 0 or len(ss) < max_length):
                        answers_to_questions[qq[answer_field]].add(ss)
        else:
            for qq in tqdm(training_data):
                text = qq["text"]
                if (min_length < 0 or len(text) > min_length) and \
                    (max_length < 0 or len(text) < max_length):
                    answers_to_questions[qq[answer_field]].add(qq["text"])
        return answers_to_questions

    @staticmethod
    def filter_answers(questions_keyed_by_answers, remove_missing_pages=False,
                       answer_lookup=None):
        """
        Remove missing answers or answers that aren't included in lookup.
        """
        
        from tqdm import tqdm        
        answers = []
        questions = []
        for answer in tqdm(questions_keyed_by_answers):
            if remove_missing_pages and answer is None or answer is not None and answer.strip() == '':
                continue
            elif answer_lookup is not None and answer not in answer_lookup:
                continue
            for question in questions_keyed_by_answers[answer]:
                answers.append(answer)
                questions.append(question)

        return questions, answers
        

    def train(self, training_data, answer_field, split_by_sentence, min_length=-1,
              max_length=-1, remove_missing_pages=True):
        """
        Use a tf-idf vectorizer to analyze a training dataset and to process
        future examples.
        
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        min_length -- ignore all text segments less than this length (-1 for no limit)
        max_length -- ingore all text segments longer than this length (-1 for no length)
        remove_missing_pages -- remove pages without an answer_field
        """

        answers_to_questions = self.split_examples(training_data, answer_field, split_by_sentence,
                                                   min_length, max_length)
        self.questions, self.answers = self.filter_answers(answers_to_questions)
        logging.info("Trained with %i questions and %i answers filtered from %i examples" %
                     (len(self.questions), len(self.answers), len(training_data)))

        return answers_to_questions

    def find_phrases(self, questions : Iterable[str]):
        """
        Using the training question, find phrases that ofen appear together.

        Saves the resulting phrase detector to phrase_model, which can
        then be used to tokenize text using the phrase_tokenize
        function.
        """
        assert len(questions) > 0, "Cannot find phrases without questions"
        from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

        # TODO: it might be good to exclude punctuation here
        sentences = []
        for qq in self.questions:
            for ss in sent_tokenize(qq):
                sentences.append(word_tokenize(ss))

        self.phrase_model = Phrases(sentences, connector_words=ENGLISH_CONNECTOR_WORDS, min_count=30)

    def phrase_tokenize(self, question: str) -> Iterable[str]:
        """
        Given text (a question), tokenize the text and look for phrases.
        """
        assert self.phrase_model is not None
        # Todo: perhaps include additional normalization in this function (e.g., lemmatization)
        return self.phrase_model[word_tokenize(question)]
        

    def batch_guess(self, questions, n_guesses=1):
        """
        Given a list of questions, create a batch set of predictions.

        This should be overridden my more efficient implementations in subclasses.
        """
        from tqdm import tqdm
        guesses = []
        logging.info("Generating guesses for %i new question" % len(questions))
        for question in tqdm(questions):
            new_guesses = self(question, n_guesses)
            guesses.append(new_guesses)
        return guesses

    def save(self):
        """
        Save the Guesser's information to a file.  
        This will normally be handled by the subclass.
        """

        path = self.filename
        
        if self.phrase_model is not None:
            filename = "%s.phrase.pkl" % path
            logging.info("Writing Guesser phrases to %s" % filename)
            self.phrase_model.save(filename)

    def save_questions_and_answers(self):
        path = self.filename                
        with open("%s.questions.pkl" % path, 'wb') as f:
            pickle.dump(self.questions, f)

        with open("%s.answers.pkl" % path, 'wb') as f:
            pickle.dump(self.answers, f)

    def load_questions_and_answers(self):
        path = self.filename                
        with open("%s.questions.pkl" % path, 'rb') as f:
            self.questions = pickle.load(f)

        with open("%s.answers.pkl" % path, 'rb') as f:
            self.answers = pickle.load(f)        

        logging.info("Loading %i questions and %i answers" %
                     (len(self.questions), len(self.answers)))
        assert len(self.questions)==len(self.answers), "Question size mismatch"   
            
    def load(self):
        """
        Load the guesser information that's been saved to a file.  

        Normally the heavy lifting is done by a subclass.
        """
        path = self.filename        
        filename = "%s.phrase.pkl" % path
        try:
            from gensim.models.phrases import Phrases
            self.phrase_model = Phrases.load(filename)
        except FileNotFoundError:
            self.phrase_model = None
                
    def __call__(self, question, n_guesses=1):
        """
        Generate a guess set from a single question.
        """
        return [{"guess": self._default_guess, "confidence": 1.0}]
1

if __name__ == "__main__":
    # Train a guesser and save it to a file
    import argparse
    parser = argparse.ArgumentParser()
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    question_params = add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)    
    guesser = load_guesser(flags, guesser_params)
    questions = load_questions(flags)
    # TODO(jbg): Change to use huggingface data, as declared in flags

    if flags.guesser_type == 'Wiki':
        guesser.init_wiki(flags.wiki_zim_filename)        
        train_result = guesser.train(questions,
                                     flags.guesser_answer_field,
                                     flags.guesser_split_sentence,
                                     flags.guesser_min_length,
                                     flags.guesser_max_length,
                                     flags.wiki_min_frequency)
        # The WikiGuesser has some results (text from asked about Wikipedia
        # pages) from saving and we want to cache them to a file
        guesser.save()
    elif flags.guesser_type == 'President':
        from president_guesser import kPRESIDENT_DATA
        guesser.train(kPRESIDENT_DATA['train'])
    elif flags.guesser_type == "Dan":
        dev_exs = load_questions(flags, secondary=True)
        guesser.set_eval_data(dev_exs)
        guesser.train_dan()
        guesser.save()
    elif flags.guesser_type == "ToyTfidf":
        guesser.train(questions,
                      flags.guesser_answer_field,
                      flags.guesser_split_sentence)
        guesser.save()
    else:
        if flags.guesser_type not in ['Tfidf']:
            logging.info("Training with default guesser API (gave %s), this might mean something has gone wrong if you wanted to match ToyTfidf, President, Wiki, or Dan" %
                        flags.guesser_type)
        guesser.train(questions,
                      flags.guesser_answer_field,
                      flags.guesser_split_sentence,
                      flags.guesser_min_length,
                      flags.guesser_max_length)
        # DAN Guesser 
        guesser.save()
