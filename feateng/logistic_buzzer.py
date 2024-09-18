# Jordan Boyd-Graber
# 2023
#
# Buzzer using Logistic Regression

import pickle

import logging

from sklearn.linear_model import LogisticRegression
from buzzer import BuzzerParameters

from buzzer import Buzzer

class LogisticParameters(BuzzerParameters):
    def __init__(self, customized_params=None):
        BuzzerParameters.__init__(self)
        self.name = "logistic_buzzer"
        if customized_params:
            self.params += customized_params

    # TODO: These should be inherited from base class, remove 
    def __setitem__(self, key, value):
        assert hasattr(self, key), "Missing %s, options: %s" % (key, dir(self))
        setattr(self, key, value)
           
    def set_defaults(self):
        for parameter, _, default, _ in self.params:
            name = "%s_%s" % (self.name, parameter)
            setattr(self, name, default)                

class LogisticBuzzer(Buzzer):
    """
    Logistic regression classifier to predict whether a buzz is correct or not.
    """

    def train(self):
        X = Buzzer.train(self)
        
        self._classifier = LogisticRegression()
        self._classifier.fit(X, self._correct)

    def save(self):
        Buzzer.save(self)
        path = "%s.model.pkl" % self.filename
        with open(path, 'wb') as outfile:
            pickle.dump(self._classifier, outfile)
        logging.info("Saving buzzer to %s" % path)

    def load(self):
        Buzzer.load(self)
        path = "%s.model.pkl" % self.filename
        with open(path, 'rb') as infile:
            self._classifier = pickle.load(infile)
        logging.info("Reading buzzer from %s" % path)
