import sys

import nltk
from nltk.corpus import dependency_treebank
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy

VALID_TYPES = set(['s', 'l', 'r'])

class Transition:
    def __init__(self, type, edge=None):
        self._type = type
        self._edge = edge
        assert self._type in VALID_TYPES

    def pretty_print(self, sentence):
        if self._edge:
            a, b = self._edge
            return "%s\t(%s, %s)" % (self._type,
                                     sentence.get_by_address(a)['word'],
                                     sentence.get_by_address(b)['word'])
        else:
            return self._type

def transition_sequence(sentence):
    """
    Return the sequence of shift-reduce actions that reconstructs the input sentence.
    """

    sentence_length = len(sentence.nodes)
    for ii in range(sentence_length - 1):
        yield Transition('s')
    for ii in range(sentence_length - 1, 1, -1):
        yield Transition('r', (ii - 1, ii))
    yield Transition('s')
