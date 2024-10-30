# CMSC 723
# Template by: Jordan Boyd-Graber
# Homework submission by: NAME

import sys
from typing import Iterable, Tuple, Dict


import nltk
from nltk.corpus import dependency_treebank
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.classify.api import ClassifierI
from nltk.parse.dependencygraph import DependencyGraph

from nltk.corpus import dependency_treebank

kROOT = 'ROOT'
VALID_TYPES = set(['s', 'l', 'r'])


def flatten(xss: Iterable) -> Iterable:
    """
    Flatten a list of list into a list
    """
    return [x for xs in xss for x in xs]

def split_data(transition_sequence, proportion_test=10,
               generate_test=False, limit=-1):
    """
    Iterate over stntences in the NLTK dependency parsed corpus and create a
    feature representation of the sentence.

    :param proportion_test: 1 in every how many sentences will be test data?
    :param test: Return test data only if C{test=True}.
    :param limit: Only consider the first C{limit} sentences
    """
    for ii, ss in enumerate(dependency_treebank.parsed_sents()):
        item_is_test = (ii % proportion_test == 0)

        if limit > 0 and ii > limit:
            break

        example = {"sentence": ss, "features": []}
        for ff in transition_sequence(ss):
            example["features"].append(ff.feature_representation())

        if item_is_test and generate_test:
            yield example
        elif not generate_test and not item_is_test:
            yield example

class Transition:
    """
    Class to represent a dependency graph transition.
    """
    
    def __init__(self, type, edge=None):
        self.type = type
        self.edge = edge
        self.features = {}

        assert self.type in VALID_TYPES

    def __str__(self):
        return "Transition(%s, %s)" % (self.type, str(self.edge))
        
    def add_feature(self, feature_name: str, feature_value: float):
        """
        Add a feature to the transition.

        :param feature_name: The name of the feature
        :param feature_value: The value of the feature
        """

        self.features[feature_name] = feature_value

    def feature_representation(self) -> Tuple[Dict[str, float], str]:
        """
        Create a training instance for a classifier: the classifier will predict 
        the transition type from the features you create.
        """

        return (self.features, self.type)

    def pretty_print(self, sentence):
        """
        Pretty print the transition that is a part of the sentence.

        :param sentence: The sentence that the transition is a part of
        """

        if self.edge:
            a, b = self.edge
            return "%s\t(%s, %s)" % (self.type,
                                     sentence.get_by_address(a)['word'],
                                     sentence.get_by_address(b)['word'])
        else:
            return self.type
        
class ShiftReduceState:
    """
    Class to represent the state of the shift-reduce parser.
    """

    def __init__(self, words: Iterable[str], pos: Iterable[str]):
        """
        :param words: A list of words
        :param pos: A list of POS tags
        """

        assert words[0] == kROOT, "First word must be ROOT"
        assert len(words) == len(pos), "Words and POS tags must be the same length"

        self.words = words
        self.pos = pos

        self.edges = []

        self.stack = [0]
        self.buffer = list(range(1, len(words)))
        self.buffer.reverse()

    def pretty_print(self):
        return "Stack: %s\n Buffer: %s\n Edges: %s" % (str(self.stack), str(self.buffer), str(self.edges))

    def apply(self, transition: Transition):
        
        if transition.type == 's':
            self.shift()
        elif transition.type == 'l':
            self.left_arc()
        elif transition.type == 'r':
            self.right_arc()

    def shift(self) -> Transition:
        """
        Shift the top of the buffer to the stack and return the transition.
        """

        index = -1
        assert len(self.buffer) > 0, "Buffer is empty for shift"
        # Implement this

        return Transition('s', None)

    def left_arc(self) -> Transition:
        """
        Create a new left dependency edge and return the transition.
        """

        stack_top = -1

        assert len(self.buffer) > 0, "Buffer is empty for left arc"
        assert len(self.stack) > 0, "Stack is empty for left arc"

        # Implement this


        return Transition('l', (buffer_top, stack_top))

    def right_arc(self) -> Transition:
        """
        Create a new right dependency edge and return the transition.
        """

        stack_top = -1

        assert len(self.buffer) > 0, "Buffer is empty for right arc"
        assert len(self.stack) > 0, "Stack is empty for right arc"

        # Implement this

        return Transition('r', (stack_top, buffer_top))
    
    def feature_extractor(self, index: int) -> Iterable[Tuple[str, float]]:
        """
        Given the state of the shift-reduce parser, create a feature vector from the
        sentence.

        :param index: The current offset of the word under consideration (wrt the
        original sentence).

        :return: Yield tuples of feature -> value
        """

        yield ("Buffer size", len(self.buffer))
        yield ("Stack size", len(self.stack))
    
        # Implement this





def heuristic_transition_sequence(sentence: DependencyGraph) -> Iterable[Transition]:
    """
    Implement this for extra credit
    """

    return []

def classifier_transition_sequence(classifier: MaxentClassifier, sentence: DependencyGraph) -> Iterable[Transition]:
    """
    Unlike transition_sequence, which uses the gold stack and buffer states,
    this will predict given the state as you run the classifier forward.

    :param sentence: A dependency parse tree

    :return: A list of transition objects that reconstructs the depndency
    parse tree as predicted by the classifier.
    """

    # Complete this for extra credit

    return
        
def transition_sequence(sentence: DependencyGraph) -> Iterable[Transition]:
    """
    :param sentence: A dependency parse tree

    :return: A list of transition objects that creates the dependency parse
    tree.
    """

 
    

    # Exclude the root node





    return
    yield # We write this yield to make the function iterable

def parse_from_transition(word_sequence: Iterable[Tuple[str, str]], transitions: Iterable[Transition]) -> DependencyGraph:
  """
  :param word_sequence: A list of tagged words (list of pairs of words and their POS tags) that
  need to be built into a tree

  :param transitions: A a list of Transition objects that will build a tree.

  :return: A dependency parse tree
  """
  assert len(transitions) >= len(word_sequence), "Not enough transitions"

  # insert root if needed
  if word_sequence[0][0] != kROOT:
    word_sequence.insert(0, (kROOT, 'TOP'))

  sent = ['']*(len(word_sequence))         


  
  # get the head index of each word

      

  # You're allowed to create your DependencyGraph however you like, but this
  # is how I did it.
  reconstructed = '\n'.join(sent)
  return nltk.parse.dependencygraph.DependencyGraph(reconstructed)

def sentence_attachment_accuracy(reference: DependencyGraph, sample: DependencyGraph) -> float :
    """
    Given two parse tree transition sequences, compute the number of correct
    attachments (ROOT is always correct)
    """

    correct = 0                                                            
    return correct

def attachment_accuracy(classifier: ClassifierI, reference_sentences: Iterable[DependencyGraph]) -> float:
    """
    Compute the average attachment accuracy for a classifier on a corpus
    of sentences.
    """
    correct = 0
    num_attachments = 0
    for example in reference_sentences:
        # Implement this for extra credit
        sentence = example["sentence"]
        num_attachments += len(sentence.nodes) - 1                                     

    return correct / num_attachments

def classifier_accuracy(classifier: ClassifierI, reference_transitions: Iterable[Tuple[Dict[str, float], str]]) -> float:
    """
    Compute the average attachment accuracy for a classifier on a corpus
    of sentences.

    """

    correct = 0
    total_examples = 0
    for sentence in reference_transitions:
        predictions = [classifier.classify(x[0]) for x in sentence["features"]]
        labels = [x[1] for x in sentence["features"]]

        correct += sum(1 for x, y in zip(predictions, labels) if x == y)
        total_examples += len(predictions)
    return correct / total_examples
        

if __name__ == "__main__":
    from itertools import chain
    # Create an example sentence
    from test_dependency import kCORRECT
    sent = nltk.parse.dependencygraph.DependencyGraph(kCORRECT)
    words = [x.split('\t')[0] for x in kCORRECT.split('\n')]
    words = [kROOT] + words

    for ii in transition_sequence(sent):
        print(ii.pretty_print(sent))

    train_data = list(split_data(transition_sequence))
    test_data = list(split_data(transition_sequence, generate_test=True))

    print("Training data: %i" % len(train_data))
    print("Test data: %i" % len(test_data))

    feature_examples = []
    for sentence in train_data:
        feature_examples += sentence["features"]
    classifier = MaxentClassifier.train(feature_examples, algorithm='IIS', max_iter=25, min_lldelta=0.001)

    # Classification accuracy
    classifier_acc = classifier_accuracy(classifier, test_data)
    print('Held-out Classification Accuracy: %6.4f' % classifier_acc)

    attachment_acc = attachment_accuracy(classifier, test_data)
    print('Held-out Attachment Accuracy:     %6.4f' % attachment_acc)

