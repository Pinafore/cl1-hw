# CMSC 723
# Template by: Jordan Boyd-Graber
# Homework submission by: NAME

import sys
from typing import Iterable


import nltk
from nltk.corpus import dependency_treebank
from nltk.classify.maxent import MaxentClassifier
from nltk.classify.util import accuracy
from nltk.parse.dependencygraph import DependencyGraph

from nltk.corpus import dependency_treebank

kROOT = 'ROOT'
VALID_TYPES = set(['s', 'l', 'r'])

def split_data(transition_sequence, proportion_test=10,
               test=False, limit=-1):
    """
    Iterate over stntences in the NLTK dependency parsed corpus and create a
    feature representation of the sentence.

    :param proportion_test: 1 in every how many sentences will be test data?
    :param test: Return test data only if C{test=True}.
    :param limit: Only consider the first C{limit} sentences
    """
    for ii, ss in enumerate(dependency_treebank.parsed_sents()):
        is_test = (ii % proportion_test == 0)

        if limit > 0 and ii > limit:
            break

        if is_test:
            if test:
                for ff in transition_sequence(ss):
                    yield ff.feature_representation()
        else:
            if not test:
                for ff in transition_sequence(ss):
                    yield ff.feature_representation()

class Transition:
    """
    Class to represent a dependency graph transition.
    """
    
    def __init__(self, type, edge=None):
        self.type = type
        self.edge = edge
        self.features = {}

        assert self.type in VALID_TYPES

    def add_feature(self, feature_name, feature_value):
        self.features[feature_name] = feature_value

    def feature_representation(self):
        return (self.features, self.type)

    def pretty_print(self, sentence):
        if self.edge:
            a, b = self.edge
            return "%s\t(%s, %s)" % (self.type,
                                     sentence.get_by_address(a)['word'],
                                     sentence.get_by_address(b)['word'])
        else:
            return self.type
        

def feature_extractor(stack, buffer, index, sentence):
    """
    Given the state of the shift-reduce parser, create a feature vector from the
    sentence.

    :param stack: The list representation of the state's stack.

    :param buffer: The list representnation of the state's buffer.

    :param index: The current offset of the word under consideration (wrt the
    original sentence).

    :return: Yield tuples of feature -> value
    """

    yield ("Buffer size", len(buffer))
    yield ("Stack size", len(buffer))
    





def classifier_transition_sequence(sentence, classifier):
    """
    :param sentence: A dependency parse tree

    :return: A list of transition objects that reconstructs the depndency
    parse tree as predicted by the classifier.

    Unlike transition_sequence, which uses the gold stack and buffer states,
    this will predict given the state as you run the classifier forward.
    """

    # Complete this for extra credit

    return
        
def transition_sequence(sentence: DependencyGraph) -> Iterable[Transition]:
    """
    :param sentence: A dependency parse tree

    :return: A list of transition objects that creates the dependency parse
    tree.
    """
    




    return

def parse_from_transition(word_sequence: Iterable[str], transitions: Iterable[Transition]) -> DependencyGraph:
  """
  :param word_sequence: A list of tagged words (list of pairs of words) that
  need to be built into a tree

  :param transitions: A a list of Transition objects that will build a tree.

  :return: A dependency parse tree in CoNLL format.
  """
  assert len(transitions) >= len(word_sequence), "Not enough transitions"

  # insert root if needed
  if word_sequence[0] != kROOT:
    word_sequence.insert(0, (kROOT, 'TOP'))

  sent = ['']*(len(word_sequence))         


  
  # get the head index of each word

      

  # You're allowed to create your DependencyGraph however you like, but this
  # is how I did it.
  reconstructed = '\n'.join(sent)
  return nltk.parse.dependencygraph.DependencyGraph(reconstructed)

def sentence_attachment_accuracy(reference, sample):
    """
    Given two parse tree transition sequences, compute the number of correct
    attachments (ROOT is always correct)
    """

    return 0

def parse_accuracy(classifier, reference_sentences):
    """
    Compute the average attachment accuracy for a classifier on a corpus
    of sentences.

    """

    num_attachments = 0
    correct = 0
    for sentence in reference_sentences.parsed_sents():
        num_sents += 1
        sequence = transition_sequence(sentence)

        features = [x.feature_representation() for x in sequence]
        predicted_sequence = [classifier(x) for x in features]

        correct += sentence_attachment_accuracy(sequence, predicted_sequence) 
        num_attachments += len(features)

    return 
        

if __name__ == "__main__":

    # Create an example sentence
    sent = nltk.parse.dependencygraph.DependencyGraph(sample_sentence)
    words = [x.split('\t')[0] for x in sample_sentence.split('\n')]
    words = [kROOT] + words

    # See if we set a limit on the number of sentences to consider    
    try:
        lim = int(sys.argv[1])
    except IndexError:
        lim = -1

    print(type(sent))

    index = 1
    for ii in sent.to_conll(4).split("\n"):
        print("%i\t%s" % (index, ii))
        index += 1

    for ii in transition_sequence(sent):
        print(ii.pretty_print(sent))

    train_data = split_data(limit=lim)
    train_features = [x for x in train_data]

    classifier = MaxentClassifier.train(train_features, algorithm='IIS')

    test_data = split_data(test=True, limit=lim)
    test_features = [x for x in test_data]

    # Classification accuracy
    acc = accuracy(classifier, test_features)
    print('Classification Accuracy: %6.4f' % acc)
