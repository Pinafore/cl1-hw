import nltk
from collections import defaultdict

class PcfgEstimator:
    """
    Estimates the production probabilities of a PCFG
    """
    
    def __init__(self):
        self._counts = defaultdict(nltk.FreqDist)

    def add_sentence(self, sentence):
        """
        Add the sentence to the dataset
        """

        assert isinstance(sentence, nltk.tree.Tree), "Can only add counts from a tree"

        # FINISH THIS!

    def query(self, lhs, rhs):
        """
        Returns the MLE probability of this production
        """
        
        return self._counts[lhs].freq(rhs)
