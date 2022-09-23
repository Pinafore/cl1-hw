# Author: Jordan Boyd-Graber
# Date: 22. Sept 2022
# Homework Template: A bad searcher that tries to find a target synset in WordNet

from nltk.corpus import wordnet as wn

class Searcher:
    """
    Class to search WordNet through logical queries.
    """
    
    def __init__(self):
        
        # Feel free to add your own data members
        self._searched = {}

    def check_lemma(self, oracle, candidate):
        """
        Convenience method to check whether two synsets are the same by seeing if the oracle matches a lemma of this synset.
        
        Keyword Arguments:
        oracle -- The oracle that can check whether the candidate matches
        candidate -- The synset to check
        """
        print("Searching %s" % str(candidate))        
        lemma = candidate.lemmas()[0]
        self._searched[candidate] = oracle.there_exists('lemmas', [lemma])
        
    def __call__(self, oracle):
        """
        Given an oracle, return the synset that the oracle has as its target.
        
        Keyword Arguments:
        oracle -- The oracle being searched
        """

        # Feel free to change the code within
        # --------------------------------------

        # Start at the top, go breadth first
        self.check_lemma(oracle, wn.synset('entity.n.01'))

        while not any(self._searched.values()):
            previously_searched = list(self._searched.keys())
            for parent in previously_searched:
                for candidate in parent.hyponyms():
                    if not candidate in self._searched:
                        self.check_lemma(oracle, candidate)
        # ---------------------------------------
        assert any(self._searched.values()), "Searched all of WN without finding it!"
        found = [x for x in self._searched if self._searched[x]][0]

        return found

if __name__ == "__main__":
    from wn_eval import Oracle

    oracle = Oracle(wn.synset('dog.n.01'))

    searcher = Searcher()
    print("Search result is:")
    print(searcher(oracle))
    print("Took %i steps to get there" % oracle.num_queries())
        
        
