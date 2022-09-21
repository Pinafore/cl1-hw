from nltk.corpus import wordnet as wn

class Searcher:
    def __init__(self):
        self._searched = {}

    def check_lemma(self, oracle, candidate):
        print("Searching %s" % str(candidate))        
        lemma = candidate.lemmas()[0]
        self._searched[candidate] = oracle.there_exists('lemmas', [lemma])
        
    def __call__(self, oracle):

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
        
        
