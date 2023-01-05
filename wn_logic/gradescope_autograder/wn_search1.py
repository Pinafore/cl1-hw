import nltk
from regex import R
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.wordnet import Lemma
from wn_eval import Oracle
from typing import Union


class Searcher:
    """
    Class to search WordNet through logical queries.
    """
    
    def __init__(self):
        
        # Feel free to add your own data members
        self._searched = {}
        self.all_synsets = list(wn.all_synsets('n'))

    def check(self, oracle: Oracle, candidate: Synset) -> bool:
        """
        Convenience method to check whether two synsets are the same
        and storing the result.
        
        Keyword Arguments:
        oracle -- The oracle that can check whether the candidate matches
        candidate -- The synset to check
        """
        # print("Searching %s" % str(candidate))        
        self._searched[candidate] = oracle.check(candidate)
        return self._searched[candidate]
        
    def __call__(self, oracle: Oracle) -> Synset:
        """
        Given an oracle, return the synset that the oracle has as its target.
        
        Keyword Arguments:
        oracle -- The oracle being searched
        """

        # Feel free to change the code within
        # --------------------------------------

        l, r = 0, len(self.all_synsets)
        while r - l > 1:
            m = int((l + r)/2)
            ql = self.all_synsets[l : m]
            qr = self.all_synsets[m : r]
            if oracle.there_exists('hyponyms', list(ql)):
                r = m
            elif oracle.there_exists('hyponyms', list(qr)):
                l = m
            else:
                break

        for syn in self.all_synsets[l : r]:
            for hyp in syn.hypernyms():
                if oracle.check(hyp):
                    return hyp

        # print('ran {} iterations'.format(i))
        # ---------------------------------------
        assert any(self._searched.values()), "Searched all of WN without finding it!"

        found = [x for x in self._searched if self._searched[x]][0]

        return found

if __name__ == "__main__":
    search_lst = ['dog.n.01', 'corgi.n.01', 'pug.n.01', 'dalmatian.n.02', 'man.n.01',
                      'anatidae.n.01', 'pack.n.06', 'flag.n.07', 'president_of_the_united_states.n.01']
    
    for ele in search_lst:
        oracle = Oracle(wn.synset(ele))

        searcher = Searcher()
        print("Search result is:")
        print(searcher(oracle))
        print("Took %i steps to get there" % oracle.num_queries())
