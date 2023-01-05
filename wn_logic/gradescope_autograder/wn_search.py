
import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.wordnet import Lemma
from wn_eval import Oracle
from typing import Union



def offsets(self):
    return [self.offset()]
Synset.offsets = offsets


class Searcher:

    # def word_offset(self):
    #     return [self.word_offset()]
    # Synset.offsets = word_offset
    """
    Class to search WordNet through logical queries.
    """
    
    def __init__(self):
        
        # Feel free to add your own data members
        self._searched = {}

    def check(self, oracle: Oracle, candidate: Synset) -> bool:
        """
        Convenience method to check whether two synsets are the same
        and storing the result.
        
        Keyword Arguments:
        oracle -- The oracle that can check whether the candidate matches
        candidate -- The synset to check
        """
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

        # Start at the top, go breadth first
        _flag = False
        d = ""
        synsets = list(wn.all_synsets(pos=wn.NOUN))
        offsets = [s.offset() for s in synsets]
        min_idx = 0
        max_idx = len(offsets) - 1
        mid_point = 0
        while max_idx >= min_idx:
            mid_point = ((max_idx - min_idx) // 2)
            oracle_res = oracle.cnf_eval([['offsets' for x in range(len(offsets))]],
                                [offsets[min_idx:min_idx+mid_point]])
            candidate = wn.synset_from_pos_and_offset('n', offsets[min_idx+mid_point])
            if oracle_res:
                max_idx = max_idx - mid_point - 1
            else:
                min_idx = min_idx + mid_point

            if min_idx == max_idx:
                break

            if mid_point == 0:
                # if oracle.check(candidate):
                # if self.check(oracle, candidate=candidate):
                    return(synsets[min_idx])

# if __name__ == "__main__":
#     oracle = Oracle(wn.synset('corgi.n.01'))

#     searcher = Searcher()
#     print("Search result is:")
#     print(searcher(oracle))
#     print("Took %i steps to get there" % oracle.num_queries())

if __name__ == "__main__":
    search_lst = ['dog.n.01', 'corgi.n.01', 'pug.n.01', 'dalmatian.n.02', 'man.n.01',
                      'anatidae.n.01', 'pack.n.06', 'flag.n.07', 'president_of_the_united_states.n.01']
    
    for ele in search_lst:
        oracle = Oracle(wn.synset(ele))

        searcher = Searcher()
        print("Search result is:")
        print(searcher(oracle))
        print("Took %i steps to get there" % oracle.num_queries())

