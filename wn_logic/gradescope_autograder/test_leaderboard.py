# -*- coding: utf-8 -*-

from unicodedata import name
import unittest
import random
from gradescope_utils.autograder_utils.decorators import leaderboard, weight, visibility, number
from wn_search import Searcher
from wn_eval import Oracle
# from wn_search import Oracle
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset



search_lst = ['dog.n.01', 'corgi.n.01', 'pug.n.01', 'dalmatian.n.02', 'man.n.01',
                      'anatidae.n.01', 'pack.n.06', 'flag.n.07', 'president_of_the_united_states.n.01']

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.res_lst = []
      
        for ele in search_lst:
            searcher = Searcher()
            oracle = Oracle(wn.synset(ele))
            searcher(oracle)
            self.res_lst.append(oracle.num_queries())


    
    # @leaderboard("mean iterations")
    # def test_forward1(self, set_leaderboard_value=None):
    #     # self.assertTrue(min(self.res_lst) < 100000)

    #     # result_str = '{}'.format(sum(self.res_lst)/len(self.res_lst))
    #     result = sum(self.res_lst)/len(self.res_lst)
    #     print('Average iteration {}'.format(result))
    #     set_leaderboard_value(result)

   
    # @leaderboard("max iterations")
    # def test_forward2(self, set_leaderboard_value=None):
    #     # self.assertTrue(max(self.res_lst) > 10)

    #     # result_str = '{}'.format(max(self.res_lst))
    #     result = max(self.res_lst)
    #     print('max iteration is {}'.format(result))
    #     set_leaderboard_value(result)

    # @leaderboard("min iterations")
    # def test_forward3(self, set_leaderboard_value=None):
    #     # self.assertTrue(max(self.res_lst) > 10)

    #     # result_str = '{}'.format(min(self.res_lst))
    #     result = min(self.res_lst)
    #     print('min ieration is {}'.format(result))
    #     set_leaderboard_value(result)
    

    # @leaderboard("main iterations")
    # def test_forward4(self, set_leaderboard_value=None):
    #     # self.assertTrue(max(self.res_lst) > 10)

    #     # result_str = '{}'.format(self.res_lst[0])
    #     result = self.res_lst[0]
    #     print('dog iteration is {}'.format(result))
    #     set_leaderboard_value(result)

    @weight(40)
    @visibility('visible')
    @number("1.0.1")
    def test_num_queries(self):
        # self.assertEqual(self.chart.get_score(3, 3, False, True), 0.0)
        self.assertTrue(self.res_lst[0] < 100)

  

if __name__ == '__main__':
    unittest.main()


# yhsu1235@umd.edu