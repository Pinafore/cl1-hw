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



# class TestLeaderboard(unittest.TestCase):
#     name = ['3.5', '4.6', '7.8', '15.5']

#     def setUp(self):
#         dev_file = 'dev_results.txt'
#         search_lst = ['dog.n.01', 'corgi.n.01', 'pug.n.01', 'dalmatian.n.02', 'man.n.01',
#                       'anatidae.n.01', 'pack.n.06', 'flag.n.07', 'president_of_the_united_states.n.01']
#         self.res_lst = []
      
#         for ele in search_lst:
#             searcher = Searcher()
#             oracle = Oracle(wn.synset(ele))
#             searcher(oracle)
#             self.res_lst.append(oracle.num_queries())
        
        
#     @weight(2)
#     @visibility('visible')
#     @number("1.0.1")
#     @leaderboard("dev prec/rec")
#     def test_leaderboard_6_dev_prec_rec(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""
       
#         # res_lst = []
#         sum = 0
#         wn.synset('dog.n.01')
    
#         # result_str = '{}'.format(min(res_lst))
#         self.assertTrue(min(self.res_lst) < 100000)

#         result_str = '{}'.format(3+5+8)
       
#         set_leaderboard_value(result_str)


#     @weight(2)
#     @visibility('visible')
#     @number("1.0.2")
#     @leaderboard("test prec/rec")
#     def test_leaderboard_3_test_prec_rec(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""

#         # overall, _ = conlleval.metrics(self.test_counts)

#         # result_str = '%.1f/%.1f' % (100*overall.prec,
#         #                             100*overall.rec)

#         # set_leaderboard_value(result_str)
#         # result_str = '4.5'
#         # result_str = '%.1f' % (min(self.res_lst))
#         # set_leaderboard_value(result_str)
#         self.assertTrue(max(self.res_lst) > 10)
#         set_leaderboard_value(PUT[1])


#     @weight(2)
#     @visibility('visible')
#     @number("1.0.3")
#     @leaderboard("dev f1")
#     def test_leaderboard_4_dev_f1(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""
#         mean = sum(self.res_lst) / len(self.res_lst)
#         res = sum((i - mean) ** 2 for i in self.res_lst) / len(self.res_lst)
        

#         self.assertTrue(res < 1000000)
#         set_leaderboard_value(PUT[2])


#     @weight(2)
#     @visibility('visible')
#     @number("1.0.4")
#     @leaderboard("test f1")
#     def test_leaderboard_1_test_f1(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""

#         self.assertTrue(max(self.res_lst) > 10)
#         # set_leaderboard_value(result_str)
#         set_leaderboard_value(PUT[3])


#     @weight(4)
#     @visibility('visible')
#     @number("2.0.1")
#     @leaderboard("dev accuracy")
#     def test_leaderboard_5_dev_acc(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""

#         # counts = self.dev_counts
#         # if counts.token_counter > 0:
#         #     acc = counts.correct_tags / counts.token_counter
#         # else:
#         #     acc = 0

#         # set_leaderboard_value('%.1f' % (100 * acc))
#         self.assertTrue(max(self.res_lst) > 10)
#         result_str = '7.5'
#         set_leaderboard_value(result_str)
        
#     @weight(4)
#     @visibility('visible')
#     @number("2.0.3")
#     @leaderboard("test accuracy")
#     def test_leaderboard_2_test_acc(self, set_leaderboard_value=None):
#         """Sets a leaderboard value"""

#         # counts = self.test_counts
#         # if counts.token_counter > 0:
#         #     acc = counts.correct_tags / counts.token_counter
#         # else:
#         #     acc = 0

#         # set_leaderboard_value('%.1f' % (100 * acc))
#         self.assertTrue(max(self.res_lst) > 10)
#         result_str = '8.5'
#         set_leaderboard_value(result_str)

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


    
    @leaderboard("mean iterations")
    def test_forward1(self, set_leaderboard_value=None):
        # self.assertTrue(min(self.res_lst) < 100000)

        # result_str = '{}'.format(sum(self.res_lst)/len(self.res_lst))
        result = sum(self.res_lst)/len(self.res_lst)

        print('mean {}'.format(result))
        set_leaderboard_value(result)

   
    @leaderboard("max iterations")
    def test_forward2(self, set_leaderboard_value=None):
        # self.assertTrue(max(self.res_lst) > 10)

        # result_str = '{}'.format(max(self.res_lst))
        result = max(self.res_lst)
        print('max {}'.format(result))
        set_leaderboard_value(result)

    @leaderboard("min iterations")
    def test_forward3(self, set_leaderboard_value=None):
        # self.assertTrue(max(self.res_lst) > 10)

        # result_str = '{}'.format(min(self.res_lst))
        result = min(self.res_lst)
        print('min {}'.format(result))
        set_leaderboard_value(result)
    

    @leaderboard("main iterations")
    def test_forward4(self, set_leaderboard_value=None):
        # self.assertTrue(max(self.res_lst) > 10)

        # result_str = '{}'.format(self.res_lst[0])
        result = self.res_lst[0]
        print('main {}'.format(result))
        set_leaderboard_value(result)


if __name__ == '__main__':
    unittest.main()
