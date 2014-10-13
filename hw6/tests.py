import unittest
from math import log
import nltk

from oracle import transition_sequence

from scipy.stats import poisson

from dependency import BigramInterpScoreFunction, EisnerParser, kNEG_INF, kROOT
from treebank import PcfgEstimator

class StubScoreFunction(BigramInterpScoreFunction):

    def __init__(self):
        self._word_scores = {}
        self._tag_scores = {}
        self._poisson = poisson(0.0001)

        for aa, bb, ss in [(kROOT, "sat", 10),
                           ("the", "cat", 2), ("cat", "the", 1),
                           ("sat", "on", 1), ("sat", "cat", 1),
                           ("on", "sat", 2), ("on", "mat", 1),
                           ("a", "the", 2), ("a", "on", 2),
                           ("a", "mat", 2), ("mat", "a", 1)]:
            self._word_scores[(aa, bb)] = ss

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        sample_sentence = sample_sentence = """economic\tATT\t2
news\tSBJ\t3
had\tPRED\t0
little\tATT\t5
effect\tOBJ\t3
on\tatt\t5
financial\tATT\t8
markets\tPC\t6
.\tPU\t3"""
        self.dep = nltk.parse.dependencygraph.DependencyGraph(sample_sentence)
        self.constit = nltk.corpus.treebank.parsed_sents('wsj_0001.mrg')[0]
        self.sf = StubScoreFunction()
        self.sent = "the cat sat on a mat".split()
        self.tags = "D N V P D N".split()

    def test_initialization(self):
        chart = EisnerParser(self.sent, self.tags, self.sf)

        chart.initialize_chart()

        self.assertEqual(chart.get_score(1, 1, False, False), 0.0)
        self.assertEqual(chart.get_score(2, 2, True, False), 0.0)
        self.assertEqual(chart.get_score(3, 3, False, True), 0.0)
        self.assertEqual(chart.get_score(4, 4, True, True), 0.0)
        self.assertEqual(chart.get_score(5, 5, True, True), 0.0)
        self.assertEqual(chart.get_score(6, 6, True, True), 0.0)
        self.assertEqual(chart.get_score(6, 6, False, True), 0.0)

    def test_single_spans(self):
        chart = EisnerParser(self.sent, self.tags, self.sf)

        chart.initialize_chart()
        chart.fill_chart()

        self.assertEqual(round(chart.get_score(5, 6, True, False)), 2.0)
        self.assertEqual(round(chart.get_score(5, 6, True, True)), 2.0)
        self.assertEqual(round(chart.get_score(5, 6, True, False)), 2.0)
        self.assertEqual(round(chart.get_score(5, 6, False, True)), 1.0)
        self.assertEqual(round(chart.get_score(5, 6, False, False)), 1.0)

        self.assertLess(chart.get_score(4, 5, True, False), 0.0)
        self.assertEqual(round(chart.get_score(4, 5, False, False)), 2.0)

    def test_big_spans(self):
        chart = EisnerParser(self.sent, self.tags, self.sf)

        chart.initialize_chart()
        chart.fill_chart()

        self.assertEqual(round(chart.get_score(3, 5, False, True)), 4.0)
        self.assertEqual(round(chart.get_score(3, 6, True, True)), 3.0)
        self.assertEqual(round(chart.get_score(1, 2, True, True)), 2.0)
        self.assertEqual(round(chart.get_score(1, 3, False, True)), 2.0)
        self.assertEqual(round(chart.get_score(3, 4, False, True)), 2.0)
        self.assertEqual(round(chart.get_score(4, 5, False, False)), 2.0)

        self.assertEqual(round(chart.get_score(1, 5, False, False)), 8.0)
        self.assertEqual(round(chart.get_score(1, 5, False, True)), 8.0)
        self.assertEqual(round(chart.get_score(0, 3, True, False)), 12.0)
        self.assertEqual(round(chart.get_score(0, 6, True, True)), 15.0)

        self.assertSetEqual(set(chart.reconstruct()), set([(0, 3), (3, 2),
                                                           (2, 1), (3, 4),
                                                           (4, 6), (6, 5)]))

    def test_oracle(self):
        transitions = list(transition_sequence(self.dep))

        self.assertEqual(transitions[0]._type, 's')
        self.assertEqual(transitions[1]._type, 'l')
        self.assertEqual(transitions[1]._edge, (2, 1))
        self.assertEqual(transitions[2]._type, 's')
        self.assertEqual(transitions[3]._type, 'l')
        self.assertEqual(transitions[3]._edge, (3, 2))        
        self.assertEqual(transitions[4]._type, 's')                                
        self.assertEqual(transitions[5]._type, 's')                                
        self.assertEqual(transitions[6]._type, 'l')
        self.assertEqual(transitions[6]._edge, (5, 4))        
        self.assertEqual(transitions[7]._type, 's')
        self.assertEqual(transitions[8]._type, 's')
        self.assertEqual(transitions[9]._type, 's')
        self.assertEqual(transitions[10]._type, 'l')
        self.assertEqual(transitions[10]._edge, (8, 7))                
        self.assertEqual(transitions[11]._type, 'r')
        self.assertEqual(transitions[11]._edge, (6, 8))                        
        self.assertEqual(transitions[12]._type, 'r')
        self.assertEqual(transitions[12]._edge, (5, 6))                                
        self.assertEqual(transitions[13]._type, 'r')
        self.assertEqual(transitions[13]._edge, (3, 5))                                        
        self.assertEqual(transitions[14]._type, 's')
        self.assertEqual(transitions[15]._type, 'r')
        self.assertEqual(transitions[15]._edge, (3, 9))                                                
        self.assertEqual(transitions[16]._type, 'r')
        self.assertEqual(transitions[16]._edge, (0, 3))                                                


    def test_pcfg(self):
        pe = PcfgEstimator()
        pe.add_sentence(self.constit)

        self.assertEqual(pe.query('NNP', "Pierre"), 1/3.)
        self.assertEqual(pe.query('NP', "DT NN"), 1/4.)
        self.assertEqual(pe.query('DT', 'a'), 1/2.)
        self.assertEqual(pe.query('NNP', "John"), 0.0)
        
if __name__ == '__main__':
    unittest.main()
