import unittest
from math import log
import nltk

from oracle import transition_sequence
from treebank import PcfgEstimator

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        sample_sentence = """economic\tATT\t2
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

        self.sent = "the cat sat on a mat".split()
        self.tags = "D N V P D N".split()

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
