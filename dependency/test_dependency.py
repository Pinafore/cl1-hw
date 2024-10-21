kCORRECT = """economic\tATT\t2
news\tSBJ\t3
had\tPRED\t0
little\tATT\t5
effect\tOBJ\t3
on\tatt\t5
financial\tATT\t8
markets\tPC\t6
.\tPU\t3"""

kRIGHT = """economic\tATT\t0
news\tSBJ\t1
had\tPRED\t2
little\tATT\t3
effect\tOBJ\t4
on\tatt\t5
financial\tATT\t6
markets\tPC\t7
.\tPU\t8"""


import unittest
from dependency import Transition, transition_sequence, parse_from_transition, kROOT, sentence_attachment_accuracy

import nltk


class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.sent = nltk.parse.dependencygraph.DependencyGraph(kCORRECT)
        self.words = [kROOT] + [x.split('\t')[0] for x in kCORRECT.split('\n')]
        self.right_actions = [Transition('s')] * len(self.words) + [Transition('r')] * len(self.words)        
        self.right_branch = nltk.parse.dependencygraph.DependencyGraph(kRIGHT)
        self.sequence = []
    
    def test_sequence(self):
        sequence = [x.type for x in transition_sequence(self.sent)]
        self.assertEqual(sequence, self.sequence)

    def test_true_reconstruct(self):
        sentence = parse_from_transition(self.words, self.sequence)
        self.assertEqual(sentence, self.sent)

    def test_right_reconstruct(self):
        right_branch = parse_from_transition(self.words, self.right_actions)

        self.assertEqual(right_branch, self.right_branch)
        
    def test_accuracy_ec(self):
        correct_accuracy = sentence_attachment_accuracy(self.sequence, self.sequence)
        self.assertEqual(len(self.sequence), correct_accuracy)

        right_branch_accuracy = sentence_attachment_accuracy(self.sequence, self.right_actions)

        correct_parents = [int(x.split('\t')[-1]) for x in kCORRECT.split('\n')]
        right_parents = [int(x.split('\t')[-1]) for x in kRIGHT.split('\n')]
        self.assertEqual(sum(1 for x, y in zip(correct_parents, right_parents) if x==1),
                         right_branch_accuracy)

if __name__ == '__main__':
    unittest.main()
