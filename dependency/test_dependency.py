kCORRECT = """economic\tATT\t2
news\tSBJ\t3
had\tPRED\t0
little\tATT\t5
effect\tOBJ\t3
on\tatt\t5
financial\tATT\t8
markets\tPC\t6
.\tPU\t3"""

kTOY = """the\tDET\t2
cat\tSBJ\t3
ate\tPRED\t0
lasagna\tOBJ\t3"""

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
from dependency import Transition, transition_sequence, parse_from_transition, kROOT, sentence_attachment_accuracy, flatten, ShiftReduceState

import nltk


class TestStringMethods(unittest.TestCase):

    def compare_trees(self, reference, test):
        self.assertEqual(reference.nodes, test.nodes)
        
    def setUp(self):
        self.toy_sent = nltk.parse.dependencygraph.DependencyGraph(kTOY)
        self.sent = nltk.parse.dependencygraph.DependencyGraph(kCORRECT)
        self.words = [(kROOT, 'TOP')] + [(x.split('\t')[0], x.split('\t')[1]) for x in kCORRECT.split('\n')]
        self.toy_words = [(kROOT, 'TOP')] + [(x.split('\t')[0], x.split('\t')[1]) for x in kTOY.split('\n')]
        self.right_actions = [Transition('s')] * (len(self.words) - 2) + [Transition('r')] * (len(self.words) - 1) + [Transition('s')]
        self.right_branch = nltk.parse.dependencygraph.DependencyGraph(kRIGHT)
        self.toy_sequence = [Transition('s', None), Transition('l', (2, 1)), Transition('s', None), Transition('l', (3, 2)), Transition('s', None), Transition('r', (3, 4)), Transition('r', (0, 3)), Transition('s', None)]
        self.sequence = [Transition('s', None), Transition('l', (2, 1)), Transition('s', None), Transition('l', (3, 2)), Transition('s', None), Transition('s', None), Transition('l', (5, 4)), Transition('s', None), Transition('s', None), Transition('s', None), Transition('l', (8, 7)), Transition('r', (6, 8)), Transition('r', (5, 6)), Transition('r', (3, 5)), Transition('s', None), Transition('r', (3, 9)), Transition('r', (0, 3)), Transition('s', None)]
    
    def test_toy_state(self):

        state = ShiftReduceState([x[0] for x in self.toy_words], [x[1] for x in self.toy_words])

        state_checks = [{"stack": [0], "buffer": [4, 3, 2, 1], "edges": []},                     #1: shift
                        {"stack": [0, 1], "buffer": [4, 3, 2], "edges": [(2,1)]},                #2: left
                        {"stack": [0], "buffer": [4, 3, 2], "edges": [(2,1)]},                   #3: shift
                        {"stack": [0, 2], "buffer": [4, 3], "edges": [(2,1), (3,2)]},            #4: left
                        {"stack": [0], "buffer": [4, 3], "edges": [(2,1), (3,2)]},               #5: shift
                        {"stack": [0, 3], "buffer": [4], "edges": [(2,1), (3,2), (3,4)]},        #6: right
                        {"stack": [0], "buffer": [3], "edges": [(2,1), (3,2), (3,4), (0,3)]},    #7: right
                        {"stack": [], "buffer": [0], "edges": [(2,1), (3,2), (3,4), (0,3)]},     #8: shift
                        {"stack": [0], "buffer": [], "edges": [(2,1), (3,2), (3,4), (0,3)]}      #9: null
                        ] 
        
        step = 0
        for action, state_check in zip(self.toy_sequence, state_checks):
            prior_stack = state.stack[:]
            prior_buffer = state.buffer[:]
            self.assertEqual(prior_stack, state_check["stack"], f"Stack Step {step}")
            self.assertEqual(prior_buffer, state_check["buffer"], f"Buffer Step {step}")

            step += 1

            state.apply(action)
            self.assertEqual(state.edges, state_check["edges"], f"Edges Step {step}")

            if action.type == 's':
                self.assertEqual(prior_stack + [prior_buffer[-1]], state.stack, f"Stack after shift (step={step})")
                self.assertEqual(state.buffer + [state.stack[-1]], prior_buffer, f"Buffer after shift (step={step})")
            elif action.type == 'l':
                start, stop = state.edges[-1]
                self.assertEqual(state.stack + [stop], prior_stack, f"Stack after left arc (step={step})")
                self.assertEqual(state.buffer, prior_buffer, f"Buffer after left arc (step={step})")
                self.assertEqual(state.buffer[-1], start, f"Buffer after left arc (step={step})")
                self.assertEqual(action.edge, state.edges[-1], f"Edge after left arc (step={step})")
            elif action.type == 'r':
                start, stop = state.edges[-1]
                self.assertEqual(action.edge, state.edges[-1], f"Edge after right arc (step={step})")
                self.assertEqual(state.stack + [start], prior_stack, f"Stack after right arc (step={step})")
                self.assertEqual(state.buffer[:-1], prior_buffer[:-1], f"Buffer after right arc (step={step})")
                self.assertEqual(state.buffer[-1], start, f"Buffer after right arc (step={step})")
                self.assertEqual(prior_buffer[-1], stop, f"Buffer before right arc (step={step})")


    def test_toy_sequence(self):
        sequence = transition_sequence(nltk.parse.dependencygraph.DependencyGraph(kTOY))
        self.assertEqual([x.type for x in sequence], [x.type for x in self.toy_sequence])

    def test_right_sequence(self):
        sequence = transition_sequence(nltk.parse.dependencygraph.DependencyGraph(kRIGHT))
        self.assertEqual([x.type for x in sequence], [x.type for x in self.right_actions])

    # test that the transition sequence is correct
    def test_correct_sequence(self):
        sequence = transition_sequence(self.sent)
        self.assertEqual([x.type for x in sequence], [x.type for x in self.sequence])

    def test_toy_reconstruct(self):
        sentence = parse_from_transition(self.toy_words, self.toy_sequence)
        self.compare_trees(sentence, self.toy_sent)

    # test that the right branch is correct
    def test_right_sequence_reconstruct(self):
        right_branch = parse_from_transition(self.words, self.right_actions)
        self.compare_trees(right_branch, self.right_branch)

    def test_reconstruct(self):
        sentence = parse_from_transition(self.words, self.sequence)
        self.compare_trees(sentence, self.sent)
        
    def test_accuracy_ec(self):
        correct_accuracy = sentence_attachment_accuracy(self.sent, self.sent)
        self.assertEqual(len(self.words) - 1, correct_accuracy)

        right_branch_accuracy = sentence_attachment_accuracy(self.sent, self.right_branch)

        correct_parents = [int(x.split('\t')[-1]) for x in kCORRECT.split('\n')]
        right_parents = [int(x.split('\t')[-1]) for x in kRIGHT.split('\n')]
        self.assertEqual(sum(1 for x, y in zip(correct_parents, right_parents) if x==y),
                         right_branch_accuracy)

if __name__ == '__main__':
    unittest.main()
