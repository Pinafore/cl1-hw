import unittest

from scipy.stats import poisson

from dependency import BigramInterpScoreFunction, EisnerParser, kNEG_INF, kROOT


class StubScoreFunction(BigramInterpScoreFunction):

    def __init__(self):
        self._word_scores = {}
        self._pos_scores = {}
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

if __name__ == '__main__':
    unittest.main()
