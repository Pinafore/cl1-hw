# A file to test that your implementation is working correctly.
#
# CL2 HW1
# Jordan Boyd-Graber
#
# Feel free to modify and extend this file to more thoroughly test your
# implementation.
#
# Note that the tests here assume a particular translation direction (e -> f),
# so if you're going in the other direction, you will need to tweak the tests.
#
# USAGE: python ibm_trans_test.py

from string import lower
from math import log

import unittest

from ibm_trans import Translation, ModelOne, BiCorpus, UniformTranslation


class TestTranslation(unittest.TestCase):
    """
    Make sure that new translations don't have counts and that when you add
    counts to a model, it stays there.
    """

    def new_translation(self):
        return Translation()

    def test_empty(self):
        t = self.new_translation()
        self.assertEqual(t.get_count("dog", "dog"), 0)

    def test_increment(self):
        t = self.new_translation()
        t.collect_count(.75, "dog", "hund")
        self.assertEqual(t.get_count("dog", "hund"), .75)

        self.assertEqual(t.score("dog", "hund"), 1.0)

        t.collect_count(.25, "dog", "klein")

        # Use this test if you're doing p(f|e)
        self.assertEqual(t.score("dog", "klein"), .25)


class TrivialCorpus(BiCorpus):
    """
    A non-sensical corpus for unit tests.
    """

    def raw_sentences(self):
        data = [("the house is small",
                 "das haus ist ja klein")]

        for ii, jj in [map(lower, x) for x in data]:
            yield ii.lower().split(), jj.lower().split()


class TestModelOne(unittest.TestCase):
    """
    Compute translation probabilities from counts.
    """

    def test_accumulate(self):
        mo = ModelOne()
        trans = mo.accumulate_counts(TrivialCorpus(), UniformTranslation())

        # Use this test if you're doing p(f|e)

        self.assertAlmostEqual(trans.score("the", "das"), .2)
        self.assertAlmostEqual(trans.score("the", "haus"), .2)
        self.assertAlmostEqual(trans.score("house", "haus"), .2)
        self.assertAlmostEqual(trans.score(None, "haus"), .2)

    def test_score(self):
        mo = ModelOne()
        mo.em(TrivialCorpus(), 1)

        # After one iteration of EM, all the translation scores are .2
        # (as above).  So the overall translation probability is
        #
        # p(f|e) = 1 / (l_e + 1) ^ l_f *
        #               \prod_j^l_f \sum_i^l_e t(f_j | e_i)
        #
        #        = 1 / (4 + 1) ^ 4 (5 * .2) ^ 4
        #        = 1 / 625
        #
        # There are 5 English words counting the "NULL"

        mo.build_lm([], 0)
        self.assertAlmostEqual(\
            mo.translate_score("the house is small".split(), \
                                   "das haus ist klein".split()), \
                -1.0 * log(625., 2))



if __name__ == '__main__':
    unittest.main()
