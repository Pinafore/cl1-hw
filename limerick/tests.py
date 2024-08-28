import unittest
import json
from limerick import LimerickDetector

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.ld = LimerickDetector()

        with open("sample_limericks.json") as infile:
            self.limerick_tests = json.load(infile)

    def test_rhyme(self):
        self.assertEqual(self.ld.rhymes("dog", "bog"), True)
        self.assertEqual(self.ld.rhymes("eleven", "seven"), True)
        self.assertEqual(self.ld.rhymes("nine", "wine"), True)
        self.assertEqual(self.ld.rhymes("dine", "fine"), True)
        self.assertEqual(self.ld.rhymes("wine", "mine"), True)
        self.assertEqual(self.ld.rhymes("dock", "sock"), True)
        self.assertEqual(self.ld.rhymes("weigh", "fey"), True)
        self.assertEqual(self.ld.rhymes("tree", "debris"), True)
        self.assertEqual(self.ld.rhymes("niece", "peace"), True)
        self.assertEqual(self.ld.rhymes("read", "need"), True)

        self.assertEqual(self.ld.rhymes("dog", "cat"), False)
        self.assertEqual(self.ld.rhymes("bagel", "sail"), False)
        self.assertEqual(self.ld.rhymes("wine", "rind"), False)
        self.assertEqual(self.ld.rhymes("failure", "savior"), False)
        self.assertEqual(self.ld.rhymes("cup", "duck"), False)

    def test_syllables(self):
        self.assertEqual(self.ld.num_syllables("dog"), 1)
        self.assertEqual(self.ld.num_syllables("asdf"), 1)
        self.assertEqual(self.ld.num_syllables("letter"), 2)
        self.assertEqual(self.ld.num_syllables("washington"), 3)
        self.assertEqual(self.ld.num_syllables("dock"), 1)
        self.assertEqual(self.ld.num_syllables("dangle"), 2)
        self.assertEqual(self.ld.num_syllables("thrive"), 1)
        self.assertEqual(self.ld.num_syllables("fly"), 1)
        self.assertEqual(self.ld.num_syllables("placate"), 2)
        self.assertEqual(self.ld.num_syllables("renege"), 2)
        self.assertEqual(self.ld.num_syllables("reluctant"), 3)

    def test_examples(self):
        self.setUp()
        a,b,c,d,e,f,g = ['\n'.join(test['lines']) for test in self.limerick_tests]
        self.assertEqual(self.ld.is_limerick(a), True)
        self.assertEqual(self.ld.is_limerick(b), False)
        self.assertEqual(self.ld.is_limerick(c), False)
        self.assertEqual(self.ld.is_limerick(d), False)
        self.assertEqual(self.ld.is_limerick(f), False)
        self.assertEqual(self.ld.is_limerick(e), True)
        self.assertEqual(self.ld.is_limerick(g), True)

if __name__ == '__main__':
    unittest.main()
