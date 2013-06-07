import unittest
from limerick import LimerickDetector

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.ld = LimerickDetector()

    def test_rhyme(self):
        self.assertTrue(self.ld.rhymes("dog", "bog"))
        self.assertFalse(self.ld.rhymes("dog", "cat"))

    def test_syllables(self):
        self.assertEqual(self.ld.num_syllables("dog"), 1)
        self.assertEqual(self.ld.num_syllables("asdf"), 1)
        self.assertEqual(self.ld.num_syllables("letter"), 2)
        self.assertEqual(self.ld.num_syllables("washington"), 3)

    def test_examples(self):

        a = """
a bather whose clothing was strewed
by winds that left her quite nude
saw a man come along
and unless we are wrong
you expected this line to be lewd
        """

        b = """while it's true all i've done is delay
in defense of myself i must say
today's payoff is great
while the workers all wait
"""

        c = """
this thing is supposed to rhyme
but I simply don't got the time
who cares if i miss,
nobody will read this
i'll end this here poem potato
"""

        self.assertTrue(self.ld.is_limerick(a))
        self.assertFalse(self.ld.is_limerick(b))
        self.assertFalse(self.ld.is_limerick(c))

if __name__ == '__main__':
    unittest.main()
