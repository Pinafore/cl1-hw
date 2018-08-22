import unittest
from limerick import LimerickDetector

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.ld = LimerickDetector()

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

        a = """
a woman whose friends called a prude
on a lark when bathing all nude
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

        d = """There was a young man named Wyatt
whose voice was exceedingly quiet
And then one day
it faded away"""

        e = """An exceedingly fat friend of mine,
When asked at what hour he'd dine,
Replied, "At eleven,     
At three, five, and seven,
And eight and a quarter past nine"""

        f = """A limerick fan from Australia
regarded his work as a failure:
his verses were fine
until the fourth line"""

        g = """There was a young lady one fall
Who wore a newspaper dress to a ball.
The dress caught fire
And burned her entire
Front page, sporting section and all."""

        h = "dog\ndog\ndog\ndog\ndog"

        self.assertEqual(self.ld.is_limerick(a), True)
        self.assertEqual(self.ld.is_limerick(b), False)
        self.assertEqual(self.ld.is_limerick(c), False)
        self.assertEqual(self.ld.is_limerick(d), False)
        self.assertEqual(self.ld.is_limerick(f), False)
        self.assertEqual(self.ld.is_limerick(e), True)
        self.assertEqual(self.ld.is_limerick(g), True)
        
if __name__ == '__main__':
    unittest.main()
