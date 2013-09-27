import unittest
from math import log

from language_model import BigramLanguageModel, kLM_ORDER, kUNK_CUTOFF

kNEG_INF = float("-inf")


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.lm = BigramLanguageModel(kUNK_CUTOFF)

    def test_vocab(self):
        self.lm.train_seen("a", 300)

        self.lm.train_seen("b")
        self.lm.train_seen("c")
        self.lm.finalize()

        # Infrequent words should look the same
        self.assertEqual(self.lm.vocab_lookup("b"),
                         self.lm.vocab_lookup("c"))

        # Infrequent words should look the same as never seen words
        self.assertEqual(self.lm.vocab_lookup("b"),
                         self.lm.vocab_lookup("d"),
                         "")

        # The frequent word should be different from the infrequent word
        self.assertNotEqual(self.lm.vocab_lookup("a"),
                            self.lm.vocab_lookup("b"))

    def test_censor(self):
        self.lm.train_seen("a", 300)

        self.lm.train_seen("b")
        self.lm.train_seen("c")
        self.lm.finalize()

        censored_a = list(self.lm.censor(["a", "b", "d"]))
        censored_b = list(self.lm.censor(["d", "b", "a"]))
        censored_c = list(self.lm.censor(["a", "d", "b"]))

        self.assertEqual(censored_a, censored_c)

        # Should add start and end tag
        self.assertEqual(len(censored_a), 5)
        self.assertEqual(censored_a[0], censored_b[0])
        self.assertEqual(censored_a[-1], censored_b[-1])
        self.assertEqual(censored_a[1], censored_b[3])
        self.assertEqual(censored_a[2], censored_b[2])
        self.assertEqual(censored_a[3], censored_b[3])

    def test_lm(self):
        self.lm.train_seen("a", 300)
        self.lm.finalize()

        self.lm.add_train(["a", "a", "b"])

        # Test MLE
        self.assertEqual(self.lm.mle("<s>", "b"), kNEG_INF)
        self.assertEqual(self.lm.mle("<s>", "a"), log(1.0))
        self.assertEqual(self.lm.mle("a", "a"), log(0.5))
        self.assertEqual(self.lm.mle("a", "b"), log(0.5))
        self.assertEqual(self.lm.mle("a", "c"), log(0.5))

        # Test Add one
        self.assertEqual(self.lm.laplace("<s>", "b"), log(1.0 / 5.0))
        self.assertEqual(self.lm.laplace("<s>", "a"), log(2.0 / 5.0))
        self.assertEqual(self.lm.laplace("a", "a"), log(2.0 / 6.0))
        self.assertEqual(self.lm.laplace("a", "b"), log(2.0 / 6.0))
        self.assertEqual(self.lm.laplace("a", "c"), log(2.0 / 6.0))

if __name__ == '__main__':
    unittest.main()
