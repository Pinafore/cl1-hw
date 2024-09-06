import unittest
from math import log

from bigram_lm import BigramLanguageModel, kLM_ORDER, \
    kUNK_CUTOFF, kNEG_INF, kSTART, kEND, lg


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=0.6, \
                                      kn_discount = 0.1,
                                      kn_concentration = 1.0,
                                      dirichlet_alpha=0.1)

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


        censored_a = list(self.lm.censor(['a', 'b', 'd']))
        censored_b = list(self.lm.censor(['d', 'b', 'a']))
        censored_c = list(self.lm.censor(['a', 'b', 'd']))
        censored_d = list(self.lm.censor(['b', 'd', 'a']))

        self.assertEqual(censored_a, censored_c)
        self.assertEqual(censored_b, censored_d)

        # Should add start and end tag
        print(censored_a)
        self.assertEqual(len(censored_a), 5)
        self.assertEqual(censored_a[0], censored_b[0])
        self.assertEqual(censored_a[-1], censored_b[-1])
        self.assertEqual(censored_a[1], censored_b[3])
        self.assertEqual(censored_a[2], censored_b[1])

    def test_lm(self):
        self.lm.train_seen("a", 300)
        self.lm.finalize()


        self.lm.add_train(['a', 'a', 'b'])

        # Test MLE
        word_start = self.lm.vocab_lookup(kSTART)
        word_end = self.lm.vocab_lookup(kEND)
        word_a = self.lm.vocab_lookup("a")
        word_b = self.lm.vocab_lookup("b")
        word_c = self.lm.vocab_lookup("c")

        self.assertAlmostEqual(self.lm.mle(word_start, word_b), kNEG_INF)
        self.assertAlmostEqual(self.lm.mle(word_start, word_a), lg(1.0))
        self.assertAlmostEqual(self.lm.mle(word_a, word_a), lg(0.5))
        self.assertAlmostEqual(self.lm.mle(word_a, word_b), lg(0.5))
        self.assertAlmostEqual(self.lm.mle(word_a, word_c), lg(0.5))

        # Test Add one
        self.assertAlmostEqual(self.lm.laplace(word_start, word_b),
                               lg(1.0 / 5.0))
        self.assertAlmostEqual(self.lm.laplace(word_start, word_a),
                               lg(2.0 / 5.0))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_a),
                               lg(2.0 / 6.0))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_b),
                               lg(2.0 / 6.0))
        self.assertAlmostEqual(self.lm.laplace(word_a, word_c),
                               lg(2.0 / 6.0))

        # Test Dirichlet
        self.assertAlmostEqual(self.lm.dirichlet(word_start, word_b),
                               lg(0.1 / 1.4))
        self.assertAlmostEqual(self.lm.dirichlet(word_start, word_a),
                               lg(1.1 / 1.4))
        self.assertAlmostEqual(self.lm.dirichlet(word_a, word_a),
                               lg(1.1 / 2.4))
        self.assertAlmostEqual(self.lm.dirichlet(word_a, word_b),
                               lg(1.1 / 2.4))
        self.assertAlmostEqual(self.lm.dirichlet(word_a, word_c),
                               lg(1.1 / 2.4))

        # Test Kneser-Ney
        self.assertAlmostEqual(self.lm.kneser_ney(word_start, word_a),
                               lg(0.69475))
        self.assertAlmostEqual(self.lm.kneser_ney(word_start, word_b),
                               lg(0.13475))
        self.assertAlmostEqual(self.lm.kneser_ney(word_start, word_end),
                               lg(0.13475))

        # Test Jelinek Mercer
        self.assertAlmostEqual(self.lm.jelinek_mercer(word_start, word_end),
                               lg(0.1))
        self.assertAlmostEqual(self.lm.jelinek_mercer(word_start, word_a),
                               lg(0.8))
        
if __name__ == '__main__':
    unittest.main()
