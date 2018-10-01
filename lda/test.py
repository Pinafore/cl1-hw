
import unittest

from lda import Sampler

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self._vocab = "dog cat pig hamburger iron".split()
        self._lda = Sampler(3, self._vocab, alpha=1.0, beta=1.0)

        # Unassign all of the topics
        self._lda.add_doc("dog cat cat pig".split(), self._vocab, 0)
        self._lda.add_doc("hamburger dog hamburger asdf".split(), self._vocab, 1)
        self._lda.add_doc("iron iron pig iron".split(), self._vocab, 2)
                
        doc_len = {0:4, 1:3, 2:4}
        for dd in doc_len:
            for ii in range(doc_len[dd]):
                self._lda.change_topic(dd, ii, -1)

        # Set up the sample probabilities like in the worksheet
        self._lda.change_topic(0, 0, 0)
        self._lda.change_topic(0, 1, 1)
        self._lda.change_topic(0, 2, 2)                
        self._lda.change_topic(0, 3, 0)
                          
        self._lda.change_topic(1, 0, 1)
        self._lda.change_topic(1, 1, 2)
        self._lda.change_topic(1, 2, 0)                

        self._lda.change_topic(2, 0, 0)
        self._lda.change_topic(2, 1, 2)
        self._lda.change_topic(2, 2, 1)                
        self._lda.change_topic(2, 3, 1)                                

        self._lda.report_topics(self._vocab, "test_init")        
    
    def testConditionals(self):

        # Check the probabilities
        self._lda.change_topic(0, 0, -1)
        sample_probs = self._lda.sample_probs(0, 0)
        self.assertAlmostEqual(sample_probs[0], 0.041667, 5)
        self.assertAlmostEqual(sample_probs[1], 0.037037, 5)
        self.assertAlmostEqual(sample_probs[2], 0.083333, 5)
        self._lda.change_topic(0, 0, 2)

        self._lda.change_topic(0, 1, -1)
        sample_probs = self._lda.sample_probs(0, 1)
        self.assertAlmostEqual(sample_probs[0], 0.041667, 5)
        self.assertAlmostEqual(sample_probs[1], 0.020833, 5)
        self.assertAlmostEqual(sample_probs[2], 0.111111, 5)
        self._lda.change_topic(0, 1, 2)
        
        self._lda.change_topic(0, 2, -1)
        sample_probs = self._lda.sample_probs(0, 2)
        self.assertAlmostEqual(sample_probs[0], 0.041667, 5)
        self.assertAlmostEqual(sample_probs[1], 0.020833, 5)
        self.assertAlmostEqual(sample_probs[2], 0.111111, 5)
        self._lda.change_topic(0, 2, 2)

        self._lda.change_topic(0, 3, -1)
        sample_probs = self._lda.sample_probs(0, 3)
        self.assertAlmostEqual(sample_probs[0], 0.023810, 5)
        self.assertAlmostEqual(sample_probs[1], 0.041667, 5)
        self.assertAlmostEqual(sample_probs[2], 0.066667, 5)
        self._lda.change_topic(0, 3, 2)

        self._lda.change_topic(1, 0, -1)
        sample_probs = self._lda.sample_probs(1, 0)
        self.assertAlmostEqual(sample_probs[0], 0.114286, 5)
        self.assertAlmostEqual(sample_probs[1], 0.028571, 5)
        self.assertAlmostEqual(sample_probs[2], 0.036364, 5)
        self._lda.change_topic(1, 0, 0)        

        self._lda.change_topic(1, 1, -1)
        sample_probs = self._lda.sample_probs(1, 1)
        self.assertAlmostEqual(sample_probs[0], 0.075000, 5)
        self.assertAlmostEqual(sample_probs[1], 0.028571, 5)
        self.assertAlmostEqual(sample_probs[2], 0.040000, 5)
        self._lda.change_topic(1, 1, 0)        

                        
if __name__ == '__main__':
    unittest.main()        
