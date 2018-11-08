import unittest
from NER_Datasets import ToyDataset,tags,vocabulary, dataset_to_sents_and_tags
from Named_Entity_Tagger import TaggingPerceptron

from numpy import zeros, nonzero
def gen_sparse_vector(size, nnz_vals):
    f = zeros(size)
    for ix, val in nnz_vals.items():
        f[ix] = val
    return f


class TestNERFunctions(unittest.TestCase):

    def setUp(self):
        self.dataset = ToyDataset()
        self.sents, self.tags = dataset_to_sents_and_tags(self.dataset)

        #expected feature vectors
        self.expected_train_ftr_vectos = [[]]*5

        self.expected_train_ftr_vectos[0] = gen_sparse_vector(141,\
            {3:1,15: 1,21: 1,25: 4,46: 1,51: 1,58: 1,70: 1,80: 1,95: 1,105: \
            1,125: 1})
        
            
        self.expected_train_ftr_vectos[1] = gen_sparse_vector(141,\
            {3: 1.0,10: 1.0,21: 1.0,25: 4.0,32: 1.0,51: 1.0,58: 1.0,65: 1.0,\
            100: 1.0,110: 1.0,120: 1.0,130: 1.0})
        
        self.expected_train_ftr_vectos[2] = gen_sparse_vector(141,\
            {3: 1.0,9: 1.0,20: 1.0,21: 1.0,25: 4.0,42: 1.0,49: 1.0,51: 1.0,\
            58: 1.0,65: 1.0,75: 1.0,85: 1.0,90: 1.0,100: 1.0})

        self.expected_train_ftr_vectos[3] = gen_sparse_vector(141,\
            {9: 1.0,20: 1.0,22: 1.0,25: 1.0,39: 1.0,42: 1.0,47: 1.0,135: 1.0,\
            140: 1.0})

        self.expected_train_ftr_vectos[4] = gen_sparse_vector(141,\
            {3: 1.0,9: 1.0,20: 1.0,21: 1.0,22: 1.0,25: 4.0,30: 1.0,39: 1.0,\
            47: 1.0,51: 1.0,58: 1.0,100: 1.0,110: 1.0,115: 1.0,120: 1.0,\
            130: 1.0})


        #expected w's after each update
        self.predicted_tags = [['B-LOC', 'B-LOC', 'B-LOC', 'B-LOC', 'B-LOC', \
        'B-LOC', 'B-LOC', 'B-LOC'],['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],\
        ['O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC', 'O', 'B-LOC', 'I-LOC'],\
        ['O', 'O', 'O', 'O', 'O'],['B-ORG', 'I-ORG', 'O', 'O', 'B-ORG', \
        'I-ORG', 'O', 'O', 'B-LOC', 'I-LOC']]

        self.expected_ws = [gen_sparse_vector(141,{1: -7.0, 3: 1.0, 15: 1.0, \
            21: 1.0, 25: 4.0, 56: -1.0, 58: 1.0, 66: -1.0, 70: 1.0, 76: -1.0,\
             80: 1.0, 91: -1.0, 95: 1.0, 101: -1.0, 105: 1.0, 121: -1.0, \
             125: 1.0}),gen_sparse_vector(141,{1: -7.0, 3: 2.0, 10: 1.0, \
             15: 1.0, 21: 2.0, 25: 1.0, 32: 1.0, 35: -1.0, 51: 1.0, 55: -1.0,\
              56: -1.0, 58: 2.0, 60: -1.0, 66: -1.0, 70: 1.0, 76: -1.0, \
              80: 1.0, 91: -1.0, 95: 1.0, 101: -1.0, 105: 1.0, 121: -1.0, \
              125: 1.0}),gen_sparse_vector(141,{1: -7.0, 9: 1.0, 10: 1.0, \
              15: -1.0, 20: 1.0, 25: 5.0, 32: 1.0, 35: -1.0, 42: 1.0, \
              45: -1.0, 46: -1.0, 49: 1.0, 51: 1.0, 55: -1.0, 56: -1.0, \
              58: 2.0, 60: -1.0, 66: -1.0, 70: 1.0, 73: -1.0, 75: 1.0, \
              76: -1.0, 80: 1.0, 83: -1.0, 85: 1.0, 86: -1.0, 90: 1.0, \
              91: -1.0, 95: 1.0, 101: -1.0, 105: 1.0, 121: -1.0, 125: 1.0}),\
              gen_sparse_vector(141,{1: -7.0, 9: 2.0, 10: 1.0, 15: -1.0, \
                20: 2.0, 22: 1.0, 25: 2.0, 32: 1.0, 35: -1.0, 39: 1.0, \
                40: -1.0, 42: 2.0, 45: -2.0, 46: -1.0, 47: 1.0, 49: 1.0, \
                50: -1.0, 51: 1.0, 55: -1.0, 56: -1.0, 58: 2.0, 60: -1.0, \
                66: -1.0, 70: 1.0, 73: -1.0, 75: 1.0, 76: -1.0, 80: 1.0, \
                83: -1.0, 85: 1.0, 86: -1.0, 90: 1.0, 91: -1.0, 95: 1.0, \
                101: -1.0, 105: 1.0, 121: -1.0, 125: 1.0}),gen_sparse_vector(\
                141,{1: -7.0, 9: 1.0, 10: 1.0, 15: -1.0, 20: 1.0, 22: 1.0, \
                25: 4.0, 27: -1.0, 30: 1.0, 32: 1.0, 35: -1.0, 39: 1.0, \
                40: -1.0, 42: 2.0, 45: -2.0, 46: -1.0, 47: 1.0, 49: 1.0,\
                 50: -1.0, 51: 1.0, 55: -1.0, 56: -1.0, 58: 2.0, 60: -1.0, \
                 66: -1.0, 70: 1.0, 73: -1.0, 75: 1.0, 76: -1.0, 80: 1.0, \
                 83: -1.0, 85: 1.0, 86: -1.0, 90: 1.0, 91: -1.0, 95: 1.0, \
                 101: -1.0, 105: 1.0, 109: -1.0, 110: 1.0, 121: -1.0, \
                 125: 1.0})]


        #expected tag-sequences after two iteration of updates
        self.expected_tags_two_itr = [['B-LOC', 'I-LOC', 'O', 'O', 'O', 'O',\
         'O', 'O'],['O', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],['B-ORG',\
          'I-ORG', 'O', 'O', 'O', 'O', 'O', 'B-LOC', 'I-LOC'],['I-ORG', 'O', \
          'O', 'O', 'B-ORG'],['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-LOC',\
           'I-LOC']]


    def test_feature_vector(self):

        tagger = TaggingPerceptron(vocabulary(self.dataset),
                                            tags(self.dataset)) 
        for sample_ix, (sent, tag) in enumerate(zip(self.sents, \
                    self.tags)):
            ftr_vectr = tagger.feature_vector(sent,tag)
            self.assertSequenceEqual(list(ftr_vectr),
                    list(self.expected_train_ftr_vectos[sample_ix]))
    
    def test_update(self):
        tagger = TaggingPerceptron(vocabulary(self.dataset),
                                            tags(self.dataset))
        for ss, pred ,tt, expected_w in zip(self.sents, self.predicted_tags,
                     self.tags, self.expected_ws):
            
            w = tagger.update(ss, pred, tt)
            self.assertSequenceEqual(list(w), list(expected_w))
    
    def test_decode(self):
        tagger = TaggingPerceptron(vocabulary(self.dataset),
                                    tags(self.dataset))

        # one iteration of updates
        tagger.train(2, self.sents, self.tags,
                                self.sents, self.tags)

        for sent, expected_output in zip(self.sents,
                                    self.expected_tags_two_itr):
            tag_seq = tagger.decode(sent)
            self.assertSequenceEqual(tag_seq, expected_output)

    
if __name__ == '__main__':
    unittest.main()
