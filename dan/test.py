import unittest
from dan import DanModel, vectorize
import numpy as np
import torch
import torch.nn as nn




text1 = {'text':torch.LongTensor([[2, 3]]).view(1, 2), 'len': torch.FloatTensor([2])}
text2 = {'text':torch.LongTensor([[1, 3, 4, 2, 1, 0]]).view(1, 6), 'len': torch.FloatTensor([5])}

text3 = {'text':torch.LongTensor([[2, 3, 1], [3, 4, 0]]), 'len': torch.FloatTensor([3, 2])}
text4 = {'text':torch.LongTensor([[1, 0, 0, 0, 0], [2, 4, 4, 3, 1], [3, 4, 1, 0, 0]]), 'len': torch.FloatTensor([1, 5, 3])}



class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.toy_dan_model = DanModel(2, 5, emb_dim=2, n_hidden_units=2)
        self.toy_dan_model.eval()
        weight_matrix = torch.tensor([[0, 0], [0.1, 0.9], [0.3, 0.4], [0.5, 0.5], [0.6, 0.2]])
        self.toy_dan_model.embeddings.weight.data.copy_(weight_matrix)
        l1_weight = torch.tensor([[0.2, 0.9], [-0.1, 0.7]])
        self.toy_dan_model.linear1.weight.data.copy_(l1_weight)
        l2_weight = torch.tensor([[-0.2, 0.4], [-1, 1.3]])
        self.toy_dan_model.linear2.weight.data.copy_(l2_weight)

        nn.init.ones_(self.toy_dan_model.linear1.bias.data)
        nn.init.zeros_(self.toy_dan_model.linear2.bias.data)



    def test_forward(self):
        logits = self.toy_dan_model(text1['text'], text1['len'])
        print(logits[0][1])
        self.assertAlmostEqual(logits[0][0].item(), 0.2130)
        self.assertAlmostEqual(logits[0][1].item(), 0.1724999)
        logits = self.toy_dan_model(text2['text'], text2['len'])
        self.assertAlmostEqual(logits[0][0].item(), 0.2324001)
        self.assertAlmostEqual(logits[0][1].item(), 0.2002001)
        probs = self.toy_dan_model(text1['text'], text1['len'], is_prob=True)
        self.assertAlmostEqual(probs[0][0].item(), 0.5101237)
        self.assertAlmostEqual(probs[0][1].item(), 0.4898764)
        probs = self.toy_dan_model(text2['text'], text2['len'], is_prob=True)
        self.assertAlmostEqual(probs[0][0].item(), 0.5080493)
        self.assertAlmostEqual(probs[0][1].item(), 0.4919507)




    def test_minibatch(self):
        logits = self.toy_dan_model(text3['text'], text3['len'])
        self.assertAlmostEqual(logits[0][0].item(), 0.2360)
        self.assertAlmostEqual(logits[0][1].item(), 0.2070)
        self.assertAlmostEqual(logits[1][0].item(), 0.1910)
        self.assertAlmostEqual(logits[1][1].item(), 0.1219999)
        logits = self.toy_dan_model(text4['text'], text4['len'])
        self.assertAlmostEqual(logits[0][0].item(), 0.2820)
        self.assertAlmostEqual(logits[0][1].item(), 0.2760)
        self.assertAlmostEqual(logits[1][0].item(), 0.2104)
        self.assertAlmostEqual(logits[1][1].item(), 0.1658)
        self.assertAlmostEqual(logits[2][0].item(), 0.2213333)
        self.assertAlmostEqual(logits[2][1].item(), 0.1733332)
        probs = self.toy_dan_model(text3['text'], text3['len'], is_prob=True)
        self.assertAlmostEqual(probs[0][0].item(), 0.5072495)
        self.assertAlmostEqual(probs[0][1].item(), 0.4927505)
        self.assertAlmostEqual(probs[1][0].item(), 0.5172432)
        self.assertAlmostEqual(probs[1][1].item(), 0.4827568)
        probs = self.toy_dan_model(text4['text'], text4['len'], is_prob=True)
        self.assertAlmostEqual(probs[0][0].item(), 0.5015)
        self.assertAlmostEqual(probs[0][1].item(), 0.4985)
        self.assertAlmostEqual(probs[1][0].item(), 0.5111482)
        self.assertAlmostEqual(probs[1][1].item(), 0.4888518)
        self.assertAlmostEqual(probs[2][0].item(), 0.5119978)
        self.assertAlmostEqual(probs[2][1].item(), 0.4880023)

    def test_vectorize(self):
        word2ind = {'text': 0, '<unk>': 1, 'test': 2, 'is': 3, 'fun': 4, 'check': 5, 'vector': 6, 'correct': 7}
        lb = 1
        text1 = ['text', 'test', 'is', 'fun']
        ex1 = text1, lb
        vec_text, _ = vectorize(ex1, word2ind)
        self.assertEqual(vec_text[0], 0)
        self.assertEqual(vec_text[1], 2)
        self.assertEqual(vec_text[2], 3)
        self.assertEqual(vec_text[3], 4)
        text2 = ['check', 'vector', 'correct', 'hahaha']
        ex2 = text2, lb
        vec_text, _ = vectorize(ex2, word2ind)
        self.assertEqual(vec_text[0], 5)
        self.assertEqual(vec_text[1], 6)
        self.assertEqual(vec_text[2], 7)
        self.assertEqual(vec_text[3], 1)


        
        

    


if __name__ == '__main__':
    unittest.main()
