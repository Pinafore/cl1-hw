
import unittest
from guesser import kTOY_DATA
from dan_guesser import *

class DanTest(unittest.TestCase):
    def setUp(self):

        parameters = DanParameters()
        parameters.unit_test()
        
        self.wide_dan_model = DanModel("", n_classes=1, vocab_size=1, emb_dim=4, n_hidden_units=50, device='cpu')
        self.toy_dan_guesser = DanGuesser(parameters)

        self.toy_json = kTOY_DATA["tiny"]
        
        # Create data object and turn it into the toy dataset
        toy_dataset = QuestionData(parameters)

        toy_dataset.build_vocab(self.toy_json)
        toy_dataset.set_data(self.toy_json)

        self.toy_dan_guesser.train(toy_dataset, "page", False)        
        self.toy_qa = DanModel("model/toy_dan", n_classes=4, vocab_size=5,
                               device="cpu", emb_dim=2, activation=nn.Hardtanh(),
                               n_hidden_units=2, nn_dropout=0.0)


        embedding = [[ 0,  0],           # UNK
                     [ 1,  0],           # England
                     [-1,  0],           # Russia                     
                     [ 0,  1],           # capital
                     [ 0, -1],           # currency
                     ]

        first_layer = [[1, 0], [0, 1]] # Identity matrix
            
        second_layer = [[ 1,  1],        # -> London
                        [-1,  1],        # -> Moscow                        
                        [ 1, -1],        # -> Pound
                        [-1, -1],        # -> Rouble
                        ]

        with torch.no_grad():
            self.toy_qa.linear1.bias *= 0.0
            self.toy_qa.linear2.bias *= 0.0
            self.toy_qa.embeddings.weight = nn.Parameter(torch.FloatTensor(embedding))
            self.toy_qa.linear1.weight.copy_(torch.FloatTensor(first_layer))
            self.toy_qa.linear2.weight.copy_(torch.FloatTensor(second_layer))

        self.toy_dan_guesser.set_data_model(toy_dataset, self.toy_qa)
        
        # self.toy_dan_model = DanModel(2, 5, emb_dim=2, n_hidden_units=2)
        # self.wide_dan_model = DanModel(1, 1, emb_dim=4, n_hidden_units=1)
        # self.toy_dan_model.eval()
        # weight_matrix = torch.tensor([[0, 0], [0.1, 0.9], [0.3, 0.4], [0.5, 0.5], [0.6, 0.2]])
        # self.toy_dan_model.embeddings.weight.data.copy_(weight_matrix)
        # l1_weight = torch.tensor([[0.2, 0.9], [-0.1, 0.7]])
        # self.toy_dan_model.linear1.weight.data.copy_(l1_weight)
        # l2_weight = torch.tensor([[-0.2, 0.4], [-1, 1.3]])
        # self.toy_dan_model.linear2.weight.data.copy_(l2_weight)

        # nn.init.ones_(self.toy_dan_model.linear1.bias.data)
        # nn.init.zeros_(self.toy_dan_model.linear2.bias.data)

    def testAverage(self):
        d1 = [[0, 1, 2, 3]] * 3
        d2 = [[1, 2, 4, 8]] * 2
        # Add padding to second document
        d2.append([0, 0, 0, 0])

        docs = torch.tensor([d1, d2])
        lengths = torch.tensor([3, 2])

        average = self.wide_dan_model.average(docs, lengths)
        print("AVG", average)

        for ii in range(4):
            self.assertAlmostEqual(float(average[0][ii]), ii,
                                   msg="Document 1 average test failed at index %i" % ii)
            self.assertAlmostEqual(float(average[1][ii]), 2.0**ii,
                                   msg="Document 2 average test failed at index %i" % ii)            
        
    def testCorrectPrediction(self):
        guesser = self.toy_dan_guesser        
        
        # Make sure correct embeddings are there
        embedding = {kUNK: (0, 0), "England": (1, 0), "Russia": (-1, 0), "capital": (0, 1), "currency": (0, -1)}
        for word_idx, word in enumerate(kTOY_VOCAB):
            embeddings = self.toy_qa.embeddings(torch.tensor([word_idx]))
            first, second = embedding[word]
            self.assertEqual(float(embeddings.flatten()[0]), first,
                             "First dimension of word %s (%i) does not match %f" % (word, word_idx, first))
            self.assertEqual(float(embeddings.flatten()[1]), second,
                             "Second dimension of word %s (%i) does not match %f" % (word, word_idx, second))

        # Test predictions
        for words, indices, answer, ans_idx in [("capital England",  [3, 1], "London", 0),
                                                ("capital Russia",   [3, 2], "Moscow", 1),
                                                ("currency England", [4, 1], "Pound",  2),
                                                ("currency Russia",  [4, 2], "Rouble", 3)]:
            # We need to put the indices and lengths in a list because the inference
            # and averaging assumes minibatches of multiple documents, so each
            # document needs to have the same number of words and be the first
            # dimension of the tensor.
            text_len = torch.FloatTensor([2])
            query = torch.tensor([indices])            
            embeddings = self.toy_qa.embeddings(query)

            average = self.toy_qa.average(torch.tensor(embeddings), text_len)
            print("AVERAGE", words, average)
            
            result = self.toy_qa.forward(query, text_len)
            highest = int(torch.argmax(result))

            self.assertEqual(ans_idx, highest,
                             ("Given query %s (%s) embedded as %s, got answer index %i"
                              " as the argmax from final output %s, does not match index"
                              " corresponding to expected answer %s") %
                             (words, str(indices), str(average), highest, str(result), answer))

            guess = self.toy_dan_guesser(words)[0]

            self.assertEqual(self.toy_dan_guesser.vectorize(words), indices)
            self.assertEqual([float(x) for x in self.toy_dan_guesser.dan_model.forward(query, text_len).flatten()],
                             [float(x) for x in self.toy_qa.forward(query, text_len).flatten()],
                             "Guesser model / text model mismatch")
            print("Confirmed that %s indices are %s" %  (words, str(indices)))
            self.assertEqual(guess['guess'], answer,
                             ("\n" +
                              "After checking the answer to the question: '%s' \n" +
                              "directly in the DAN model was:             '%s',\n"
                              "the guesser gave us:                       '%s',\n" +
                              "suggesting that there's a problem in the \n" +
                              " guesser code") % (words, answer, guess['guess']))

    def test_train_preprocessing(self):
        """
        On the toy data, make sure that create_indices creates the correct vocabulary and 
        """
        guesser = self.toy_dan_guesser

        print("VOCAB", guesser.train_data.int_to_vocab)
        print("ANSWERS", guesser.train_data.int_to_answer)
        
        self.assertEqual(guesser.dan_model.vocab_size, 5)
        self.assertEqual(guesser.dan_model.n_classes, 4)

        self.assertEqual(guesser.train_data.int_to_vocab,
                         [kUNK, 'England', 'Russia', 'capital', 'currency'])
        self.assertEqual(guesser.train_data.int_to_answer,
                         ['London', 'Moscow', 'Pound', 'Rouble'])

        question = "capital England"
        self.assertEqual(guesser.phrase_tokenize(question), ["capital", "England"])
        self.assertEqual(guesser.vectorize(question), [3, 1])
        self.assertEqual(guesser.train_data.vectorize(guesser.phrase_tokenize(question),
                                                      guesser.train_data.vocab_to_int),
                         [3, 1])
        
    def test_vectorize(self):
        """
        Given a vocabulary, make sure that the text is mapped to the correct vectors.
        """
        
        word2ind = {'text': 0, '<unk>': 1, 'test': 2, 'is': 3, 'fun': 4,
                    'check': 5, 'vector': 6, 'correct': 7}
        lb = 1
        text1 = ['text', 'test', 'is', 'fun']
        ex1 = text1
        vec_text = QuestionData.vectorize(ex1, word2ind)
        self.assertEqual(vec_text[0], 0)
        self.assertEqual(vec_text[1], 2)
        self.assertEqual(vec_text[2], 3)
        self.assertEqual(vec_text[3], 4)
        text2 = ['check', 'vector', 'correct', 'hahaha']
        ex2 = text2
        vec_text = QuestionData.vectorize(ex2, word2ind)
        self.assertEqual(vec_text[0], 5)
        self.assertEqual(vec_text[1], 6)
        self.assertEqual(vec_text[2], 7)
        self.assertEqual(vec_text[3], 1)
        

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
