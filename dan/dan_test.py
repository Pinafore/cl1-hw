
import unittest
from guesser import kTOY_DATA
from dan_guesser import *

def create_model(criterion):
        unit_test_params = [("embed_dim", int, 2, "How many dimensions in embedding layer"),
                            ("nn_dropout", float, 0, "How much dropout we use"),
                            ("max_classes", int, 4, "How many classes our dataset can have"),
                            ("ans_min_freq", int, 0, "How many times an answer must appear"),
                            ("plot_viz", str, "test", "Where to write parameter plots"),
                            ("device", str, "cpu", "Where we run pytorch inference"),
                            ("criterion", str, criterion, "Loss function"),
                            ("plot_every", int, 1, "How often we plot"),
                            ("initialization", str, "", "Initialization"),
                            ("vocab_size", int, 5, "How many words in the vocabulary"),
                            ("neg_samp", int, 1, "Number of negative training examples"),
                            ("batch_size", int, 1, "How many examples per batch"),
                            ("num_workers", int, 1, "How many workers to serve examples"),
                            ("num_epochs", int, 1, "How many training epochs"),
                            ("grad_clipping", float, 5.0, "How much we clip the gradients")]
        if criterion == "MarginRankingLoss":
          unit_test_params.append(("hidden_units", int, 2, "Number of dimensions of hidden state"))
        else:
          unit_test_params.append(("hidden_units", int, 4, "Number of dimensions of hidden state"))
        parameters = DanParameters(unit_test_params)
        parameters.set_defaults()
        dan = DanGuesser(parameters)
        dan.initialize_model()

        with torch.no_grad():
          if criterion == "MarginRankingLoss":
            dan.dan_model.embeddings.weight.copy_(torch.tensor([[0,0], [0, -1], [0, 1], [1, 0], [-1, 0]]))
            dan.dan_model.linear1.weight.copy_(torch.tensor([[2, 0], [0, 2]]))
            dan.dan_model.linear1.bias.copy_(torch.tensor([0, 0]))
            dan.dan_model.linear2.weight.copy_(torch.tensor([[2, 0], [0, 2]]))
            dan.dan_model.linear2.bias.copy_(torch.tensor([-1, -1]))
          elif criterion == "CrossEntropyLoss":
            dan.dan_model.embeddings.weight.copy_(torch.tensor([[0, 0], [0, -1], [0, 1], [1, 0], [-1, 0]]))
            dan.dan_model.linear1.weight.copy_(torch.tensor([[+1.0, -1.0],        #london
                                                             [-1.0, -1.0],        #moscow
                                                             [+1.0, +1.0],        #pound
                                                             [-1.0, +1.0]         #rouble
                                                             ]))                  
            dan.dan_model.linear1.bias.copy_(torch.tensor([0, 0, 0, 0]))
            
            dan.dan_model.linear2.weight.data.copy_(torch.eye(4))            
            dan.dan_model.linear2.bias.copy_(torch.tensor([0, 0, 0, 0]))
            
 
        return dan, parameters

def break_dan(dan: DanModel):
  """
  initialize parameters works for our data, but let's change the embedding
  of the word "capital" so that it's broken and we have a non-zero loss.
  """

  with torch.no_grad():
    dan.embeddings.weight.copy_(torch.tensor([[0,0], [1, 0], [1, 0], [0, -1], [0, -1]]))
    dan.linear1.weight.copy_(torch.tensor([[-1, 1], [1, -1]]))
    dan.linear1.bias.copy_(torch.tensor([0, 0]))    
    dan.linear2.weight.copy_(torch.tensor([[1, 1], [1, 1]]))
    dan.linear2.bias.copy_(torch.tensor([1, 1]))
    
  return dan

                                                          
def initialize_data(parameters, model, raw_questions, max_answers):
    # Duplicated the initial dataset and add an additional document that has a
    # different answer
    questions = raw_questions + raw_questions + raw_questions + \
        [{"text": "currency capital", "page": "UNK"},
         {"text": "currency capital", "page": "UNK"}]
    parameters["dan_guesser_max_classes"] = max_answers
    
    data = QuestionData(parameters)
    data.set_data(questions)
    data.build_vocab(questions)

    # build the representations
    data.initialize_lookup_representation()
    data.build_representation(model)
    lookup = data.refresh_index()
    
    return data, lookup

class DanTest(unittest.TestCase):
    def setUp(self):
        self.documents = torch.LongTensor([[2, 3],    # currency england (these are verified in testVocab)
                                           [2, 4],    # currency russia
                                           [1, 4],    # captial russia
                                           [1, 3]])   # capital england
        self.length = torch.IntTensor([2]* 4)

        self.mr_dan, mr_parameters = create_model("MarginRankingLoss")
        self.ce_dan, ce_parameters = create_model("CrossEntropyLoss")

        self.censored_data, self.censored_lookup = initialize_data(mr_parameters,
                                                                   self.mr_dan.dan_model,
                                                                   kTOY_DATA["tiny"], 4)
        self.full_data, self.full_lookup = initialize_data(mr_parameters, self.mr_dan.dan_model,
                                                           kTOY_DATA["tiny"], 5)

        self.vocab = self.full_data.vocab

    def testErrors(self):
        None
        
    def testPlotter(self):
        from dan_guesser import DanPlotter
        new_model = break_dan(self.mr_dan.dan_model)
        
        plotter = DanPlotter("test_plot.pdf")

        plotter.add_checkpoint(self.mr_dan.dan_model, self.full_data, self.full_data, 0)        
        plotter.add_checkpoint(new_model, self.full_data, self.full_data, 10)

        # Fake some accuracy and loss numbers
        from random import random
        import math
        def sigmoid(x):
          return 1 / (1 + math.exp(-x))
  
        accuracy = 3
        loss = 5.0
        
        for ii in range(100):
          sig = sigmoid(-(ii - 50)*2)

          if random() > 0.5:
            loss -= random() * 0.1 * (1 - abs(.5 - sig))                  
            accuracy += round(15 * random() - 6) 
          plotter.accumulate_metrics(ii, min(100, max(0, accuracy)), loss + sig)
        
        plotter.save_plot()
        
    def testSettingRepresentation(self):
        representations = torch.FloatTensor([[4.0, -1.0]])
        indices = [0]

        self.full_data.set_representation(indices, representations)

        result = self.full_data.get_representation(0)
        self.assertEqual(result[0], 4.0, "First dimension of set representation")
        self.assertEqual(result[1], -1,  "Second dimension of set representation")

    def testBatch(self):
        sampler = torch.utils.data.sampler.SequentialSampler(self.censored_data)
        loader = DataLoader(self.full_data, batch_size=4, sampler=sampler,
                            collate_fn=DanGuesser.batchify)

        for idx, batch in enumerate(loader):
           # Because we've repeated the same data with four elements
           # and the batch size is also four, all batches will be
           # the same
           for question, ref in zip(batch['question_text'], self.documents):
                self.util_tensor_compare(question, ref)

           for positive, ref in zip(batch['pos_text'], self.documents):
                # The positive examples are the same as the references
                self.util_tensor_compare(positive, ref)

           for negative, ref in zip(batch['neg_text'], self.documents):
                self.assertTrue(negative[0] != ref[0] or negative[1] != ref[1],
                                "Negative sample should be different from document" +
                                "%s vs %s" % (str(negative), str(ref)))

           for target in [batch['pos_len'], batch['neg_len'], batch['question_len']]:
             for example in target:
               self.assertEqual(example, 2)     

                
    def testNearest(self):
        queries = [([2, 3], "england currency foo", "Pound"),
                   ([4, 2], "russia currency foo", "Rouble"),                   
                   ([4, 1], "russia capital foo", "Moscow"),
                   ([3, 1], "england capital foo", "London")]

        embeddings = self.mr_dan.dan_model.embeddings(self.documents)
        average = self.mr_dan.dan_model.average(embeddings, self.length)
        representation = self.mr_dan.dan_model.network(average).detach().numpy()

        for idx, query in enumerate(queries):
            _, doc, ans = query
            print("???", representation[idx])            
            result = list(self.censored_data.get_nearest(representation[idx], 3))
            print("###", result)            
            annotated_result = [(x, self.censored_data.answers[x]) for x in result]
            ref = [idx for idx, name in enumerate(self.censored_data.answers)
                   if name==ans]

            self.assertEqual(set(result), set(ref), "Query: %s -> %s\nDB:\n%s Ans: %s\nRef: %s Result: %s" %
                             (doc, np.array2string(representation[idx], precision=1),
                              self.censored_data.representation_string(representation[idx]),
                              ans, str(ref), str(annotated_result)))

        # Now do it in batch
        results = self.censored_data.get_batch_nearest(representation, 3,
                                                       lookup_answer=False,
                                                       lookup_answer_id=False)
        for idx, query in enumerate(queries):
          _, doc, ans = query                
          result = results[idx]
          print("####", result)
          annotated_result = [(x, self.censored_data.answers[x]) for x in result]
          ref = [idx for idx, name in enumerate(self.censored_data.answers) if name==ans]
          self.assertEqual(set(result), set(ref), "Query: %s -> %s\nDB:\n%s Ans: %s\nRef: %s Result: %s" %
                             (doc, np.array2string(representation[idx], precision=1),
                              self.censored_data.representation_string(representation[idx]),
                              ans, str(ref), str(annotated_result)))

                             
    def testDataAnswers(self):
        basic_answers = ["Pound", "Rouble", "Moscow", "London"] * 3
        self.assertEqual(self.censored_data.answers, basic_answers)

        self.assertEqual(self.full_data.answers, basic_answers + ["UNK", "UNK"])

    def testPosIndex(self):
        for idx, answer in enumerate(x['page'] for x in kTOY_DATA["tiny"]):
            equivalents = [x for x in range(idx + 1, 12) if x % 4 == idx]
            pos = self.full_data.comparison_indices(idx, answer, self.full_data.answers,
                                                    self.full_data.questions,
                                                    lambda x, y: x==y)
            self.assertEqual(pos, equivalents, "Equivalent answers for %s (%i)" % (answer, idx))
        
    def testNegIndex(self):
        for idx, answer in enumerate(x['page'] for x in kTOY_DATA["tiny"]):
            ref_neg = [x for x in range(12) if x % 4 != idx] + [12, 13]
            neg = self.full_data.comparison_indices(idx, answer, self.full_data.answers,
                                                    self.full_data.questions,
                                                    lambda x, y: x!=y)
            self.assertEqual(neg, ref_neg, "Negative samples for %s (%i)" % (answer, idx))

    def posNegVectorization(self):
        for doc_index in range(len(self.full_data)):
            question, pos, neg = self.full_data[doc_index]
            
            self.assertEqual(question, pos, "Testing %i: %s should be the same as %s" %
                             (doc_index, str(pos), str(neg)))
            self.assertNotEqual(question, neg)
        
    def testEmbeddingNetwork(self):
        embeddings = self.mr_dan.dan_model.embeddings(self.documents)
        average = self.mr_dan.dan_model.average(embeddings, self.length)
        representation = self.mr_dan.dan_model.network(average)

        reference = [([+1.0, +1.0], "currency england"),
                     ([-1.0, +1.0], "currency russia"),                     
                     ([-1.0, -1.0], "capital russia"),
                     ([+1.0, -1.0], "capital england")]

        for row, expected in enumerate(reference):
            expected_vector, text = expected
            text += "\n" + self.full_data.representation_string()
            self.util_tensor_compare(representation[row], torch.FloatTensor(expected_vector),
                                     text + " (Direct)")
            self.util_tensor_compare(self.full_data.get_representation(row),
                                     torch.FloatTensor(expected_vector),
                                     text + "[row: %i] (Internal)" % row)
        
    def testRealAverage(self):       
        reference = [([+0.5, +0.5], "england currency"),
                     ([-0.5, +0.5], "russia currency"),                     
                     ([-0.5, -0.5], "russia capital"),
                     ([+0.5, -0.5], "england capital")]
       
        embeddings = self.mr_dan.dan_model.embeddings(self.documents)
        average = self.mr_dan.dan_model.average(embeddings, self.length)

        for row, expected in enumerate(reference):
            expected_vector, text = expected
            self.util_tensor_compare(average[row], torch.FloatTensor(expected_vector), text)
        
    def testZeroAverage(self):
        documents = torch.IntTensor([[0, 0, 0, 0, 0], 
                                     [0, 1, 2, 3, 4],
                                     [1, 2, 0, 0, 0]])

        length = torch.IntTensor([1, 5, 2])

        embeddings = self.mr_dan.dan_model.embeddings(documents)
        average = self.mr_dan.dan_model.average(embeddings, length)

        for row in range(3):
            self.util_tensor_compare(average[0], torch.FloatTensor([0.0, 0.0]), "Zero %i" % row)

    def testVectorize(self):
        documents = [x["text"] for x in kTOY_DATA["tiny"]]
        reference = [[2, 3], [2, 4], [1, 4], [1, 3]]
        for idx, doc in enumerate(documents):
           q_vec = self.full_data.vectorize(doc, self.vocab, self.full_data.tokenizer)
           self.util_tensor_compare(q_vec[0], torch.LongTensor(reference[idx]),
                                    str(documents[idx]))

    def testSoftmaxRepresentation(self):
        embeddings = self.ce_dan.dan_model.embeddings(self.documents)
        average = self.ce_dan.dan_model.average(embeddings, self.length)
        representation = self.ce_dan.dan_model.network(average)

        reference = [([0, 0, 1, 0], "currency england"),
                     ([0, 0, 0, 1], "currency russia"),                     
                     ([0, 1, 0, 0], "capital russia"),
                     ([1, 0, 0, 0], "capital england")]

        for row, expected in enumerate(reference):
            expected_vector, text = expected
            self.util_tensor_compare(representation[row], torch.FloatTensor(expected_vector),
                                     text + " (Direct)")

    def testMarginErrors(self):
        # should get zero errors with right labels
        right_labels = ["pound", "rouble", "moscow", "london"]
        wrong_labels = ["london", "pound", "rouble", "moscow"]
        
        errors = number_errors(self.documents, self.length, right_labels,
                               self.full_data, self.mr_dan.dan_model)
        self.assertEqual(errors, 0)

        errors = number_errors(self.documents, self.length, wrong_labels,
                               self.full_data, self.mr_dan.dan_model)
        self.assertEqual(errors, 4)
            
    def testSoftmaxErrors(self):
        right_labels = ["Pound", "Rouble", "Moscow", "London"]
        wrong_labels = ["London", "Pound", "Rouble", "Moscow"]
        
        # should get zero errors with right labels
        errors = number_errors(self.documents, self.length, right_labels,
                               self.full_data, self.ce_dan.dan_model)

        self.assertEqual(errors, 0)

        errors = number_errors(self.documents, self.length,
                               wrong_labels,
                               self.full_data, self.ce_dan.dan_model)
        self.assertEqual(errors, 4)
           
    def testVocab(self):
        vocab = self.vocab

        self.assertEqual(vocab["<unk>"],    0)
        self.assertEqual(vocab["capital"],  1)
        self.assertEqual(vocab["currency"], 2)
        self.assertEqual(vocab["england"],  3)
        self.assertEqual(vocab["russia"],   4)

    def testEmbedding(self):
        for word, embedding in [["unk",      [+0, +0]],
                                ["capital",  [+0, -1]],
                                ["currency", [+0, +1]],
                                ["england",  [+1, +0]],
                                ["russia",   [-1, +0]]]:
            model = self.mr_dan.dan_model.embeddings(torch.tensor(self.vocab[word]))
            reference = torch.FloatTensor(embedding)
            self.util_tensor_compare(model, reference, word)

        references = [{"description": "currency england", "embed": [[0, 1], [1, 0]], "tokens": [2, 3]},
                      {"description": "currency russia", "embed": [[0, 1], [-1, 0]], "tokens": [2, 4]},
                      {"description": "capital russia", "embed": [[0, -1], [-1, 0]], "tokens": [1, 4]},                      
                      {"description": "capital england", "embed": [[0, -1], [1, 0]], "tokens": [1, 3]}]
            
        for idx, doc in enumerate(self.documents):
            reference = references[idx]
            description = reference["description"]

            # Even though the vocab was checked before, let's check again for good measure
            print("Vocab check", reference["tokens"], self.documents[idx])
            for token_idx, vocab in enumerate(reference["tokens"]):
                self.assertEqual(vocab, self.documents[idx][token_idx].item(),
                                 "Token %i of Example %i (%s)" % (token_idx, idx, reference["description"]))
            embed = self.mr_dan.dan_model.embeddings(torch.tensor(doc))

            for word_idx, reference_word in enumerate(reference["embed"]):
                reference_word = torch.FloatTensor(reference_word)
                print("// embed  //", embed, reference["embed"], word_idx)
                self.util_tensor_compare(embed[word_idx], reference_word, description + "[%s] (%i, %i)" % (description.split()[word_idx], idx, word_idx))
        # Todo: embed all of the documents in self.documents, check to see that you get the correct result

    def util_tensor_compare(self, model: Tensor, reference: Tensor, extra_info: str=""):
        for position, test_result in enumerate(torch.isclose(model, reference)):
            self.assertTrue(test_result, "Position %i: %0.2f != %0.2f (result=%s, reference=%s)" %
                                (position, model[position], reference[position], str(model), str(reference)) +
                                ": %s" % extra_info if extra_info else "")
        


if __name__ == '__main__':
    import logging


    unittest.main()
    





        
        




        



