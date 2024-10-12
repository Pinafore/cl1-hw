import argparse
import pickle
import json
import time

from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple, Callable
from collections import Counter, defaultdict
from random import random
from math import copysign

import logging

import nltk
import faiss
import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as ptfunc
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_value_
from torch.nn.modules.loss import _Loss

from guesser import Guesser
from eval import rough_compare
from parameters import Parameters, GuesserParameters, DanParameters
from vocab import Vocab

kUNK = '<unk>'
kPAD = '<pad>'

def ttos(t: torch.tensor):
    """
    Helper function to print tensors without too much wasted space
    """
    return "[" + ",".join("%0.2f" % x for x in t.detach().numpy()) + "]"



class DanPlotter:
    """
    Class to plot the state of DAN inference and create pretty(-ish) plots
    of what the current embeddings and internal representations look like.
    """

    def __init__(self, output_filename="state.pdf"):
        self.output_filename = output_filename
        self.observations = []

        self.metrics = []
        self.temp_gradients = defaultdict(list)
        self.gradients = defaultdict(dict)
        self.parameters = []

    def accumulate_gradient(self, model_params, max_grad=1.0):
        for name, parameter in model_params:
            if parameter.grad is not None:
                if parameter.grad.max() > max_grad:
                    logging.warning("Bad gradient", name, parameter.grad)
                else:
                    self.temp_gradients[name].append(parameter.grad)

    def accumulate_metrics(self, iteration, accuracy, loss):
        self.metrics.append({"iteration": iteration, "value": accuracy, "metric": "accuracy"})
        self.metrics.append({"iteration": iteration, "value": loss, "metric": "loss"})        
                
    def clear_gradient(self, epoch):
        for ii in self.temp_gradients:
            self.gradients[epoch][ii] = torch.mean(torch.stack(self.temp_gradients[ii]), 0)








            if type(self.criterion).__name__ == "MarginRankingLoss":
            
            

    def gradient_vector(self, epoch, name, initial_value, indices, max_gradient=5):
        if epoch in self.gradients:
            for name in self.gradients[epoch]:
                grad = self.gradients[epoch][name]

                # This is a for loop because we don't know how deep we need to go into tensor
                for idx in indices:
                    grad = grad[idx]

                if abs(float(grad)) > max_gradient:
                    logging.warning("Bad gradient print", epoch, name, indices, grad)
                    grad = copysign(max_gradient, float(grad))

                val = float(initial_value + grad)

            return val
        else:
            return float(initial_value)
        
    def add_checkpoint(self, model, train_docs, dev_docs, iteration):
        vocab = train_docs.vocab
        
        # Plot the individual voab words
        for word_idx in range(len(vocab)):
            word = vocab.lookup_token(word_idx)
            embedding = model.embeddings(torch.tensor(word_idx)).clone().detach()
            
            self.parameters.append({"text": word,
                                       "dim_0": float(embedding[0]),
                                       "dim_1": float(embedding[1]),
                                       "layer": "Embedding",
                                       "label": "",
                                       "fold": "",
                                       "grad_0": self.gradient_vector(iteration, "embeddings.weight", embedding[0], (word_idx, 0)),
                                       "grad_1": self.gradient_vector(iteration, "embeddings.weight", embedding[1], (word_idx, 1)),
                                       "iteration": iteration})

        for name, parameter in model.named_parameters():
            if name in ["embeddings.weight"]:
                continue

            if "bias" in name:
                value = parameter.clone().detach()
                self.parameters.append({"dim_0": float(value[0]),
                                        "dim_1": float(value[1]),
                                        "text": name,
                                        "layer": "bias",
                                        "label": "",
                                        "fold": "",
                                        "grad_0": self.gradient_vector(iteration, name, value[0], [0]),
                                        "grad_1": self.gradient_vector(iteration, name, value[1], [1]),
                                        "iteration": iteration})

            else:
                for row, value in enumerate(parameter):
                    value = value.clone().detach()
                    self.parameters.append({"dim_0": float(value[0]),
                                            "dim_1": float(value[1]),
                                            "text": "%s_%i" % (name, row),
                                            "layer": "param",
                                            "label": "",
                                            "fold": "",
                                            "grad_0": self.gradient_vector(iteration, name, value[0], [row, 0]),
                                            "grad_1": self.gradient_vector(iteration, name, value[1], [row, 1]),
                                            "iteration": iteration})

            
        # Plot document averages
        for fold, doc_set in [("train", train_docs),
                              ("dev", dev_docs)]:
            for doc_idx in range(len(doc_set)):
                doc, pos, neg, ans = doc_set[doc_idx]
                embeddings = model.embeddings(doc)
                average = model.average(embeddings, torch.IntTensor([len(doc)])).clone().detach()
                final = model.network(average).clone().detach()
                
                for layer, representation in [("average", average),
                                              ("output", final)]:
                    representation = representation[0]
                    self.observations.append({"label": doc_set.answers[doc_idx],
                                              "dim_0": float(representation[0]),
                                              "dim_1": float(representation[1]),
                                              "layer": layer,
                                              "fold": fold,
                                              "text": doc_set.questions[doc_idx],
                                              "iteration": iteration})

                                             

    def save_plot(self):
        from pandas import DataFrame
        import plotnine as p9

        data = DataFrame(self.metrics)
        data.to_csv(self.output_filename + "_metrics.csv")

        plot = p9.ggplot(data=data, mapping=p9.aes(x='iteration', y='value')) + p9.geom_line() + p9.facet_grid("metric~", scales="free_y")
        plot.save(filename=self.output_filename + "_metrics.pdf")
        
        data = DataFrame(self.observations)
        data.to_csv(self.output_filename + "_data.csv")

        plot = p9.ggplot(data=data, mapping=p9.aes(x='dim_0', y='dim_1', color='fold', shape='label', label='text')) + p9.geom_point() + p9.facet_grid("iteration ~ layer") + p9.geom_text(size=2, nudge_x=0.25, nudge_y=0.25)

        plot.save(filename=self.output_filename + "_data.pdf")

        data = DataFrame(self.parameters)
        data.to_csv(self.output_filename + "_parameters.csv")        
        if data.isnull().any().any():
            logging.warning("NaN values!")
            data = data.replace({np.nan: None})

        plot = p9.ggplot(data=data, mapping=p9.aes(x='dim_0', y='dim_1', label='text')) + p9.geom_point() + p9.facet_grid("iteration ~ layer") + p9.geom_text(size=3, nudge_x=0.25, nudge_y=0.25) + p9.geom_segment(p9.aes(x='dim_0', y='dim_1', xend='grad_0', yend='grad_1'))# , size=2, color="grey", arrow=p9.arrow()) 

        plot.save(filename=self.output_filename + "_params.pdf")

        return plot


    
class DanModel(nn.Module):

    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model, but you can change it if you need to.  


    def __init__(self, model_filename: str, device: str, criterion: _Loss, vocab_size: int, n_answers: int,
                 emb_dim: int=50, activation=nn.ReLU(), n_hidden_units: int=50,
                 nn_dropout: float=.5, 
                 initialization=""):
        super(DanModel, self).__init__()
        self.model_filename = model_filename
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.criterion = criterion
        criterion_name = type(self.criterion).__name__
        
        logging.info("Creating embedding layer for %i vocab size, with %i dimensions (hidden dimension=%i)" % \
                         (self.vocab_size, self.emb_dim, n_hidden_units))
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)

        if criterion_name == "MarginRankingLoss":
              self.linear2 = nn.Linear(n_hidden_units, n_hidden_units)
        elif criterion_name == "CrossEntropyLoss":
              self.linear2 = nn.Linear(n_hidden_units, n_answers)


                      

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU.
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2 -> ReLU()

        self.network = None
        #### Your code here

        # To make this work on CUDA, you need to move it to the appropriate
        # device
        if self.network:
            self.network = self.network.to(device)

        self.initialize_parameters(initialization)


    def initialize_parameters(self, initialization):
        if initialization=="identity":
            assert emb_dim==n_hidden_units, "Cannot initialize to identiy matrix if embedding dimension is not hidden dimension"
            if loss=='cross_entropy':
                assert n_hidden_units == n_answers, "Cannot initialize to identity matrix if hidden dimension doesn't match the number of answers"            
            with torch.no_grad():
                self.linear1.weight.data.copy_(torch.eye(n_hidden_units))
                self.linear2.weight.data.copy_(torch.eye(n_hidden_units))
                self.linear1.bias.data.copy_(torch.zeros(n_hidden_units))
                self.linear2.bias.data.copy_(torch.zeros(n_hidden_units))                
            



                         

                         
    def average(self, text_embeddings: Tensor, text_len: Tensor):
        """
        Given a batch of text embeddings and a tensor of the corresponding lengths, compute the average of them.

        text_embeddings: embeddings of the words
        text_len: the corresponding number of words [1 x Nd]
        """
        average = torch.zeros(text_embeddings.size()[0], text_embeddings.size()[-1])
        #### Your code here

        # You'll want to finish this function.  You don't *have* to use it in
        # your forward function, but it's a good way to make sure the
        # dimensions match and to use the unit test to check your work.
        # In other words, we encourage you to use it.

        return average

    def forward(self, input_text: Iterable[int], text_len: Tensor):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text : vectorized question text, were each word is represented as an integer
        text_len : batch * 1, text length for each question
        """

        representation = torch.FloatTensor([0.0] * self.n_hidden_units)
        #### Your code here
        # Complete the forward funtion.  First look up the word embeddings.
          # Then average them
          # Before feeding them through the network

        return representation


   
class QuestionData(Dataset):
    def __init__(self, parameters: Parameters):
        """
        So the big idea is that because we're doing a retrieval-like
        process, to actually run the guesser we need to store the
        representations and then when we run the guesser, we find the
        closest representation to the query.

        Later, we'll need to store this lookup, so we need to know the embedding size.

        parameters -- The configuration of the Guesser
        """
        
        from nltk.tokenize import word_tokenize

        self.vocab_size = parameters.dan_guesser_vocab_size
        self.answer_max_count = parameters.dan_guesser_max_classes
        self.answer_min_freq = parameters.dan_guesser_ans_min_freq
        self.neg_samples = parameters.dan_guesser_neg_samp

        self.tokenizer = word_tokenize
        self.dimension = parameters.dan_guesser_hidden_units

        self.representations = None

        self.vocab = None
        self.lookup = None
        self.answer_indices = None

    def get_nearest(self, query_row: np.array, n_nearest: int):
        """
        Given the current example representations, find the closest one

        query_representation -- vector[Nh] the representation of this example
        n_nearest -- how many closest vectors to return        
        """
        assert self.lookup is not None, "Did you not initialize the KD tree by calling refresh_index?"
        scores, indices = self.lookup.search(np.array([query_row]), k=n_nearest)

        return indices[0]

    def get_answer_integer(self, item_id: int):
        """
        Given the index of an example, return the integer index of the answer.
        """

        self.answer_indices.index(self.answers[item_id])
    
    def get_batch_nearest(self, query_representation: np.ndarray, n_nearest: int,
                          lookup_answer=True, lookup_answer_id=False):
        """
        Given the current example representations, find the closest one

        query_representation -- numpy Vector[Nd, Nh] the representation of this example
        n_nearest -- how many closest vectors to return
        """
        scores, closest = self.lookup.search(query_representation, k=n_nearest)

        if lookup_answer:
          closest = [self.answers[x[0]] for x in closest]
        if lookup_answer_id:
          closest = [self.answer_indices.index(x) for x in closest]

        return closest
    
    def build_representation(self, model):
        """
        Starting from scratch, take all of the documents, tokenize them,
        look up vocab.  Then apply embeddings and feed through a model.  Save
        those representations and create the index.

        """

        for idx, doc in enumerate(self.questions):
            q_vec = self.vectorize(doc, self.vocab, self.tokenizer)
            # TODO(jbg): check if this should actually be 0 index
            length = len(q_vec[0])            
            embed = model.embeddings(q_vec)
            average = model.average(embed, torch.IntTensor([length]))
            representation = model.network(average)

            logging.debug("Setting %i to be %s (length: %i) -> %s -> %s" % (idx, doc, length, ttos(q_vec[0]), ttos(representation[0])))
            
            self.set_representation([idx], representation)

    def refresh_index(self):
        """
        We use Faiss lookup to find the closest point, so we need to rebuild that after we've updated the representations.

        Because it uses inner product 
        """

        # This only works if the representation were filled with the correct
        # values.  (Extra credit)
        training = self.representations.detach().numpy()

        if self.lookup is None:
            logging.info("Training KD Tree for lookup with dimension %i rows and %i columns" %
                         (training.shape[0], training.shape[1]))  
        self.lookup = faiss.IndexFlatL2(self.dimension)
        self.lookup.add(training)
        return training

    def representation_string(self, query=None):
        representations = self.representations.detach().numpy()

        s = ""
        for idx, row in enumerate(representations):
            s += "%03i %10s %20s %10s" % (idx, np.array2string(row, precision=2), self.questions[idx], self.answers[idx])
            if query is not None:
                s += "\tDot: %0.2f" % row.dot(np.array(query))
            s += "\n"
        return s
        
    def get_representation(self, idx):
        return self.representations[idx]

    def set_representation(self, indices: Iterable[int], representations: Tensor):
        """
        During training, we update the representations in batch, so
        this function takes in all of the indices of those dimensions
        and then copies the individual representations from the
        representions array into this class's representations.

        indices -- 
        """
        assert self.representations is not None, "Did you not call initialize_lookup?"

        assert self.dimension == representations[0].shape[0], \
            "Dimension (%i) doesn't match tensor (%s)" % \
            (self.dimension, int(representations[0].shape[0]))
        for batch_index, global_index in enumerate(indices):
            self.representations[global_index].copy_(representations[batch_index])
        
    def initialize_lookup_representation(self):
        self.representations = torch.FloatTensor(len(self.answers), self.dimension).zero_()
        
    def build_vocab(self, questions: Iterable[Dict]):
        from vocab import Vocab
        
        # TODO (jbg): add in the special noun phrase tokens
        # def yield_tokens(questions):
        #    for question in questions:
        #        yield self.tokenizer(question["text"])

        self.vocab = Vocab.build_vocab_from_iterator([self.tokenizer(x["text"]) for x in questions],
                                                     specials=["<unk>"], max_tokens=self.vocab_size)
        self.vocab.set_default_index(self.vocab["<unk>"])

        return self.vocab
    
    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab

    @staticmethod
    def comparison_indices(this_idx: int, target_answer, full_answer_list, questions,
                           answer_compare, hard_negatives=False):

        # TODO (Extra credit): Select negative samples so that they're "hard"
        # (consider using tf-idf)
        return [idx for idx, ans in enumerate(full_answer_list)
                if answer_compare(ans, target_answer) and idx != this_idx]
        
    def set_data(self, questions: Iterable[Dict], answer_field: str="page"):
        from nltk import FreqDist
        from random import sample

        # TODO(jbg): Should answers be a Vocab object?
        answer_counts = Counter(x[answer_field] for x in questions if x[answer_field])
        valid_answers = list((x, count) for x, count in answer_counts.most_common(self.answer_max_count)
                             if count > self.answer_min_freq)
        self.answer_indices = [x[0] for x in sorted(valid_answers, key=lambda x: (-x[1], x[0]))]
        valid_answers = set(self.answer_indices)

        self.questions = [x["text"] for x in questions if x[answer_field] in valid_answers]
        self.answers = [x[answer_field] for x in questions if x[answer_field] in valid_answers]

        # Extra credit opportunity: Use tf-idf to find hard negatives rather
        # than random ones
        self.positive = []
        self.negative = []
        for index, question in enumerate(tqdm(self.questions)):
            answer = self.answers[index]
            positive_indices = self.comparison_indices(index, answer, self.answers, self.questions, lambda x, y: x==y)
            negative_indices = self.comparison_indices(index, answer, self.answers, self.questions, lambda x, y: x!=y)

            positive = [self.questions[x] for x in positive_indices]
            negative = [self.questions[x] for x in negative_indices]

            self.negative.append(negative)
            self.positive.append(positive)
            
        assert len(self.positive) == len(self.questions)
        assert len(self.negative) == len(self.answers)
        assert len(self.answers) == len(self.questions)

        logging.info("Loaded %i questions with %i unique answers" % (len(self.questions), len(valid_answers)))
        
    @staticmethod
    def vectorize(ex : str, vocab: Vocab, tokenizer: Callable) -> torch.LongTensor:
        """
        vectorize a single example based on the word2ind dict.
        Keyword arguments:
        exs: list of input questions-type pairs
        ex: tokenized question sentence (list)
        label: type of question sentence
        Output:  vectorized sentence(python list) and label(int)
        e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
        """

        assert vocab is not None, "Vocab not initialized"
        
        vec_text = torch.LongTensor([[vocab[x] for x in tokenizer(ex)]])

        return vec_text

    ###You don't need to change this funtion
    def __len__(self):
        """
        How many questions are in the dataset.
        """
        
        return len(self.questions)

    ###You don't need to change this funtion
    def __getitem__(self, index : int):
        """
        Get a single vectorized question, selecting a positive and negative example at random from the choices that we have.
        """
        from random import choice
        
        assert self.tokenizer is not None, "Tokenizer not set up"
        assert len(self.positive) == len(self.questions), "Positive examples not set: %s" % str(self.positive)
        assert len(self.negative) == len(self.questions), "Negative examples not set: %s" % str(self.negative)

        assert len(self.positive[index]), "No positive example: %s" % str(self.positive)
        
        q_text = self.questions[index]
        pos_text = choice(self.positive[index])
        neg_text = choice(self.negative[index])

        q_vec = self.vectorize(q_text, self.vocab, self.tokenizer)
        pos_vec = self.vectorize(pos_text, self.vocab, self.tokenizer)
        neg_vec = self.vectorize(neg_text, self.vocab, self.tokenizer)

        answer = self.answers[index]
        
        item = q_vec, pos_vec, neg_vec, answer
        return item



class DanGuesser(Guesser):
    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.training_data = None
        self.eval_data = None
        self.filename = self.params.dan_guesser_filename
        self.plotter = None
        self.plot_every = -1        
        if parameters.dan_guesser_plot_viz != "":
            self.plotter = DanPlotter(parameters.dan_guesser_plot_viz)
            self.plot_every = parameters.dan_guesser_plot_every

    def run_epoch(self, epoch, train_data_loader: DataLoader, dev_data_loader: DataLoader,
                  checkpoint_every: int=50):
        """
        Train the current model
        Keyword arguments:
        train_data_loader: pytorch build-in data loader output for training examples
        dev_data_loader: pytorch build-in data loader output for dev examples
        """

        eval_acc = -1
        model = self.dan_model
        model.train()

        # You *are* allowed to change the optimizer
        optimizer = torch.optim.SGD(self.dan_model.parameters(), self.params.dan_guesser_learning_rate)

        print_loss_total = 0
        epoch_loss_total = 0
        start = time.time()

        if type(model.criterion).__name__ == "MarginRankingLoss":
            self.training_data.build_representation(model)
            self.training_data.refresh_index()
                    
        #### modify the batch_step to complete the update
        for idx, batch in enumerate(train_data_loader):                
            anchor_text = batch['question_text'].to(self.params.dan_guesser_device)
            anchor_length = batch['question_len'].to(self.params.dan_guesser_device)

            pos_text = batch['pos_text'].to(self.params.dan_guesser_device)
            pos_length = batch['pos_len'].to(self.params.dan_guesser_device)

            neg_text = batch['neg_text'].to(self.params.dan_guesser_device)
            neg_length = batch['neg_len'].to(self.params.dan_guesser_device)

            loss = self.batch_step(optimizer, model, anchor_text, anchor_length,
                                   pos_text, pos_length, neg_text, neg_length,
                                   batch['answers'],
                                   self.training_data.answer_indices,
                                   self.params.dan_guesser_grad_clipping)
            
            if self.plotter is not None:
                self.plotter.accumulate_gradient(model.named_parameters())

            print_loss_total += loss.data.numpy()
            epoch_loss_total += loss.data.numpy()

            if idx % checkpoint_every == 0 and idx > 0:
                print_loss_avg = print_loss_total / checkpoint_every
                self.training_data.refresh_index()

                logging.info('Epoch %i: batch: %d, loss: %.5f time: %.5f' % (epoch, idx, print_loss_avg, time.time()- start))
                print_loss_total = 0

                # TODO: batch_accuracy is not used yet
                # TODO: add in the ability to save the model
                
        eval_acc = evaluate(dev_data_loader, model, self.training_data, self.params.dan_guesser_device, model.criterion)

        return eval_acc, epoch_loss_total    

            
    def set_model(self, model: DanModel):
        """
        Instead of training, set data and model directly.  Useful for unit tests.
        """

        self.dan_model = model


    def initialize_model(self):
        criterion = getattr(nn, self.params.dan_guesser_criterion)()        
        self.dan_model = DanModel(model_filename=self.params.dan_guesser_filename,
                                  device=self.params.dan_guesser_device,
                                  criterion=criterion,
                                  vocab_size=self.params.dan_guesser_vocab_size,
                                  n_answers=self.params.dan_guesser_max_classes,
                                  emb_dim=self.params.dan_guesser_embed_dim,
                                  n_hidden_units=self.params.dan_guesser_hidden_units,
                                  nn_dropout=self.params.dan_guesser_nn_dropout,
                                  initialization=self.params.dan_guesser_initialization)        
                         
    def train_dan(self, raw_train: Iterable[Dict], raw_eval: Iterable[Dict]):
        self.training_data = QuestionData(self.params)
        self.eval_data = QuestionData(self.params)

        vocab = self.training_data.build_vocab(raw_train)
        self.eval_data.set_vocab(vocab)
        self.training_data.set_data(raw_train)
        self.eval_data.set_data(raw_eval)

        self.training_data.initialize_lookup_representation()
        
        train_sampler = torch.utils.data.sampler.RandomSampler(self.training_data)
        dev_sampler = torch.utils.data.sampler.RandomSampler(self.eval_data)
        dev_loader = DataLoader(self.eval_data, batch_size=self.params.dan_guesser_batch_size,
                                    sampler=dev_sampler, num_workers=self.params.dan_guesser_num_workers,
                                    collate_fn=DanGuesser.batchify)
        
        self.best_accuracy = 0.0
        self.plotter.add_checkpoint(self.dan_model, self.training_data, self.eval_data, 0)
        
        for epoch in range(self.params.dan_guesser_num_epochs):               
            # We set the data again to update the positive and negative examples
            # self.training_data.set_data(raw_train)            
            train_loader = DataLoader(self.training_data,
                                          batch_size=self.params.dan_guesser_batch_size,
                                          sampler=train_sampler,
                                          num_workers=self.params.dan_guesser_num_workers,
                                          collate_fn=DanGuesser.batchify)
            
            if self.plotter and epoch % self.plot_every == 0 and epoch > 0:
                logging.info("[Epoch %04i] Dev Accuracy: %0.3f Loss: %f" % (epoch, acc, loss))                
                self.plotter.add_checkpoint(self.dan_model, self.training_data, self.eval_data, epoch)
                
            acc, loss = self.run_epoch(epoch, train_loader, dev_loader)

            if self.plotter:
                self.plotter.clear_gradient(epoch)                
                self.plotter.accumulate_metrics(epoch, acc, loss)
            if acc > self.best_accuracy:
                self.best_accuracy = acc

        self.training_data.refresh_index()
        eval_acc = evaluate(dev_loader, self.dan_model, self.training_data, self.params.dan_guesser_device,
                            self.dan_model.criterion)
        logging.info('Final eval accuracy=%f' % eval_acc)
                
        if self.plotter:
            self.plotter.save_plot()


    def set_eval_data(self, dev_data):
        self.eval_data.copy_lookups(dev_data, self.training_data,
                                    self.answer_field)

    def __call__(self, question: str, n_guesses: int=1):
        model = self.dan_model

        # TODO (jbg): Does this need to be instantiated each time?
        test_question = QuestionData(self.params)
        test_question.set_data([question])
        test_question_loader = DataLoader(self.test_question)

        question_text = test_question_loader[0]['question_text'].to(self.params.dan_guesser_device)
        question_len = test_question_loader[0]['question_len']

        representation = model.forward(question_text, question_len)
        logging.debug("Guess logits (query=%s len=%i): %s" % (question, len(representation), str(representation)))
        raw_guesses = self.training_data.get_nearest(representation, n_guesses)

        guesses = []
        for guess_index in range(n_guesses):
            guess = self.training_data.answers[raw_guesses[guess_index]]

            if guess == kUNK:
                guess = ""
            # TODO: add guess "confidence"
            guesses.append({"guess": guess})
        return guesses

    def save(self):
        """
        Save the DanGuesser to a file
        """
        
        # Guesser.save(self)

        torch.save(self.dan_model, "%s.torch.pkl" % self.filename)

    def load(self):
        # Guesser.load(self)
        
        self.dan_model = torch.load("%s.torch.pkl" % self.filename)

    def batch_step(self, optimizer, model, example, example_length,
                   positive, positive_length, negative, negative_length,
                   answers, answer_lookup, grad_clip):
        """
        This is almost a static function (and used to be) except for the loss function.
        
        """
        loss = None
        criterion = self.dan_model.criterion

        # Prepare the optimizer to ignore gradients and compute predictions


        # Compute the loss
        if type(criterion).__name__ == "MarginRankingLoss":
              loss = None
        elif type(criterion).__name__ == "CrossEntropyLoss":
              loss = None
              
        return loss



    ###You don't need to change this funtion
    @staticmethod
    def batchify(batch):
        """
        Gather a batch of individual examples into one batch,
        which includes the question text, question length and labels
        Keyword arguments:
        batch: list of outputs from vectorize function
        """

        num_examples = len(batch)
        max_length = 0
        for question, pos, neg, answer in batch:
            for example in [question[0], pos[0], neg[0]]:
                max_length = max(max_length, len(example))

        question_lengths = torch.LongTensor(len(batch)).zero_()
        pos_lengths = torch.LongTensor(len(batch)).zero_()
        neg_lengths = torch.LongTensor(len(batch)).zero_()
        answers = []

        question_matrix = torch.LongTensor(num_examples, max_length).zero_()
        positive_matrix = torch.LongTensor(num_examples, max_length).zero_()
        negative_matrix = torch.LongTensor(num_examples, max_length).zero_()
        
        ex_indices = list()

        logging.debug("Batch length: %i" % num_examples)
        logging.debug("Matrix dimensions: Q [%s] P [%s] N [%s]" % (str(question_matrix.shape),
                                                                   str(positive_matrix.shape),
                                                                   str(negative_matrix.shape)))
                     
        
        num_examples = len(batch)
        for idx, ex in enumerate(batch):
            question, pos, neg, answer = [x[0] if isinstance(x, Tensor) else x for x in ex]
            question_lengths[idx] = len(question)
            pos_lengths[idx] = len(pos)
            neg_lengths[idx] = len(neg)
            answers.append(answer)
            assert answers[idx] == answer
            ex_indices.append(idx)

            question_matrix[idx, :len(question)].copy_(torch.LongTensor(question))
            positive_matrix[idx, :len(pos)].copy_(torch.LongTensor(pos))
            negative_matrix[idx, :len(neg)].copy_(torch.LongTensor(neg))

        q_batch = {'question_text': question_matrix, 'question_len': question_lengths,
                   'pos_text': positive_matrix, 'pos_len': pos_lengths,
                   'neg_text': negative_matrix, 'neg_len': neg_lengths,
                   'answers': answers,
                   'ex_indices': ex_indices}

        return q_batch

def cross_entropy_errors():
    """
    Given a model, compute examples' representation in the model, get the
    guess for each example, and then give the number of errors.
    """

def number_errors(question_text: torch.Tensor, question_len: torch.Tensor,
                  labels: Iterable[str], train_data: QuestionData, model: DanModel):
    """
    Helper function called by evaluate that given a batch, generates a
    representation and computes how many errors it had.

    Labels should be answer strings for margin loss function but integers for CE loss.

    """
    # Call the model to get the representation
    # You'll need to update the code here
    #### Your code here
    error = -1
    if type(model.criterion).__name__ == "MarginRankingLoss":
        None
    elif type(model.criterion).__name__ == "CrossEntropyLoss":
        error = len(labels)        

    return error
    
def evaluate(data_loader: DataLoader, model: DanModel, train_data: QuestionData, device: str, criterion: _Loss):
    """
    evaluate the current model, get the accuracy for dev/test set
    Keyword arguments:
    data_loader: pytorch build-in data loader output
    model: model to be evaluated
    device: cpu of gpu
    """

    model.eval()
    num_examples = 0
    error = 0
    for _, batch in enumerate(data_loader):
        question_text = batch['question_text'].to(device)
        question_len = batch['question_len'].to(device)

        labels = batch['answers']
        error += number_errors(question_text, question_len,
                               labels, train_data, model)
        num_examples += question_text.size(0)
        
    accuracy = 1 - (error / num_examples)
    return accuracy




# You basically do not need to modify the below code
# But you may need to add funtions to support error analysis

if __name__ == "__main__":
    from parameters import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging, instantiate_guesser

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    #### check if using gpu is available
    cuda = not flags.no_cuda and torch.cuda.is_available()
    flags.dan_guesser_device = "cuda" if cuda else "cpu"
    logging.info("Device= %s", str(torch.device(flags.dan_guesser_device)))
    guesser_params["dan"].load_command_line_params(flags)
    setup_logging(flags)

    ### Load data
    train_exs = load_questions(flags)
    dev_exs = load_questions(flags, secondary=True)

    logging.info("Loaded %i train examples" % len(train_exs))
    logging.info("Loaded %i dev examples" % len(dev_exs))
    logging.info("Example: %s" % str(train_exs[0]))

    guesser = instantiate_guesser("Dan", flags, guesser_params, False)
    guesser.train_dan(train_exs, dev_exs)
