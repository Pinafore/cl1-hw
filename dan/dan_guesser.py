import argparse
import pickle
import json
import time

from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple, Callable
from collections import Counter
from random import random

import logging

from guesser import Guesser, GuesserParameters
from parameters import Parameters

import numpy as np

from tqdm import tqdm
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torchtext.vocab import Vocab

import nltk

from sklearn.neighbors import KDTree

kUNK = '<unk>'

kPAD = '<pad>'

kTOY_VOCAB = [kUNK, "England", "Russia", "capital", "currency"]
kTOY_ANSWER = ["London", "Moscow", "Pound", "Rouble"]



class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model, but you can change it if you need to.  


    def __init__(self, model_filename: str, n_classes: int, device: str, vocab_size:int, emb_dim: int=50,
                 activation=nn.ReLU(), n_hidden_units: int=50, nn_dropout: float=.5):
        super(DanModel, self).__init__()
        self.model_filename = model_filename
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        logging.info("Creating embedding layer for %i vocab size, %i classes with %i dimensions (hidden dimension=%i)" % \
                         (self.vocab_size, n_classes, self.emb_dim, n_hidden_units))
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        # Create the actual prediction framework for the DAN classifier.

        # You'll need combine the two linear layers together, probably
        # with the Sequential function.  The first linear layer takes
        # word embeddings into the representation space, and the
        # second linear layer makes the final prediction.  Other
        # layers / functions to consider are Dropout, ReLU.
        # For test cases, the network we consider is - linear1 -> ReLU() -> Dropout(0.5) -> linear2

        self.network = None
        

        # To make this work on CUDA, you need to move it to the appropriate
        # device
        if self.network:
            self.network = self.network.to(device)

    def average(self, text_embeddings: Tensor, text_len: int):
        """
        Given a batch of text embeddings and a tensor of the corresponding lengths, compute the average of them.

        text_embeddings: embeddings of the words
        text_len: the corresponding number of words
        """
        average = torch.zeros(text_embeddings.size()[0], text_embeddings.size()[-1])

        # You'll want to finish this function.  You don't *have* to use it in
        # your forward function, but it's a good way to make sure the
        # dimensions match and to use the unit test to check your work.
        # In other words, we encourage you to use it.

        return average

    def forward(self, input_text: Iterable[int], text_len: int):
        """
        Model forward pass, returns the logits of the predictions.

        Keyword arguments:
        input_text : vectorized question text, were each word is represented as an integer
        text_len : batch * 1, text length for each question
        is_prob: if True, output the softmax of last layer
        """

        logits = torch.FloatTensor([0.0] * self.n_classes)

        # Complete the forward funtion.  First look up the word embeddings.
        # Then average them
        # Before feeding them through the network

        return representation

   
class QuestionData(Dataset):
    def __init__(self, parameters: Parameters):
        """
        So the big idea is that because we're doing a retrieval-like process, to actually run the guesser we need to store the representations and then when we run the guesser, we find the closest representation to the query.

        Later, we'll need to store this lookup, so we need to know the embedding size.

        parameters -- The configuration of the Guesser
        """
        
        from torchtext.data.utils import get_tokenizer

        self.vocab_size = parameters.DanGuesser_vocab_size
        self.answer_count = parameters.DanGuesser_num_classes
        self.neg_samples = parameters.DanGuesser_neg_samp

        self.tokenizer = get_tokenizer("basic_english")
        self.embedding = parameters.DanGuesser_embed_dim

        self.vocab = None
        self.kd_tree = None

    def get_nearest(self, query_representation: Tensor, answers: Iterable[str], n_nearest: int):
        """
        Given the current example representations, find the closest one

        query_representation -- the representation of this example
        answers -- All of the answers in our dataset, this will be the label we return
        n_nearest -- how many closest vectors to return
        """
        
        closest_indices = self.kd_tree.query(query_representation.detach().numpy(),
                                                 return_distance=False, sort_results=True, k=n_nearest)

        results = []

        for row in np.nditer(closest_indices):
            results.append(answers[x] for x in np.nditer(row))

        return results
        
    def refresh_index(self):
        """
        We use a KD Tree lookup to find the closest point, so we need to rebuild that after we've updated the representations.
        """
        self.kd_tree = KDTree(self.representations)

    def set_representation(self, indices: Iterable[int], representations: Tensor):
        """
        During training, we update the representations in batch, so
        this function takes in all of the indices of those dimensions
        and then copies the individual representations from the
        representions array into this class's representations.

        indices -- 
        """
        
        for batch_index, global_index in enumerate(indices):
            self.representations[global_index].copy_(representations[batch_index])
        
    def initialize_lookup(self):
        self.representations = torch.LongTensor(len(self.answers), self.answer_count).zero_()
        
    def build_vocab(self, questions: Iterable[Dict]):
        from torchtext.vocab import build_vocab_from_iterator
        
        # TODO (jbg): add in the special noun phrase tokens
        def yield_tokens(questions):
            for question in questions:
                yield self.tokenizer(question["text"])

        self.vocab = build_vocab_from_iterator([self.tokenizer(x["text"]) for x in questions],
                                                specials=["<unk>"], max_tokens=self.vocab_size)
        self.vocab.set_default_index(self.vocab["<unk>"])

        return self.vocab

    def set_vocab(self, vocab: Vocab):
        self.vocab = vocab

    def set_data(self, questions: Iterable[Dict], answer_field: str="page"):
        from nltk import FreqDist
        from random import sample

        answer_counts = FreqDist(x[answer_field] for x in questions if x[answer_field])
        valid_answers = set(x for x, count in answer_counts.most_common(self.answer_count) if count > 1)
        self.questions = [x["text"] for x in questions if x[answer_field] in valid_answers]
        self.answers = [x[answer_field] for x in questions if x[answer_field] in valid_answers]

        # Extra credit opportunity: Use tf-idf to find hard negatives rather
        # than random ones
        self.positive = []
        self.negative = []
        for index, question in enumerate(tqdm(self.questions)):
            answer = self.answers[index]
            positive_indices = [idx for idx, ans in enumerate(self.answers) if ans==answer and idx!=index]
            negative_indices = sample([idx for idx, ans in enumerate(self.answers) if ans!=answer], self.neg_samples)

            negative = [self.questions[x] for x in positive_indices]
            positive = [self.questions[x] for x in negative_indices]

            self.negative.append(negative)
            self.positive.append(positive)
            
        assert len(self.positive) == len(self.questions)
        assert len(self.negative) == len(self.answers)
        assert len(self.answers) == len(self.questions)

        logging.info("Loaded %i questions with %i unique answers" % (len(self.questions), len(valid_answers)))
        
    @staticmethod
    def vectorize(ex : str, vocab: Vocab, tokenizer: Callable):
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
        
        vec_text = [vocab[x] for x in tokenizer(ex)]

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
        q_text = self.questions[index]
        pos_text = choice(self.positive[index])
        neg_text = choice(self.negative[index])

        q_vec = self.vectorize(q_text, self.vocab, self.tokenizer)
        pos_vec = self.vectorize(pos_text, self.vocab, self.tokenizer)
        neg_vec = self.vectorize(neg_text, self.vocab, self.tokenizer)
        
        item = (q_vec, len(q_vec),
                pos_vec, len(pos_vec),
                neg_vec, len(neg_vec),
                index)
        return item


class DanParameters(GuesserParameters):

    # parser.add_argument('--DanGuesser_filename', type=str, default="models/DanGuesser.pkl")
    # parser.add_argument('--DanGuesser_min_df', type=float, default=30,
    #                         help="How many documents terms must be in before inclusion in DAN vocab (either percentage or absolute count)")
    # parser.add_argument('--DanGuesser_max_df', type=float, default=.4,
    #                         help="Maximum documents terms can be in before inclusion in DAN vocab (either percentage or absolute count)")
    # parser.add_argument('--DanGuesser_min_answer_freq', type=int, default=30,
    #                         help="How many times we need to see an answer before including it in DAN output")
    # parser.add_argument('--DanGuesser_embedding_dim', type=int, default=100)
    # parser.add_argument('--DanGuesser_hidden_units', type=int, default=100)
    # parser.add_argument('--DanGuesser_dropout', type=int, default=0.5)
    # parser.add_argument('--DanGuesser_unk_drop', type=float, default=0.95)    
    # parser.add_argument('--DanGuesser_grad_clipping', type=float, default=5.0)
    # parser.add_argument('--DanGuesser_batch_size', type=int, default=128)    
    # parser.add_argument('--DanGuesser_num_epochs', type=int, default=20)
    # parser.add_argument('--DanGuesser_num_workers', type=int, default=0)

    
    dan_params = [("embed_dim", int, 300, "How many dimensions in embedding layer"),
                  ("batch_size", int, 120, "How many examples per batch"),
                  ("num_workers", int, 8, "How many workers to serve examples"),
                  ("num_classes", int, 10000, "Maximum number of classes"),
                  ("hidden_units", int, 100, "Number of dimensions of hidden state"),
                  ("nn_dropout", float, 0.5, "How much dropout we use"),
                  ("device", str, "cuda", "Where we run pytorch inference"),                  
                  ("num_epochs", int, 20, "How many training epochs"),
                  ("neg_samp", int, 5, "Number of negative training examples"),
                  ("unk_drop", bool, True, "Do we drop unknown tokens or use UNK symbol"),
                  ("grad_clipping", float, 5.0, "How much we clip the gradients")]
    
    def __init__(self):
        GuesserParameters.__init__(self)
        self.name = "DanGuesser"
        self.params += self.dan_params

    def unit_test(self):
        Parameters.set_defaults(self)

        self.embed_dim = 2
        self.hidden_units = 2
        self.nn_dropout = 0
        self.device = "cpu"

class DanGuesser(Guesser):
    def __init__(self, parameters: Parameters):
        from sklearn.feature_extraction.text import CountVectorizer
        
        self.params = parameters

        self.vectorizer = CountVectorizer(stop_words='english',
                                          max_features=parameters.DanGuesser_vocab_size)

        self.kd_tree = None
        self.training_data = None
        self.train_data = None
        self.eval_data = None

    def set_model(self, model: DanModel):
        """
        Instead of training, set data and model directly.  Useful for unit tests.
        """

        self.dan_model = model


    def initialize_model(self):
        self.dan_model = DanModel(model_filename=self.params.DanGuesser_filename,
                                  n_classes=self.params.DanGuesser_num_classes,
                                  device=self.params.DanGuesser_device,
                                  vocab_size=self.params.DanGuesser_vocab_size,
                                  emb_dim=self.params.DanGuesser_embed_dim,
                                  n_hidden_units=self.params.DanGuesser_hidden_units,
                                  nn_dropout=self.params.DanGuesser_nn_dropout)        
    
    def train_dan(self, raw_train: Iterable[Dict], raw_eval: Iterable[Dict]):
        self.training_data = QuestionData(self.params)
        self.eval_data = QuestionData(self.params)

        vocab = self.training_data.build_vocab(raw_train)
        self.eval_data.set_vocab(vocab)

        self.training_data.set_data(raw_train)
        self.eval_data.set_data(raw_eval)

        self.training_data.initialize_lookup()
        
        train_sampler = torch.utils.data.sampler.RandomSampler(self.training_data)
        dev_sampler = torch.utils.data.sampler.RandomSampler(self.eval_data)
        dev_loader = DataLoader(self.eval_data, batch_size=self.params.DanGuesser_batch_size,
                                    sampler=dev_sampler, num_workers=self.params.DanGuesser_num_workers,
                                    collate_fn=DanGuesser.batchify)
        self.best_accuracy = 0.0

        for epoch in range(self.params.DanGuesser_num_epochs):
            train_loader = DataLoader(self.training_data,
                                          batch_size=self.params.DanGuesser_batch_size,
                                          sampler=train_sampler,
                                          num_workers=self.params.DanGuesser_num_workers,
                                          collate_fn=DanGuesser.batchify)
            self.run_epoch(train_loader, dev_loader)


    def train(self, training_data, answer_field, split_by_sentence,
              min_length=-1, max_length=-1, remove_missing_pages=True):
        answers_to_questions = Guesser.train(self, training_data, answer_field,
                                             split_by_sentence=split_by_sentence,
                                             min_length=min_length,
                                             max_length=max_length,
                                             remove_missing_pages=remove_missing_pages)

        answer_count = dict((x, len(y)) for x, y in
                            answers_to_questions.items())

        self.train_data.initialize(answers=self.answers,
                                   questions=self.questions,
                                   answer_count=answer_count,
                                   unk_drop=self.unk_drop,
                                   tokenizer=self.phrase_tokenize)

        num_classes = self.train_data.num_answers
        vocab_size = self.train_data.num_vocab
        self.dan_model = DanModel(model_filename=self.model_filename,
                                  n_classes=num_classes,
                                  device=self.device,
                                  vocab_size=vocab_size,
                                  emb_dim=self.embedding_dimension,
                                  n_hidden_units=self.hidden_units,
                                  nn_dropout=self.nn_dropout)

    def set_eval_data(self, dev_data):
        self.eval_data.copy_lookups(dev_data, self.train_data,
                                    self.answer_field)

    def __call__(self, question: str, n_guesses: int=1):
        model = self.dan_model

        question_text = torch.LongTensor([self.vectorize(question)])
        logging.debug("Vectorizing question %s to become %s" % (question, str(question_text)))
        question_length = torch.LongTensor([[len(question_text)]])

        representation = model.forward(question_text, question_length)
        logging.debug("Guess logits (query=%s len=%i): %s" % (question, len(logits), str(logits)))

        # If we don't have enough possible answers, constrain the number of
        # top results accordingly
        n_guesses = min(n_guesses, self.train_data.num_answers)
        assert n_guesses > 0, "Need to at least return 1 guess"
        
        raw_guesses = model.get_nearest(n_guesses)
        
        for guess_index in range(n_guesses):
            guess = self.train_data.int_to_answer[indices[guess_index]]

            if guess == kUNK:
                guess = ""

            guesses.append({"guess": guess, "confidence": values[guess_index]})
        return guesses

    def save(self):
        """
        Save the DanGuesser to a file
        """
        
        Guesser.save(self)

        torch.save(self.dan_model, "%s.torch.pkl" % self.model_filename)
        self.train_data.save(self.model_filename)

    def load(self):
        Guesser.load(self)
        
        self.dan_model = torch.load("%s.torch.pkl" % self.model_filename)
        self.train_data.load(self.model_filename)

        

    def run_epoch(self, train_data_loader: DataLoader, dev_data_loader: DataLoader,
                  checkpoint_every: int=50):
        """
        Train the current model
        Keyword arguments:
        train_data_loader: pytorch build-in data loader output for training examples
        dev_data_loader: pytorch build-in data loader output for dev examples
        """

        model = self.dan_model
        model.train()
        optimizer = torch.optim.Adamax(self.dan_model.parameters())
        criterion = nn.TripletMarginLoss()
        print_loss_total = 0
        epoch_loss_total = 0
        start = time.time()

        accuracy = 0

        #### modify the following code to complete the training funtion
        for idx, batch in enumerate(train_data_loader):
            anchor_text = batch['anchor_text'].to(self.params.DanGuesser_device)
            anchor_length = batch['anchor_len'].to(self.params.DanGuesser_device)

            pos_text = batch['pos_text'].to(self.params.DanGuesser_device)
            pos_length = batch['pos_len'].to(self.params.DanGuesser_device)

            neg_text = batch['neg_text'].to(self.params.DanGuesser_device)
            neg_length = batch['neg_len'].to(self.params.DanGuesser_device)

            #### Your code here

            # This code is needed to figure out the nearest neighbors of the example
            self.training_data.set_representation(batch['ex_indices'], anchor_rep)
            self.training_data.refresh_index()

            self.kd_tree = None

            clip_grad_norm_(model.parameters(), self.params.DanGuesser_grad_clipping)
            print_loss_total += loss.data.numpy()
            epoch_loss_total += loss.data.numpy()

            if idx % checkpoint_every == 0:
                print_loss_avg = print_loss_total / checkpoint_every
                self.training_data.initialize_lookup()

                logging.info('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
                print_loss_total = 0
                curr_accuracy = evaluate(dev_data_loader, self.dan_model, self.training_data, self.params.DanGuesser_device)

                if accuracy < curr_accuracy:
                    torch.save(model, "%s.torch.pkl" % self.model_filename)
                    accuracy = curr_accuracy

        logging.info('Final accuracy=%f' % accuracy)
        return accuracy

    @staticmethod
    def batch_accuracy(anchor, positive, negative):
        """
        Check how many of the positive examples are closer to the query than the negative example
        """
        
        pos_sim = torch.sum(anchor*positive, axis=1).detach().numpy()
        neg_sim = torch.sum(anchor*negative, axis=1).detach().numpy()

        return sum(1 for x in np.greater(pos_sim, neg_sim) if x) / len(anchor)
    

    ###You don't need to change this funtion
    @staticmethod
    def batchify(batch):
        """
        Gather a batch of individual examples into one batch,
        which includes the question text, question length and labels
        Keyword arguments:
        batch: list of outputs from vectorize function
        """

        logging.info("Batch length: %i" % len(batch))

        question_lengths = torch.LongTensor(len(batch)).zero_()
        pos_lengths = torch.LongTensor(len(batch)).zero_()
        neg_lengths = torch.LongTensor(len(batch)).zero_()
        
        ex_indices = list()

        num_examples = len(batch)
        for idx, ex in enumerate(batch):
            _, question_len, _, pos_len, _, neg_len, example_index = ex
            question_lengths[idx] = question_len
            pos_lengths[idx] = pos_len
            neg_lengths[idx] = neg_len
            ex_indices.append(example_index)
        
        question_matrix = torch.LongTensor(num_examples, max(question_lengths)).zero_()
        positive_matrix = torch.LongTensor(num_examples, max(pos_lengths)).zero_()
        negative_matrix = torch.LongTensor(num_examples, max(neg_lengths)).zero_()

        for idx, ex in enumerate(batch):
            question, _, pos, _, neg, _, _ = ex
            question_matrix[idx, :len(question)].copy_(torch.LongTensor(question))
            positive_matrix[idx, :len(pos)].copy_(torch.LongTensor(pos))
            negative_matrix[idx, :len(neg)].copy_(torch.LongTensor(neg))
            

        q_batch = {'anchor_text': question_matrix, 'anchor_len': question_lengths,
                   'pos_text': positive_matrix, 'pos_len': pos_lengths,
                   'neg_text': negative_matrix, 'neg_len': neg_lengths,
                   'ex_indices': ex_indices}

        return q_batch

def evaluate(data_loader, model, train_data, device):
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
    for idx, batch in enumerate(data_loader):
        question_text = batch['anchor_text'].to(device)
        question_len = batch['anchor_len']
        labels = batch['ex_indices']

        # Call the model to get the logits
        # You'll need to update the code here
        closest = train_data.get_nearest(representation, labels, 1)

        error += sum(1 for guess, answer in zip(closest, labels) if guess != answer)
        num_examples += question_text.size(0)

    accuracy = 1 - error / num_examples
    return accuracy




# You basically do not need to modify the below code
# But you may need to add funtions to support error analysis

if __name__ == "__main__":
    from params import load_questions, add_guesser_params, add_general_params, add_question_params, setup_logging, instantiate_guesser

    logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Question Type')
    add_general_params(parser)
    guesser_params = add_guesser_params(parser)
    add_question_params(parser)

    flags = parser.parse_args()
    #### check if using gpu is available
    cuda = not flags.no_cuda and torch.cuda.is_available()
    flags.DanGuesser_device = "cuda" if cuda else "cpu"
    logging.info("Device= %s", str(torch.device(flags.DanGuesser_device)))
    guesser_params["dan"].load_command_line_params(flags)
    setup_logging(flags)

    ### Load data
    train_exs = load_questions(flags)
    dev_exs = load_questions(flags, secondary=True)

    # dev_exs = load_data(args.dev_file, -1)
    # test_exs = load_data(args.test_file, -1)

    logging.info("Loaded %i train examples" % len(train_exs))
    logging.info("Loaded %i dev examples" % len(dev_exs))
    logging.info("Example: %s" % str(train_exs[0]))

    ### Create vocab
    data = QuestionData(guesser_params["dan"])

    guesser = instantiate_guesser("Dan", flags, guesser_params, False)
    guesser.train_dan(train_exs, dev_exs)

