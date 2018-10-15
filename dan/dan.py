import argparse
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils import clip_grad_norm_
import json
import time

###You don't need to change this funtion

def load_data(filename):
    """
    load the json file into data list
    """

    data = list()
    with open(filename) as json_data:
        questions = json.load(json_data)
        for q in questions:
            q_text = q['question_text'].split()
            label = q['label']
            data.append((q_text, label))
    return data

###You don't need to change this funtion

def load_words(exs):
    """
    vocabuary building

    Keyword arguments:
    exs: list of input questions-type pairs
    """

    words = set()
    UNK = '<unk>'
    PAD = '<pad>'
    word2ind = {PAD: 0, UNK: 1}
    ind2word = {0: PAD, 1: UNK}
    for q_text, _ in exs:
        for w in q_text:
            words.add(w)
    words = sorted(words)
    for w in words:
        idx = len(word2ind)
        word2ind[w] = idx
        ind2word[idx] = w
    words = [PAD, UNK] + words
    return words, word2ind, ind2word


def vectorize(ex, word2ind):
    """
    vectorize a single example based on the word2ind dict. 

    Keyword arguments:
    exs: list of input questions-type pairs
    ex: tokenized question sentence (list)
    label: type of question sentence

    Output:  vectorized sentence(python list) and label(int)
    e.g. ['text', 'test', 'is', 'fun'] -> [0, 2, 3, 4]
    """

    question_text, question_label = ex
    vec_text = [0] * len(question_text)
    #### modify the code to vectorize the question text
    #### You should consider the out of vocab(OOV) cases
    #### question_text is already tokenized

    

    return vec_text, question_label


class Question_Dataset(Dataset):
    """
    Pytorch data class for question classfication data
    """

    def __init__(self, examples, word2ind):
        self.examples = examples
        self.word2ind = word2ind
    
    def __getitem__(self, index):
        return vectorize(self.examples[index], self.word2ind)
    
    def __len__(self):
        return len(self.examples)

###You don't need to change this funtion

def batchify(batch):
    """
    Gather a batch of individual examples into one batch, 
    which includes the question text, question length and labels 

    Keyword arguments:
    batch: list of outputs from vectorize function
    """

    question_len = list()
    label_list = list()
    for ex in batch:
        question_len.append(len(ex[0]))
        label_list.append(ex[1])
    target_labels = torch.LongTensor(label_list)
    x1 = torch.LongTensor(len(question_len), max(question_len)).zero_()
    for i in range(len(question_len)):
        question_text = batch[i][0]
        vec = torch.LongTensor(question_text)
        x1[i, :len(question_text)].copy_(vec)
    q_batch = {'text': x1, 'len': torch.FloatTensor(question_len), 'labels': target_labels}
    return q_batch


def evaluate(data_loader, model, device):
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
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']
        ####Your code here

        top_n, top_i = logits.topk(1)
        num_examples += question_text.size(0)
        error += torch.nonzero(top_i.squeeze() - torch.LongTensor(labels)).size(0)
    accuracy = 1 - error / num_examples
    print('accuracy', accuracy)
    return accuracy


def train(args, model, train_data_loader, dev_data_loader, accuracy, device):
    """
    Train the current model

    Keyword arguments:
    args: arguments 
    model: model to be trained
    train_data_loader: pytorch build-in data loader output for training examples
    dev_data_loader: pytorch build-in data loader output for dev examples
    accuracy: previous best accuracy
    device: cpu of gpu
    """

    model.train()
    optimizer = torch.optim.Adamax(model.parameters())
    criterion = nn.CrossEntropyLoss()
    print_loss_total = 0
    epoch_loss_total = 0
    start = time.time()

    #### modify the following code to complete the training funtion

    for idx, batch in enumerate(train_data_loader):
        question_text = batch['text'].to(device)
        question_len = batch['len']
        labels = batch['labels']

        #### Your code here



        clip_grad_norm_(model.parameters(), args.grad_clipping)
        print_loss_total += loss.data.numpy()
        epoch_loss_total += loss.data.numpy()

        if idx % args.checkpoint == 0 and idx > 0:
            print_loss_avg = print_loss_total / args.checkpoint

            print('number of steps: %d, loss: %.5f time: %.5f' % (idx, print_loss_avg, time.time()- start))
            print_loss_total = 0
            curr_accuracy = evaluate(dev_data_loader, model, device)
            if accuracy < curr_accuracy:
                torch.save(model, args.save_model)
                accuracy = curr_accuracy
    return accuracy




class DanModel(nn.Module):
    """High level model that handles intializing the underlying network
    architecture, saving, updating examples, and predicting examples.
    """

    #### You don't need to change the parameters for the model


    def __init__(self, n_classes, vocab_size, emb_dim=50,
                 n_hidden_units=50, nn_dropout=.5):
        super(DanModel, self).__init__()
        self.n_classes = n_classes
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.n_hidden_units = n_hidden_units
        self.nn_dropout = nn_dropout
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.linear1 = nn.Linear(emb_dim, n_hidden_units)
        self.linear2 = nn.Linear(n_hidden_units, n_classes)

        #### modify the init function, you need to add necessary layer definition here
        #### note that linear1, linear2 are used for mlp layer
        
       


    def forward(self, input_text, text_len, is_prob=False):
        """
        Model forward pass
        
        Keyword arguments:
        input_text : vectorized question text 
        text_len : batch * 1, text length for each question
        in_prob: if True, output the softmax of last layer

        """
        #### write the forward funtion, the output is logits 
        logits = None

        return logits



###You basically do not need to modify the below code 
###But you may need to add funtions to support doing error analysis 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Question Type')
    parser.add_argument('--no-cuda', action='store_true', default=True)
    parser.add_argument('--train-file', type=str, default='./data/question_train_cl1.json')
    parser.add_argument('--dev-file', type=str, default='./data/question_dev_cl1.json')
    parser.add_argument('--test-file', type=str, default='./data/question_test_cl1.json')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--grad-clipping', type=int, default=5)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--save-model', type=str, default='q_type.pt')
    parser.add_argument('--load-model', type=str, default='q_type.pt')
    parser.add_argument('--num-classes', type=int, default=3)
    parser.add_argument('--checkpoint', type=int, default=50)

    args = parser.parse_args()
    #### check if using gpu is available
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    ### Load data
    train_exs = load_data(args.train_file)
    dev_exs = load_data(args.dev_file)
    test_exs = load_data(args.test_file)

    ### Create vocab
    voc, word2ind, ind2word = load_words(train_exs)

    

    if args.test:
        model = torch.load(args.load_model)
        #### Load batchifed dataset
        test_dataset = Question_Dataset(test_exs, word2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)
    else:
        if args.resume:
            model = torch.load(args.load_model)
        else:
            model = DanModel(args.num_classes, len(voc))
            model.to(device)
        print(model)
        #### Load batchifed dataset
        train_dataset = Question_Dataset(train_exs, word2ind)
        train_sampler = torch.utils.data.sampler.RandomSampler(train_dataset)

        dev_dataset = Question_Dataset(dev_exs, word2ind)
        dev_sampler = torch.utils.data.sampler.SequentialSampler(dev_dataset)
        dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                               sampler=dev_sampler, num_workers=0,
                                               collate_fn=batchify)
        accuracy = 0
        for epoch in range(args.num_epochs):
            print('start epoch %d' % epoch)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=train_sampler, num_workers=0,
                                               collate_fn=batchify)
            accuracy = train(args, model, train_loader, dev_loader, accuracy, device)
        print('start testing:\n')

        test_dataset = Question_Dataset(test_exs, word2ind)
        test_sampler = torch.utils.data.sampler.SequentialSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                               sampler=test_sampler, num_workers=0,
                                               collate_fn=batchify)
        evaluate(test_loader, model, device)
