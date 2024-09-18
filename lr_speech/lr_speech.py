import random
import math
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile
import glob
from numpy import zeros, sign
from math import exp, log
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

torch.manual_seed(1701)

class SpeechDataset(Dataset):
    def __init__(self, data):
        self.n_samples, self.n_features = data.shape
        # The first column is label, the rest are the features
        self.n_features -= 1
        self.feature = torch.from_numpy(data[:, 1:].astype(np.float32)) # size [n_samples, n_features]
        self.label = torch.from_numpy(data[:, [0]].astype(np.float32)) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

def list_files(directory, vowels):
    '''
    Takes in a directory location of the Hillenbrand data and a list of vowels;
    returns a dictionary mapping from vowels to their soundfiles
    '''
    soundfile_dict = {};
    for vowel in vowels:
        soundfile_dict[vowel] = glob.glob(directory+'/*/*'+vowel+'.wav')

    return soundfile_dict

def create_dataset(soundfile_dict, vowels, num_mfccs):
    """
    Read in wav files, and return a 2-D numpy array that contains your
    speech dataset.

    :param soundfile_dict: A dictionary that, for each vowel V, contains a list of file

    paths corresponding to recordings of the utterance 'hVd'

    :param vowels: The set of vowels to be used in the logistic regression

    :param num_mfccs: The number of MFCCs to include as features

    """

    dataset = zeros((len(soundfile_dict[vowels[0]])+len(soundfile_dict[vowels[1]]),num_mfccs+1))

    # TODO: Complete this function.  You will need to:
    #
    # 1. Extract MFCCs for every wav file in soundfile_dict. The basic code for
    # extracting MFCCs is given, but you will need to store the MFCCs in an
    # appropriate data structure
    #
    # 2. Take the midpoint frame from the MFCC matrix.  If there are an even
    # number of frames in an utterance, take the second of the two midpoint frames.
    #
    # 3. z-score each feature, using the column mean and the column st. dev.
    #
    # Return a numpy array where the first element in each row is the label
    # (0 for the first element of 'vowels', 1 for the second) and the next
    # num_features elements in each row are z-scored MFCCs.


    for vowel in vowels:
        for filename in soundfile_dict[vowel]:
            utterance, _ = librosa.load(filename,sr=16000)
            mfccs = librosa.feature.mfcc(y=utterance, sr=16000, n_mfcc=num_mfccs, n_fft=512, win_length=400, hop_length=160)

    # To use the midpoint frame

    # z-score your dataset

    return dataset


class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        # TODO: Replace this with a real nn.Module
        self.linear = None

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        # TODO: Complete this function
        return 0.5

    def evaluate(self, data):
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc


def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    """Take a single step of the optimizer, we factored it into a single
    function so we could write tests.

    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """

    # You should:
    # A) get predictions
    # B) compute the loss from that prediction
    # C) backprop
    # D) update the parameters

    # There's additional code to print updates (for good software
    # engineering practices, this should probably be logging, but
    # printing is good enough for a homework).


    if (ex+1) % 20 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--vowels", help="The two vowels to be classified, separated by a comma",
                           type=str, default="ih,eh")
    argparser.add_argument("--directory", help="Main directory for the speech files",
                           type=str, default="./Hillenbrand")
    argparser.add_argument("--num_mfccs", help="Number of MFCCs to use",
                           type=int, default=13)
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)

    args = argparser.parse_args()

    directory = args.directory
    num_mfccs = args.num_mfccs
    vowels = args.vowels.split(',')

    # Vowels in the dataset (we're only using a subset):
    # ae, ah, aw, eh, ei, er, ih, iy, oa, oo, uh, uw
    files = list_files(directory, vowels)
    speechdata = create_dataset(files, vowels, num_mfccs)

## # This is for debugging the speech part without needing pytorch
##    X = speechdata[0:,1:]
##    y = speechdata[0:,0]
##
##    model = LogisticRegression(random_state=0, max_iter=1000).fit(X,y)
##    model.score(X,y)

    train_np, test_np = train_test_split(speechdata, test_size=0.15, random_state=1234)
    train, test = SpeechDataset(train_np), SpeechDataset(test_np)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.n_features)

    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    # Replace these with the correct loss and optimizer
    criterion = None
    optimizer = None

    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)
    dataiter = iter(train_loader)

    # Iterations
    for epoch in range(num_epochs):
      for ex, (inputs, labels) in enumerate(train_loader):
        # Run your training process
        step(epoch, ex, logreg, optimizer, criterion, inputs, labels)
