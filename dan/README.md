Deep Learning 
=

Overview
--------

For gaining a better understanding of deep learning, we're going to be
looking at deep averaging networks.  These are a very simple
framework, but they work well for a variety of tasks and will help
introduce some of the core concepts of using deep learning in
practice.

In this homework, you'll use pytorch to implement a DAN classifier for
determining the answer to a quizbowl question.

You'll turn in your code on gradescope. This assignment is worth 20
points.

Dataset
----------------

The data are meant to resemble the question answering dataset we used in a
previous homework, but much simpler.

Example Test Data in Unit Tests (Not the actual test)
-----------------

The toy data are designed (and the unit tests use this) so that the words when
are on +1 / -1 on the y or x axis perfectly divide the data.

    def testEmbedding(self):
        for word, embedding in [["unk",      [+0, +0]],
                                ["capital",  [+0, -1]],
                                ["currency", [+0, +1]],
                                ["england",  [+1, +0]],
                                ["russia",   [-1, +0]]]:

This is because there are only four answers in the data, and the four words
combine to signal what the answer is.  After averaging the data, the four
quadrants represent the answer space.

    def testRealAverage(self):       
        reference = [([+0.5, +0.5], "england currency"),
                     ([-0.5, +0.5], "russia currency"),                     
                     ([-0.5, -0.5], "russia capital"),
                     ([+0.5, -0.5], "england capital")]

The provided network for testing for the final layer just stretches things out a bit.

    def testNetwork(self):
        embeddings = self.dan.dan_model.embeddings(self.documents)
        average = self.dan.dan_model.average(embeddings, self.length)
        representation = self.dan.dan_model.network(average)

        reference = [([+1.0, +1.0], "currency england"),
                     ([-1.0, +1.0], "currency russia"),                     
                     ([-1.0, -1.0], "capital russia"),
                     ([+1.0, -1.0], "capital england")]

 The Actual Test Data
-------------------

For the training, the problem looks much the same, but you'll start from
random initialization and there will be lots of words that do not contribute
to finding the right answer.

The data are defined in guesser.py:

             "mini-train": [{"page": "Rouble", "text": "What is this currency of russia"},
                            {"page": "Pound", "text": "What is this currency of england"},
                            {"page": "Moscow", "text": "What is this capital of russia"},
                            {"page": "London", "text": "What is this capital of england"},
                            {"page": "Rouble", "text": "What 's russia 's currency"},
                            {"page": "Pound", "text": "What 's england 's currency"},
                            {"page": "Moscow", "text": "What 's russia 's capital"},
                            {"page": "London", "text": "What 's england 's capital"}],
             "mini-dev": [{"page": "Rouble", "text": "What currency is used in russia"},
                          {"page": "Pound", "text": "What currency is used in england"},
                          {"page": "Moscow", "text": "What is the capital and largest city of russia"},
                          {"page": "London", "text": "What is the capital and largest city of england"},
                          {"page": "Rouble", "text": "What 's the currency in russia"},
                          {"page": "Pound", "text": "What 's the currency in england"},
                          {"page": "Moscow", "text": "What 's the capital of russia"},
                          {"page": "London", "text": "What 's the capital of england"}],

The learned representations won't be as clean, but you should be able to get
perfect accuracy on this dataset.


What components does your network need to have
----------------

Your network needs to use the layers defined in the constructor:
 * `embeddings = nn.Embedding`
 * `linear1 = nn.Linear`
 * `linear2 = nn.Linear`

Between `linear1` and `linear2` (but not after `linear2`) you need to have a
non-linear activation (the unit tests assume ReLU).  You *may* have a dropout
anywhere you like in the network, but it must use the `nn_dropout` so we can
turn it off for deterministic testing.

Loss function
---------------

In the code, there's the option to use one of two loss functions:
 * `TripletMarginLoss`
 * `CrossEntropyLoss`

You are only required to implement *cross entropy*.  The other loss function
is too fiddly to require for a homework.  It is okay to leave the
implementation of the `MarginRankingLoss` blank.

Pytorch data loader
----------------

In this homework, we use pytorch build-in data loader to do data
mini-batching, which provides single or multi-process iterators over the
dataset(https://pytorch.org/docs/stable/data.html).

For data loader, there are two important functions: batichfy and
vectorize. You don't need to implement anything here, but to implement the
rest of your code, you need to understand what they do.


What you have to do
----------------

Coding:
1. Understand the structure of the code and the vectorize funtion.
2. Write DAN model initialization `__init__` of `DanModel`: replace `self.network = None` with a real network.
3. Implement `def average()` function.
4. Implement `def forward()` function.
5. In `run_epoch()`, implement the cross entropy loss function.
6. In `evaluate()` function, compute the loss; generate predicted answers and compute the accuracy.


Optional things that might help
--------------------

1. Change the initialization.  Don't cheat to hand-set parameters to the "right" answer, but you might want to use something like an identity initialization (provided, but only works for specific network configurations).  Your initialization can't depend on knowing the identity of words (in other words, the embedding initialization must always be random).
2. Swap out the optimizer
3. Play around with network widths


Training
--------------------
Here is an example run that does this.

    > python dan_guesser.py \
      --secondary_questions mini-dev \
      --questions mini-train \
      --DanGuesser_hidden_units 50 \
      --DanGuesser_vocab_size 30 \
      --DanGuesser_num_classes 4 \
      --DanGuesser_num_workers 0 \
      --DanGuesser_num_epochs 100 \
      --DanGuesser_embed_dim 50 \
      --DanGuesser_nn_dropout 0.3 \
      --DanGuesser_batch_size 4 \
      --DanGuesser_device cpu
  
    Setting up logging
    INFO:root:Loading questions from mini-train
    INFO:root:Read 8 questions
    INFO:root:Loading questions from mini-dev
    INFO:root:Read 8 questions
    INFO:root:Loaded 8 train examples
    INFO:root:Loaded 8 dev examples
    INFO:root:Example: {'page': 'Rouble', 'text': 'What is this currency of russia'}
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type Dan
    INFO:root:Creating embedding layer for 30 vocab size, 4 classes with 50 dimensions (hidden dimension=50)
    100%|██████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 22595.58it/s]
    INFO:root:Loaded 8 questions with 4 unique answers
    100%|██████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 58457.20it/s]
    INFO:root:Loaded 8 questions with 4 unique answers
    INFO:root:Evaluated 8 examples, accuracy: 0.1250, loss: 0.9452
    INFO:root:Final accuracy=0.125000
    INFO:root:Evaluated 8 examples, accuracy: 0.3750, loss: 0.9155
    INFO:root:Final accuracy=0.375000
    INFO:root:Evaluated 8 examples, accuracy: 0.2500, loss: 0.9182
    INFO:root:Final accuracy=0.250000
    INFO:root:Evaluated 8 examples, accuracy: 0.5000, loss: 0.8889
    INFO:root:Final accuracy=0.500000
    INFO:root:Evaluated 8 examples, accuracy: 0.3750, loss: 0.8835
    INFO:root:Final accuracy=0.375000
    INFO:root:Evaluated 8 examples, accuracy: 0.6250, loss: 0.8153
    INFO:root:Final accuracy=0.625000
    INFO:root:Evaluated 8 examples, accuracy: 1.0000, loss: 0.7764
    INFO:root:Final accuracy=1.000000
    ...
    INFO:root:Evaluated 8 examples, accuracy: 1.0000, loss: 0.0000
    INFO:root:Final accuracy=1.000000
    INFO:root:Evaluated 8 examples, accuracy: 1.0000, loss: 0.0014
    INFO:root:Final accuracy=1.000000
    INFO:root:Evaluated 8 examples, accuracy: 1.0000, loss: 0.0000
    INFO:root:Final accuracy=1.000000


Inspecting Training
--------------------

To help you debug and to inspect how the representations evolve, we provide a
utility to plot the internal representations.

  ./venv/bin/python3 -i dan_guesser.py --secondary_questions mini-dev --questions mini-train --dan_guesser_plot_viz viz --dan_guesser_hidden_units 2 --dan_guesser_num_workers 1 --dan_guesser_num_epochs 50 --dan_guesser_embed_dim 2 --dan_guesser_ans_min_freq=0 --dan_guesser_nn_dropout 0 --dan_guesser_vocab_size 20

Will show you the evolution in pdf files that start with ``train``.  For the data and
parameter plots, however, it will only show you the first two dimensions.  If
you're doing everything in two dimensions, you'll get the whole picture.  If
you're using more than two dimensions, you'll miss out!

Pytorch install
----------------
For more information, check
https://pytorch.org/get-started/locally/

Extra Credit (Up to 15 points)
----------------

For extra credit, you need to get this architecture working on answering real questions.  Here are some things you are allowed to do:

 * initialize with word representations like GloVe or word2vec
 * change how negative examples are selected or the loss function
 * change the nonlinearity, dropout, or optimization
 * change the number of DAN layers
 * change the dimensionality

You cannot:

 * change how the guesser finds similar questions
 * change the general structure of the model (average -> nonlinearity)

You get more points if you:

 * Use the MarginRankingLoss
 * Do not use precomputed embeddings
 * Have a higher accuracy

What to turn in 
----------------

1. Submit your `dan_guesser.py` file and `parameter.py` file (if you change any defaults)
2. Submit an analysis PDF document if you did any of the extra credit.  This document should contain:
      * An explanation of what you did
      * Your results on the full dataset (should be your accuracy given a given number of answer ... bigger the better)
      * The full command line invocation and example outputs


Grading
--------------

To get full points on this assignment, you'll need to have an implementation that can get perfect on the `mini-dev` dataset when trained on the `mini-train` dataset.  
