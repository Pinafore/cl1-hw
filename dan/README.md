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

You'll turn in your code on the submit server. This assignment is worth 40
points.

Dataset
----------------

The data are meant to resemble the question answering dataset we used in a
previous homework, but much simpler.

Toy Data in Unit Tests
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

Slightly More Realistic Data
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

Here is an example run that does this.

    > ./venv/bin/python3 -i dan_guesser.py --secondary_questions mini-dev --questions mini-train --dan_guesser_plot_viz viz --dan_guesser_hidden_units 4 --dan_guesser_vocab_size=10 --dan_guesser_max_classes=4 --dan_guesser_num_workers 1 --dan_guesser_num_epochs 1000 --dan_guesser_embed_dim 4 --dan_guesser_ans_min_freq=0 --dan_guesser_nn_dropout 0 --dan_guesser_batch_size=8 --dan_guesser_plot_every 10 --dan_guesser_criterion=CrossEntropyLoss --dan_guesser_initialization='' --dan_guesser_device=gpu --dan_guesser_learning_rate=0.1
    Setting up logging
    INFO:root:Loaded 8 train examples
    INFO:root:Loaded 8 dev examples
    INFO:root:Example: {'page': 'Rouble', 'text': 'What is this currency of russia'}
    INFO:root:Using device 'cpu' (cuda flag=False)
    INFO:root:Initializing guesser of type Dan
    INFO:root:Creating embedding layer for 10 vocab size, with 4 dimensions (hidden dimension=4)
    100%|███████████████████████████████████████████| 8/8 [00:00<00:00, 162098.71it/s]
    INFO:root:Loaded 8 questions with 4 unique answers
    100%|███████████████████████████████████████████| 8/8 [00:00<00:00, 231409.88it/s]
    INFO:root:Loaded 8 questions with 4 unique answers
    INFO:root:[Epoch 0010] Dev Accuracy: 0.250 Loss: 1.377816
    INFO:root:[Epoch 0020] Dev Accuracy: 0.375 Loss: 1.369576
    INFO:root:[Epoch 0030] Dev Accuracy: 0.250 Loss: 1.362568
    INFO:root:[Epoch 0040] Dev Accuracy: 0.250 Loss: 1.355264
    INFO:root:[Epoch 0050] Dev Accuracy: 0.375 Loss: 1.346404
    INFO:root:[Epoch 0060] Dev Accuracy: 0.375 Loss: 1.336877
    INFO:root:[Epoch 0070] Dev Accuracy: 0.500 Loss: 1.325287

    		     ... snip ...

    INFO:root:[Epoch 0960] Dev Accuracy: 0.750 Loss: 0.007144
    INFO:root:[Epoch 0970] Dev Accuracy: 0.750 Loss: 0.006988
    INFO:root:[Epoch 0980] Dev Accuracy: 0.875 Loss: 0.006836
    INFO:root:[Epoch 0990] Dev Accuracy: 1.000 Loss: 0.006690
    INFO:root:Training KD Tree for lookup with dimension 8 rows and 4 columns
    INFO:root:Final eval accuracy=1.000000
    
This will create a training plot that looks like [viz_metrics.pdf](viz_metrics.pdf).

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
 * `MarginRankingLoss`
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
1. Understand the structure of the code.
2. Understand the vectorize funtion.
3. Write DAN model initialization `__init__` of `DanModel`: replace `self.network = None` with a real network.
4. Write model forward function.
5. Write the model training/testing function in `batch_step`. We don't have unit tests for this part, but to get reasonable performance, it's necessary to get it correct.
6. Write the evaluation code `number_errors` that counts up how many examples you got right.

Optional things that might help
--------------------

1. Change the initialization.  Don't cheat to hand-set parameters to the "right" answer, but you might want to use something like an identity initialization (provided, but only works for specific network configurations).  Your initialization can't depend on knowing the identity of words (in other words, the embedding initialization must always be random).
2. Swap out the optimizer
3. Play around with network widths

Inspecting Training
--------------------

To help you debug and to inspect how the representations evolve, we provide a
utility to plot the internal representations.

    ./venv/bin/python3 -i dan_guesser.py --secondary_questions mini-dev --questions mini-train --dan_guesser_plot_viz train --dan_guesser_hidden_units 2 --dan_guesser_num_workers 1 --dan_guesser_num_epochs 50 --dan_guesser_embed_dim 2 --dan_guesser_ans_min_freq=0 --dan_guesser_nn_dropout 0 --dan_guesser_vocab_size 20

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
