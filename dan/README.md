Deep Learning 
=

Overview
--------

For gaining a better understanding of deep learning, we're going to be
looking at deep averaging networks.  These are a very simple
framework, but they work well for a variety of tasks and will help
introduce some of the core concepts of using deep learning in
practice.

In this homework, you'll use pytorch to implement a DAN classifier for determining the answer to a quizbowl question (a minor switch on lines 41-42 allows change to the much simpler task of predicting the category of a quizbowl question). 

You'll turn in your code on the submit server. This assignment is worth 40 points.

Dataset
----------------

The data are meant to resemble the question answering dataset we used in a previous homework, but much simpler.

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

This is because there are only four answers in the data, and the four words combine to signal what the answer is.  After averaging the data, the four quadrants represent the answer space.

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

Pytorch data loader
----------------

In this homework, we use pytorch build-in data loader to do data
mini-batching, which provides single or multi-process iterators over the
dataset(https://pytorch.org/docs/stable/data.html).

For data loader, there includes two functions, batichfy and vectorize. For
each example, we need to vectorize the question text in to a vector using
vocabuary. In this assignment, you need to write the vectorize function
yourself. Then we provide the batchify function to split the dataset into
mini-batches.


What you have to do
----------------

Coding:
1. Understand the structure of the code.
2. Write the data vectorize funtion.
3. Write DAN model initialization. 
4. Write model forward function.
5. Write the model training/testing function. We don't have unit test for this part, but to get reasonable performance, it's necessary to get it correct.

Inspecting Training
--------------------

To help you debug and to inspect how the representations evolve, we provide a
utility to plot the internal representations.

  ./venv/bin/python3 -i dan_guesser.py --secondary_questions mini-dev --questions mini-train --dan_guesser_plot_viz train.pdf --dan_guesser_hidden_units 2 --dan_guesser_num_workers 1 --dan_guesser_num_epochs 50 --dan_guesser_embed_dim 2 --dan_guesser_ans_min_freq=0 --dan_guesser_nn_dropout 0 --dan_guesser_vocab_size 20

Will show you the evolution in a file called ``train.pdf``.

Pytorch install
----------------
For more information, check
https://pytorch.org/get-started/locally/

Extra Credit (Up to 15 points)
----------------

We will have a separate leaderboard submission for this part of the code.  

For extra credit, you need to get this architecture working on answering real questions.  Here are some things you are allowed to do:

 * initialize with word representations like GloVe or word2vec
 * change how negative examples are selected or the loss function
 * change the nonlinearity, dropout, or optimization
 * change the number of DAN layers
 * change the dimensionality

You cannot:

 * change how the guesser finds similar questions
 * change the general structure of the model (average -> nonlinearity)


What to turn in 
----------------

1. Submit your dan_guesser.py file
