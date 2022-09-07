Deep Learning 
=

Due: 8 March 2019 (23:55)


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

The data is sampled from quiz bowl bonus question. We tokenize the questuion and split them into train/dec/test set.
Each example includes the question text and the label (0 for Literature, 1 for History and 2 for Science). 

Pytorch data loader
----------------

In this homework, we use pytorch build-in data loader to do data mini-batching, which provides single or multi-process iterators over the dataset(https://pytorch.org/docs/stable/data.html).

For data loader, there includes two functions, batichfy and vectorize. For each example, we need to vectorize the question text in to a vector using vocabuary. In this assignment, you need to write the vectorize function yourself. Then we provide the batchify function to split the dataset into mini-batches. 





What you have to do
----------------

Coding: (35 points)
1. Understand the structure of the code.
2. Write the data vectorize funtion.
3. Write DAN model initialization. 
4. Write model forward function.
5. Write the model training/testing function. We don't have unit test for this part, but to get reasonable performance, it's necessary to get it correct.

Analysis: (5 points)
1. Report the accuracy of test set. (Should easily get above 0.8 for category prediction, answer prediction is trickier but please report the things you tried)
2. Look at the dev set, give some examples and explain the possible reasons why these examples are predicted incorrectly. 


Pytorch install
----------------
In this homework, we use CPU version of Pytorch 0.4.1(latest version)

You can install it using the following command(linux):
conda install pytorch torchvision -c pytorch or conda install pytorch=0.4.1 torchvision -c pytorch

If you are using macos or windows, check the pytorch website for installation



For more information, check
https://pytorch.org/get-started/locally/

Extra Credit
----------------

(Please code extra credit part separately, not for submission to submit server) For extra credit, you need to initialize the word representations with word2vec,
GloVe, or some other representation.  Compare the final performance
based on these initializations *and* see how the word representations
change. Write down your findings in analysis.pdf.

What to turn in 
----------------

1. Submit your dan.py file
2. Submit your analysis.pdf file 

    No more than one page 
    
    Include your name at the top of the pdf


