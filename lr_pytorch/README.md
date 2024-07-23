Logistic Regression Redux: Pytorch
=

Overview
--------

In this homework you'll implement a stochastic gradient **descent** for
logistic regression and you'll apply it to the task of determining
whether documents are talking about hockey or baseball.

What you have to do
----

There are simple unit tests in `tests.py`, make sure those work before moving on to the "real" data.

Setup:

1. You may need to create a virtual environment and install Python
   packages:
   
```
python3 -m venv venv
./venv/bin/pip3 install -r requirements.txt
```

1.  You'll need to grab [the data](https://github.com/Pinafore/cl1-hw/tree/master/logreg) if you don't clone the repository: there are two possible datasets the real one (data) and a tiny one for debugging (toy_text).

Coding (15 points):

1. Load in the data in the `read_dataset` function.  This will be the most
   difficult bit.  You may use the sklearn feature creation functions
   or you can do it yourself to directly create a matrix.
   
   The matrix needs to have a row for each example and a column for
   each word in the vocabulary.  The sample code gives you an empty
   matrix of the correct size, you just need to fill it in with the
   appropriate values.
   
1. Create a logistic regression model with a softmax/sigmoid
   activation function.  To make unit tests work, we had to initialize
   a member of the SimpleLogreg class.  Replace the none object with
   an [appropriate nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).
1. Optimize the function (remember to zero out gradients) in the
   `step` function. 
1. Finish the `inspect` function to get the features that did the best
   job of predicting the label of the documents.

