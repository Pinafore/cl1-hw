Logistic Regression Redux: Pytorch
=

Overview
--------

In this homework you'll implement a stochastic gradient ascent for
logistic regression and you'll apply it to the task of determining
whether documents are talking about hockey or baseball.  Sound familiar?  It should be!

Indeed, it will be doing exactly the same thing on exactly the same data as the previous homework.  The only difference is that while you had to do logistic regression yourself, this time you'll be able to use Pytorch directly.

What you have to do
----

Coding (15 points):

1. Load in the data and create a data iterator.  This will be the most difficult bit.  You may use the sklearn feature creation functions or you can do it yourself to directly create a matrix.
1. Create a logistic regression model with a softmax/sigmoid
   activation function.  To make unit tests work, we had to initialize
   a member of the SimpleLogreg class.  Replace the none object with
   an appropriate nn.Module.
1. Optimize the function (remember to zero out gradients) and analyze the output.

Analysis (10 points):

1. How does the setup differ from the model you learned "by hand" in terms of initialization, number of parameters, activation?
2. Look at the top features again.  Do they look consistent with your results for the last homework?

What to turn in
-

1. Submit your _pytorch-lr.py_ file (include your name at the top of the source)
1. Submit your _analysis.pdf_ file
    - no more than one page
    - pictures are better than text
    - include your name at the top of the PDF




