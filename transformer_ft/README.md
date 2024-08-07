Fine-Tuning
=

Overview
--------

In this homework you'll implement a classifier.  This sounds a lot like previous homeworks, but instead of starting from scratch, we'll be starting from a pre-trained language model.

Like the DAN homework, we'll be classifier our quiz bowl questions into the category they're about.  This should be much easier than that homework, though, as you'll be using lots of code, models, and data provided by Huggingface.


What you have to do
----

Coding (30 points):

1. Understand how the code is grabbing a model and data from huggingface
2. Turn one of the columns from the dataset into a label and limit how much data you read in.  **You should really implement this first, as otherwise your debugging process will be really slow.  Please take this advice from me, who wrote the solution much more slowly as a result because I thought: "It's just reading in some data; I can't screw that up, right?  Save yourself an hour and implement that limit keyword argument first thing.**
3. Finetune a classifier on that prediction task
3. Write code to record how well your model is doing.

Analysis (5 points):

1. What is the accuracy of the model as a function of the amount of fine-tuning data?  Make sure to investigate with very little fine-tuning data.

Extra credit (5 points):

1. Improve the accuracy (3 points)
    - You cannot change the base model
    - You can add more data (beyond the QANTA datset) or change the encoding / tokenization 
    - Show the effect in your analysis document
1.  Investigate predicting the page (the Wikipedia page associated with the answer).  You will want to restrict the number of possible answers, but always evaluate against the whole set (e.g., use an UNK label for the pages you exclude).  Show the results of your exploration in the writeup. (1 point)
1.  Visualize the attention weights for examples you got wrong and explain what went wrong (these visualizations don't count against the page limit) (1 point)
    
Caution: When implementing extra credit, make sure your implementation of the
regular algorithms doesn't change.

What to turn in
-

1. Submit your _train_classifier.py_ file (include your name at the top of the source) 
1. Submit your _analysis.pdf_ file
    - no more than one page (NB: This is also for the extra credit.  To minimize effort for the grader, you'll need to put everything on a page.  

Hints
-

**Q.  This homework is just looking up documentation from HuggingFace and StackOverflow and doesn't have any of the clever dynamic programming we did before.  Is this representative of NLP Research today?**

_A.  Yes._

**Q.  Loading the data from Huggingface gives me a dictionary, what's up with that?**

_A.  There's a separate Huggingface Dataset for each fold, and the QANTA dataset has a bunch of folds for training different aspects of the system.  You can get more complicated, but just train on ``guesstrain`` and evaluate on ``guessdev`` to get started._

**Q.  I'm really at a loss of where to start.  Can you give a hint which functions we should use?**

_A.  Look at the imports._

**Q.  Can we use other training data?**

_A. You can use any data you'd like except for the data from a fold that ends in "dev" or "test"._

**Q.  Can I use the category feature in the input?**

_A.  No, you cannot use ``category`` or ``subcategory`` as part of the training data (but of course you can use it as a label)._

**Q.  Can I use a different base model?**

_A.  In the interest of fairness (not everyone has a beefy compute), you can't change the underlying model.  Also, Gradescope only accepts uploads of a given size._

**Q. What if I get a ``IndexError`` while training the model?**

_A. This probably means that you didn't specify all of the labels you'd see; perhaps increase your training size slightly (or remove filters)._

**Q. What if I get a wrong type error?**

_A.  This suggests that you didn't set up the label in the datatype that the model is expecting.  Take a look at ``class_encode_column``._
