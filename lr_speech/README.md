Logistic Regression Redux: Pytorch


Overview
--------

In this homework you'll implement a stochastic gradient **descent** for
logistic regression and you'll apply it to vowel classification, for vowels
spoken in 'hVd' contexts.  The dataset has the following vowels:

   ae, ah, aw, eh, ei, er, ih, iy, oa, oo, uh, uw

and the user will input a list of two of these, separated by a comma.


What you have to do
----

Setup:

1. You may need to create a virtual environment and install Python
   packages:
   
```
python3 -m venv venv
./venv/bin/pip3 install -r requirements.txt
```

Coding:

1. Create a dataset in the `create_dataset` function.  You'll be extracting 
   MFCCs from each wav file and taking the middle frame of the utterance as 
   your feature vector.  The matrix needs to have a row for each utterance 
   and a column for each feature.  Once you've extracted the middle frame of
   each utterance, you'll z-score

2. Create a logistic regression model with a softmax/sigmoid
   activation function.  To make unit tests work, we had to initialize
   a member of the SimpleLogreg class.  Replace the none object with
   an [appropriate nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).

3. Optimize the function (remember to zero out gradients) in the
   `step` function. 


Extra credit: 

MFCCs computed at the midpoint frame may not be the best way of 
classifying vowels.  Write an alternative feature extractor function, 
create_alt_dataset, that gets better classification performance on the
pairs of vowels that weren't already at ceiling.
