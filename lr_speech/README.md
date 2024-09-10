Logistic Regression Redux: Pytorch


Overview
--------

In this homework you'll implement a stochastic gradient **descent** for
logistic regression and you'll apply it to vowel classification, for vowels
spoken in 'hVd' contexts.  The dataset has the following vowels:

   ae, ah, aw, eh, ei, er, ih, iy, oa, oo, uh, uw

and the user will input a list of two of these, separated by a comma.

The dataset is described in Hillenbrand et al. (1995), and is contained
in the 'Hillenbrand' directory, with subdirectories for men, women, and
children's utterances.  You'll be using the librosa package to read in
the wav files and compute 13-dimensional MFCCs.  The basic code for this 
is given.

You'll be running logistic regression on pairs of vowels, in order to
classify them.  Some pairs of vowels (like iy vs. ah) are easy to tell 
apart, and classification accuracy will be at ceiling, whereas other pairs 
of vowels (like eh vs. ae) are more difficult to tell apart.  You should 
expect your regressions to yield test accuracies ranging between 0.5 and 1, 
depending on which pair of vowels you are classifying.


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
   each utterance, you'll z-score each feature across all of the utterances.

3. Create a logistic regression model with a softmax/sigmoid
   activation function.  To make unit tests work, we had to initialize
   a member of the SimpleLogreg class.  Replace the none object with
   an [appropriate nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html).

4. Optimize the function (remember to zero out gradients) in the
   `step` function.
   

Extra credit: 

MFCCs computed at the midpoint frame may not be the best way of 
classifying vowels.  Write an alternative feature extractor function, 
create_alt_dataset, that gets better classification performance on the
pairs of vowels that weren't already at ceiling.


References:

Hillenbrand, J., Getty, L. A., Clark, M. J., and Wheeler, K. (1995). Acoustic characteristics of American English vowels. Journal of the Acoustical Society of America, 97(5):3099– 3111.

