Language Models
=

As always, check out the Github repository with the course homework templates:

http://github.com/ezubaric/cl1-hw.git

The code for this homework is in the _lm_ directory.  This assignment is worth 40 points.

Preparing Data (10 points)
---
 
We will use the Brown corpus (nltk.corpus.brown) as our training set and the Treebank (nltk.corpus.treebank) as our test set.  Eventually, we'll want to build a language model from the Brown corpus and apply it on the Treebank corpus.  First, however, we need to prepare our corpus.
* First, we need to collect word counts so that we have a vocabulary.  This is done by the _train\_seen_ function.  Modify this function so that it will keep track of all of the tokens in the training corpus and their counts.
* After that is done, you can complete the _vocab\_lookup_ function.  This should return a unique identifier for a word, or a common "unknown" identifier for words that do not meet the _unk\_cutoff_ threshold.  You can use strings as your identifier (e.g., leaving inputs unchanged if they pass the threshold) or you can replace strings with integers (this will lead to a more efficient implementation).  The unit tests are engineered to accept both options.
* After you do this, then the finalize and censor functions should work (but you don't need to do anything).  But check that the appropriate unit tests are working correctly.

Estimation (20 points)
------

After you've finalized the vocabulary, then you need to add training
data to the model.  This is the most important step!  Modify the
_add\_train_ function so that given a sentence it keeps track of the
necessary counts you'll need for the probability functions later.  You
will probably want to use default dictionaries or probability
distributions.  Finally, given the counts that you've stored in
_add\_train_, you'll need to implement probability estimates for
contexts.  These are the required probability estimates you'll need to
implement:
* _mle_: Simple division of counts for that observation by total counts for the context
* _laplace_: Add one to all counts
* _dirichlet_: Add a specified parameter greater than zero to all counts
* _jelinek_mercer_: Interpolate between probability distributions with parameter lambda

Now if you run the main section of the _language\_model_ file, you'll
get per-sentence reports of perplexity.  Take a look at what sentences
are particularly hard or easy (you don't need to turn anything in
here, however).

Exploration (10 points)
----------

Try finding sentences from the test dataset that get really low perplexities for each of the estimation schemes (you may want to write some code to do this).  Can you find any patterns?  Turn in your findings and discussion as \texttt{discussion.txt}.

Extra Credit
------

Extra Credit (make sure they don't screw up required code / functions that will be run by the autograder):
* _kneser\_ney_: Use discounting and prefixes with discount parameter $\delta$ and concentration parameter alpha to implement interpolated Kneser-Ney.
* Implement a function to produce English-looking output (return an iterator or list) from your language model (function called _sample_)
* Make the code really efficient for reading in sequences of characters

FAQ
--------

*Q: What if the counts of \<s\> and \<\/s\> fall below the threshold?*

A: They should always be included in the vocabulary.
