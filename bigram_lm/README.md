Language Models
=

As always, check out the Github repository with the course homework templates:

http://github.com/ezubaric/cl1-hw.git

The code for this homework is in the _bigram lm_ directory.  This assignment is worth 40 points.

Preparing Data (Done for you, but extra credit opportunity)
---
 
We will use the Brown corpus (nltk.corpus.brown) as our training set and the
Treebank (nltk.corpus.treebank) as our test set.  Eventually, we'll want to
build a language model from the Brown corpus and apply it on the Treebank
corpus.  First, however, we need to prepare our corpus.

* First, we need to collect word counts so that we have a vocabulary.  This is
  done by the _train\_seen_ function.  Modify this function so that it will
  keep track of all of the tokens in the training corpus and their counts.
* After that is done, you can complete the _vocab\_lookup_ function.  This
  should return a unique identifier for a word, or a common "unknown"
  identifier for words that do not meet the _unk\_cutoff_ threshold.  You can
  use strings as your identifier (e.g., leaving inputs unchanged if they pass
  the threshold) or you can replace strings with integers (this will lead to a
  more efficient implementation).  The unit tests are engineered to accept
  both options.  **Your extra credit opportunity is to change the token representation to integers.**
* After you do this, then the finalize and censor functions should work (but
  you don't need to do anything).  But check that the appropriate unit tests
  are working correctly.


What's the Vocab
---------------

Now, we could do this more efficiently without multiple passes, but this will
make debugging simpler.  After the first two steps, to let the code know that
we're done, we'll call the `finalize` function to tell the code to not let
that change any more.  After that point, we won't be able to tell the code
that we've seen new words or new documents.

So the very first step of this process is figure out what goes into the vocab.
Simple, right?

Now, of course there are some complications.  
  
First complication: what if after we've seen a new word that wasn't in the
vocabulary?  Anything that isn't mapped to the vocabulary will then
become the "unknown token" (`kUNK` in the code).
 
That leads to a second complication: we need to compute statistics for
   how often contexts have unknown words.  If we add every single
   word in our training set to the vocabulary, then there won't be any
   unknown words and thus no statistics about unknown words.

So what do unknown words look like?  Think about
[Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law).  There are very few frequent
words but many infrequent words.  So we're likely to have most of the frequent
words in our vocabulary.  That means we're missing infrequent words.

So what words that we have seen
will look most like the unknown words that we'll see in the future?
Our least frequent words!  So we'll use the ``unk_cutoff`` argument to
turn all of the words that we initially saw into the unknown token
``kUNK = "<UNK>"``.

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

Try finding sentences from the test dataset that get really low perplexities
for each of the estimation schemes (you may want to write some code to do
this).  Can you find any patterns?  Turn in your findings and discussion as
\texttt{discussion.txt}.

Extra Credit
------

Extra Credit (make sure they don't screw up required code / functions that
will be run by the autograder):

* _kneser\_ney_: Use discounting and prefixes with discount parameter $\delta$
  and concentration parameter alpha to implement interpolated Kneser-Ney.

* Improve the word representation to make it more efficient.

* Implement a function to produce English-looking output (return an iterator
  or list) from your language model (function called _sample_) Make the code
  really efficient for reading in sequences of characters

* Find good parameters for various estimation techniques that optimizes
  held-out perplexity.

FAQ
--------
*Q: Why are there two passes over the data?*

A: The first pass establishes the vocabulary, the second pass accumulates the counts.  You could in theory do it in one pass, but it gets much more complicated to implement (and to grade).

*Q: What if the counts of \<s\> and \<\/s\> fall below the threshold?*

A: They should always be included in the vocabulary.

*Q: And what about words that fall below the threshold?*

A: They must also be represented, so the vocab size will be the number of tokens at or above the UNK threshold plus three (one for UNK, one for START, and one for END).  

*Q: What happens when I try to take the log of zero?*

A: Return kNEG\_INF instead.

*Q: Do I tune the hyperparameters for interpolation, discount, etc.?*

A: No, that's not part of this assignment.
