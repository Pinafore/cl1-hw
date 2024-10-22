
This homework is about dependency parsing.  We'll create a classifier to create a shift-reduce parser.


Data
===========

First, we’ll need some dependency parsed sentences.  NLTK has a small dataset available:


    >>> from nltk.corpus import dependency_treebank
    >>> s = dependency_treebank.parsed_sents()[0]
    >>> for ii in s.to_conll(4).split("\n"): print(ii)
    ...
    Pierre        NNP        2        
    Vinken        NNP        8        
    ,        ,        2        
    61        CD        5        
    years        NNS        6        
    old        JJ        2        
    ,        ,        2        
    will        MD        0        j
    join        VB        8        
    the        DT        11        
    board        NN        9        
    as        IN        9        
    a        DT        15        
    nonexecutive        JJ        15        
    director        NN        12        
    Nov.        NNP        9        
    29        CD        16        
    .        .        8

What to do
============

1. Given a DependencyGraph object, create a method (transition_sequence) that produces a series of shift-reduce moves (instances of the Transition object) that produces the tree.  You’ll also want to generate (or otherwise reconstruct) the buffer and stack.
1. Given a series of shift-reduce moves and an input sentence, produce the vector of governing heads (implement the function parse_from_transition).  In practice, this means creating a new dependency parse.
1. Generate feature vectors for your transitions.  Start with something simple!
1. Given a set of transitions, train a MaxEnt classifier to produce the same moves using the feature set.  (Use “IIS” first for the algorithm - I had trouble with others.)
1. Report your classifier accuracy and on the test / train split produced by oracle.py for both versions.


Extra Credit
===============

1. Implement this the RULES baseline and report its accuracy on this dataset

https://aclanthology.org/W12-1910.pdf

2.  Do feature engineering to improve the accuracy.  (You cannot change the classifier or dataset.)

3.  Complete the `sentence_attachment_accuracy`
