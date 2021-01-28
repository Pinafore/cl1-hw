Sequence Tagging (40 Points)
================

Introduction
----------------------
In this homework, we will implement a named entity tagging model
using the structured perceptron algorithm.
The homework has three required parts and an open-ended extra credit part
that aims at improving the accuracy of the tagger.

The task is to assign a tag to each word in a sentence.
The tag determines whether a word is the beginning of a 
named entity (e.g. `B-LOC`, `B-ORG`), an inside part of an entity
(e.g. `I-LOC`, `I-ORG`) or is not part of any entity `O`.
The tag also determines the type of the entity: person, location,
organization .. etc. Here is an example

`Sentence: Adobe opens a new office in College Park`


`Tags: B-ORG O O O O O B-LOC I-LOC`

All your implementation should be in `Named_Entity_Tagger.py`.
For the extra credit part, feel free to write separate files and import
them in `Named_Entity_Tagger.py`, but make sure you include
them in your submission.


Feature Vector (10 points)
---------------------------

We define a basic set of features: 1) tag x following tag y
and 2) word x is assigned tag y.

```
for ii in self._tags:
    for jj in self._tags:
        self._feature_ids[(ii, jj)] = self._num_feats
        self._num_feats += 1

for word in vocabulary:
    for tag in tag_set:
        self._feature_ids[(tag, word)] = self._num_feats
        self._num_feats += 1

```

Your first task is to implement the `feature_vector` method
that takes a sentence and a sequence of tags and generates
a feature vector based on the features defined above.

Structured perceptron update (10 points)
-----------------------------------------

The second task is to implement the update function
of the structured perceptron. Given a sentence, a 
predicted tag sequence and the gold tag sequence,
update the vector `w` in the `update` method.

Decoding the most likely tag sequence (20 points)
-----------------------------------------

The third task is to implement the `decode` method.
Given a set of weights (trained model) and a new sentence,
find the most likely tag sequence. 

Extra Credit -- NER on CoNLL 2003 (10 Points)
---------------------------------------------

Now that you implemented the structured perceptron algorithm
with basic features, let's push its accuracy on a realistic benchmark dataset.
We will work with the dataset of CoNLL 2003 shared task. The dataset
has four types of named entities: persons, locations, organizations and names of miscellaneous entities.
The extra credit part is about improving the accuracy of the structured
perceptron on CoNLL 2003 dataset by devising a new set of features.
Before defining your features, make sure you set `only_basic_features` to
`False` when calling `TaggingPerceptron`.

Submit a file `extra_credit.txt` that describes the features you tried,
the intuition behind each of them and how they affected the accuracy (increased or decreased).

