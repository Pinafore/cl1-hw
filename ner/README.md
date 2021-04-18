Entities
=

Overview
--------

Homework adapted from a [homework at UPenn](http://computational-linguistics-class.org/homework/ner/ner.html).

Named Entity Recognition (NER) is the task of finding and classifying named entities in text. This task is often considered a sequence tagging task, like part of speech tagging, where words form a sequence through time, and each word is given a tag. Unlike part of speech tagging however, NER usually uses a relatively small number of tags, where the vast majority of words are tagged with the ‘non-entity’ tag, or O tag.

NER is useful as a preprocessing step for doing QA.  Given raw text, find the people and places in the question, and link them to knowledge resources like Wikipedia (this step is "entity linking" as discussed in lecture.

Your task is to implement your own named entity recognizer, the first step of the process. You will implement an entity tagger using scikit learn, filling out the stub that we give you. There will be a leaderboard on Gradescope.

As with nearly all NLP tasks, you will find that the two big points of variability in NER are (a) the features, and (b) the learning algorithm, with the features arguably being the more important of the two. The point of this assignment is for you to think about and experiment with both of these. Are there interesting features you can use? What latent signal might be important for NER? What have you learned in the class so far that can be brought to bear?

Get a headstart on common NER features by looking at [Figure 17.5](https://web.stanford.edu/~jurafsky/slp3/17.pdf) in the textbook.

Typical features for a feature-based NER system:
* What are the neighboring words
* Embeddings of this word or neighboring words
* Part of speech for neighboring words
* Does the word appear in a gazetteer
* The word has a particular prefix or suffix
* The word shape of the word (Initial Capital, ALL UPPER CASE, CamelCase)

The file ner.py has a really simple implementation of a feature-based perceptron.  You are free to use any features you wish, but please stick with models implemented in sklearn (although you can use features like word embeddings). One of the goals here is to get you thinking about old-fashioned feature engineering.

The file conlleval.py tells you the score you get.

The data we use comes from the Conference on Natural Language Learning (CoNLL) 2002 shared task of named entity recognition for Spanish and Dutch, and it's nicely packaged up by the nltk package ([Advice on installing NLTK](http://www.nltk.org/install.html)). [The introductory paper to the shared task](http://www.aclweb.org/anthology/W02-2024) will be of immense help to you, and you should definitely read it. [You may also find the original shared task page](https://www.clips.uantwerpen.be/conll2002/ner/) helpful. We will use the Spanish corpus (although you are welcome to try out Dutch too).

The tagset is:
* `PER`: for Person
* `LOC`: for Location
* `ORG`: for Organization
* `MISC`: for miscellaneous named entities

The data uses BIO encoding (called IOB in the textbook), which means that each named entity tag is prefixed with a `B-`, which means beginning, or an `I-`, which means inside. So, for a multiword entity, like “James Earle Jones”, the first token “James” would be tagged with `B-PER`, and each subsequent token is `I-PER`. The `O` tag is for non-entities.

We strongly recommend that you study the training and dev data (no one’s going to stop you from examining the test data, but for the integrity of your model, it’s best to not look at it). Are there idiosyncracies in the data? Are there patterns you can exploit with cool features? Are there obvious signals that identify names? For example, in some Turkish writing, there is a tradition of putting an apostrophe between a named entity and the morphology attached to it. Thus, in Turkish, a feature of `isApostrophePresent()` goes a long way. Of course, in English and several other languages, capitalization is a hugely important feature. In some African languages, there are certain words that always precede city names.

You will be glad to hear that the data is a mercifully small download. See the NLTK data page for for download options, but one way to get the conll2002 data is:
    $ python -m nltk.downloader conll2002

Evaluation
---

There are two common ways of evaluating NER systems: phrase-based, and token-based. In phrase-based, the more common of the two, a system must predict the entire span correctly for each name. For example, say we have text containing “James Earle Jones”, and our system predicts `[PER James Earle] Jones`. Phrase-based gives no credit for this because it missed “Jones”, whereas token-based would give partial credit for correctly identifying “James” and “Earle” as `B-PER` and `I-PER` respectively. We will use phrase-based to report scores.

The output of your code must be word gold pred, as in:

    La B-LOC B-LOC
    Coruña I-LOC I-LOC
    , O O
    23 O O
    may O O
    ( O O
    EFECOM B-ORG B-ORG
    ) O O
    . O O
	
Here’s how to get scores (assuming the above format is in a file called results.txt):

    # Phrase-based score
    $ python conlleval.py results.txt

Please create this output for the training set (as train_results.txt), development set as (dev_results.txt), and test set (as test_results.txt). You can retrieve the sentences with the following code:

    train_sents = list(conll2002.iob_sents('esp.train'))
    dev_sents = list(conll2002.iob_sents('esp.testa'))
    test_sents = list(conll2002.iob_sents('esp.testb'))

(The python version of conlleval doesn’t calculate the token-based score, but if you really want it, you can use the [original perl version](https://www.clips.uantwerpen.be/conll2000/chunking/output.html). You would use the -r flag.)

Baselines
--------

The version we have given you gets about 49% F1 right out of the box. Some very simple modifications gets it to 60%. The threshold we ask you to beat is 65%, with partial credit available. The state of the art on the Spanish dataset is over 90%. If you manage to beat that, then look for conference deadlines and start writing, because you can publish it.

To earn full marks on this assignment, demonstrate that you have thought about the problem carefully, and come up with solutions beyond what was strictly required. This is a very open-ended homework and we hope you take advantage of that to get out of your comfort zone and experiment.

Report
----------

Explain the features you added for NER, why you expected them to help, and how they affected your performance. Include a table detailing the change in F1-score as a result of adding each feature or set of features.

Explain the different types of models you experimented with, how they performed, and which you chose for your final model. Include a table comparing the scores of different models. For each model, be sure to tune your parameters on the dev set (optimize your performance with regards to dev F1-score) and include tables recording the training F1-score and dev F1-score attained for each set of parameters. You will also need to submit your final, trained model. We will be using your trained model to confirm that the .txt files you submit are the same as the output of your final model. You can save your model or load your model in the following way:

    import pickle
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    pickle.dump(model, open('model.pkl', 'wb'))
    
    loaded_model = pickle.load(open(filename, 'rb'))

Using your best performing model, do some error analysis (a necessary skill for any researcher to have!) and determine what types of mistakes your model seems to be making. Some things you can think about are in what cases the mistakes are typing issues (i.e. predicting `ORG` instead of `LOC`) vs. span issues (i.e. predicting `B-LOC` when it should be `I-LOC`), and whether those correlate with certain POS tags or contexts. A thoughtful analysis with cited examples should easily get full points for this part of the report.

Please limit your report to two pages.

Deliverables
---------

Here are the deliverables that you will need to submit:
* Code (`ner.py`), as always, in Python 3.
* Saved model[`model.pkl`] - Your final trained model. 
* Results (in files called `train_results.txt`, `dev_results.txt`, `test_results.txt`)
* PDF Report (called writeup.pdf)
