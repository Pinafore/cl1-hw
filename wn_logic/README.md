

What the Homework is About
=====================

There's a game called twenty questions where you need to guess what
someone is thinking of within twenty questions, and you can only ask
yes or no questions.

But when humans play the game, the questions are fairly simple
... people get pretty annoyed if you ask a question like "Is this a
dog, cat, elephant, or bagel?"

This is called "yes and no" in the UK, "Was bin ich" in Germany, and
probably many other things around the world.

But we're going to allow you to ask any yes or no question you like as
long as it can be expressed in a logical form.

How to Ask Questions
=====================

You'll have to ask questions in a logical form.  For full generality, you'll be able to pose questions in Conjunctive Normal Form (CNF).  The basic building blocks of a CNF are an OR of individual predicates in a clause.  All of the clauses are combined by an AND.

> If your thesis is utterly vacuous
> 
> Use first-order prediate calculus
> 
> With sufficient formality
> 
> The sheerest banality
> 
> Will be hailed by the critics: "Miraulous!"


Thus, if you want to know if the concept is either a dog or a cat, you can form this as:

    oracle.cnf_eval([["hypernyms", "hypernyms"]], [[wn.synset("dog.n.01"), wn.synset("cat.n.01"))]])
    
We express the statement as a list of lists.  Each inner list is a clause, and the whole list contains all the clauses.  So to add another clause to check whether the synset also has tail, you'd do:

    oracle.cnf_eval([["hypernyms", "hypernyms"], ["part_meronyms"]], [[wn.synset("dog.n.01"), wn.synset("cat.n.01"))], [wn.synset('flag.n.07')]])

What's the best question to ask?
================================

Ideally, you should cut in half all of possible concepts that could be the answer with each question.  So, since there are fewer than 65 thousand nouns in WordNet, in theory, you should be able to guess the concept in fewer than 16 questions.

What are the Possible Answers
=====================




