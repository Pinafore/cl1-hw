Overview
=======

In this homework you'll use logic to search for a hidden concept.

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
    
While this is very flexible, it's a little cumbersome.  So you can
also use the functions ``for_all`` and ``there_exists`` to check
whether a particular set of synsets or lemmas have a specificed
connection to the mystery concept.

What you have to do
==============

I've implemented a really inefficient depth-first search to find the
target synset.  Clearly, this is a really bad way to do it.  On the
version of WN on my computer, this took over 18000 steps to get to the
solution.

        self.check_lemma(oracle, wn.synset('entity.n.01'))

        while not any(self._searched.values()):
            previously_searched = list(self._searched.keys())
            for parent in previously_searched:
                for candidate in parent.hyponyms():
                    if not candidate in self._searched:
                        self.check_lemma(oracle, candidate)
						
You will need to replace this code to make the code get to the goal
concept more quickly.

How is Homework Graded
==================

This homework is worth 40 points.

Unlike other homeworks, where you need to implement some basic
functionality, you need to make the code faster.  This homework will
also have a leaderboard to compare your efficiency to other students.

You can get extra points by doing particularly well compared to other
students.

If you can find the goal concept in under 100 steps, you will get full
credit for the assignment.

What's the best question to ask?
================================

Ideally, you should cut in half all of possible concepts that could be the answer with each question.  So, since there are fewer than 65 thousand nouns in WordNet, in theory, you should be able to guess the concept in fewer than 16 questions.

What are the Possible Answers
=====================



FAQ
===

*Q: Can't I just look at the synset?*

_A: We haven't mangled the synset member of the oracle, but because you have to turn in your code, we will verify that you're not using the synset in your search algorithm._


