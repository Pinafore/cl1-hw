The goal of this project is to predict whether the answer to a trivia
question is correct or not (these are the same questions you classified in an earlier homework).  Thus, it is a binary classification
problem.  You will have to train effective classification models, but
the more important (and effective) route for success is to engineer
features and gather additional data to help your predictions.

About the Data
==============

Quiz bowl is an academic competition between schools in
English-speaking countries; hundreds of teams compete in dozens of
tournaments each year. Quiz bowl is different from Jeopardy, a recent
application area.  While Jeopardy also uses signaling devices, these
are only usable after a question is completed (interrupting Jeopardy's
questions would make for bad television).  Thus, Jeopardy is rapacious
classification followed by a race---among those who know the
answer---to punch a button first.

Here's an example of a quiz bowl question:

Expanding on a 1908 paper by Smoluchowski, he derived a formula for
the intensity of scattered light in media fluctuating densities that
reduces to Rayleigh's law for ideal gases in The Theory of the
Opalescence of Homogenous Fluids and Liquid Mixtures near the Critical
State.  That research supported his theories of matter first developed
when he calculated the diffusion constant in terms of fundamental
parameters of the particles of a gas undergoing Brownian Motion.  In
that same year, 1905, he also published On a Heuristic Point of View
Concerning the Production and Transformation of Light.  That
explication of the photoelectric effect won him 1921 Nobel in Physics.
For ten points, name this German physicist best known for his theory
of Relativity.

*ANSWER*: Albert _Einstein_

Two teams listen to the same question. Teams interrupt the question at
any point by "buzzing in"; if the answer is correct, the team gets
points and the next question is read.  Otherwise, the team loses
points and the other team can answer.

Why we want to use Quiz Bowl Data for Classification
-----------------------------------------------

It's very easy to generate guesses (in fact, we could generate every
possible guess).  The challenge is knowing whether any given guess is
good or not.  We can treat this as a classification problem.  Every
guess can be described by features that measure how well it matches
the question.  The classifier tells us whether we got the question
wrong or right.

Data Format
--------------------

Each line has a guess (page) and a correct answer (answer) given some
fraction of the question revealed so far (text).  Your goal is to
predict whether they match.  Each guess is the title of a Wikipedia
page.  To get you started, you have the following columns:


Data for History and Literature questions will be available October 24.  Data for all other categories of questions will be available November 1.  

Competition
==================

We will use Kaggle InClass for this competition.  This will be a competition between students in the Colorado and Maryland graduate courses on natural language processing and computational linguistics.  A large portion of your grade will be how you perform on this Kaggle competition.  

Proposal
==================

The project proposal is due Nov. 1.  This one page PDF document should describe:
* Who is on your team
* What techniques you will explore 
* Your timeline for completing the project (be realistic!)

Final Presentation
======================

The final presentation will be in class on Dec. 16.  In the final presentation you will:
* Explain what you did
* Who did what
* What challenges you had
* How well you did (based on the Kaggle competition)
* An error analysis.  An error analysis must contain examples from the development set
that you get wrong.  You should show those sentences and explain why
(in terms of features or the model) they have the wrong answer.  You should have been doing this all along as your derive new features
(e.g., 2b), but this is your final inspection of your errors. The
feature or model problems you discover should not be trivial features
you could add easily.  Instead, these should be features or models
that are difficult to correct.  An error analysis is not the same thing as simply presenting the error
matrix, as it does not inspect any individual examples.
* The linguistic motivation for your features.  This is a computational linguistics class, so you should give precedence to features / techniques that we use in this class (e.g., syntax, morphology, part of speech, word sense, etc.).  Given two features that work equally well and one that is linguistically motivated, we'll prefer the linguistically motivated one.

