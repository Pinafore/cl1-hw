
About the Data
==============

Quiz bowl is an academic competition between schools in
English-speaking countries; hundreds of teams compete in dozens of
tournaments each year. Quiz bowl is different from Jeopardy, a recent
application area.  While Jeopardy also uses signaling devices, these
are only usable after a question is completed (interrupting Jeopardy's
questions would make for bad television).  Thus, Jeopardy is a
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
any point by "buzzing in"; if the answer is correct, the team gets
points and the next question is read.  Otherwise, the team loses
points and the other team can answer.

There are several ways of doing a fun and interesting project in this space as we detail below.

Possible Project Ideas
============================

Create a better system for answering questions
----------------------------------------------

*GOAL*: The is most straightforward.  Given some text, predict what
 the answer is.  If more than a couple teams do this project (or the
 next), we can create a leaderboard to compare systems on heldout
 systems.

You are welcome to use any *automatic* method to choose an answer.  It
need not be similar nor build on our provided systems.  In addition to
the data we provide, you are welcome to use any external data *except*
our test quiz bowl questions (i.e., don't hack our server!).  You are
welcome (an encouraged) to use any publicly available software, but
you may want to check on Piazza for suggestions as many tools are
better (or easier to use) than others.

This can be through:
1. Adding additional data (https://www.gutenberg.org/) (https://wikis.fandom.com/wiki/List_of_Wikia_wikis)
2. Using better retrieval systems (https://github.com/facebookresearch/DPR)
3. Trying to train on multiple datasets at once (https://arxiv.org/abs/1905.13453)
4. Using better methods to find the answer span
5. Adding additional features for a reranker
   (https://arxiv.org/abs/2102.03016), potentially using adversarial
   data (https://sites.google.com/view/qanta/projects/adversarial)
6. Improving the calibration

*FIRST STEP*: Train an existing system on new data, analyze how it
 works on the original task.  For example, train and deploy a DPR /
 ORQA system on quiz bowl and see how well it does.

*KEY RESULTS*: What is unique about this competition is that its [metric](https://drive.google.com/file/d/1byJ0_HYFBa-4y6SWHMf5JYC_cshE2JeG/view) tests
 now just how often the guess is correct but also how well the system
 assigns a probability to its guess being correct.

Create a better system for knowing when to signal to answer
-----------------------------------------------------------

*GOAL*: Unlike other datasets, a challenging aspect of playing quiz
 bowl competitively is knowing when your system is confident enough to
 answer, not just selecting the best answer.

You can improve this by transfer learning going to/from SQuAD 2.0
(https://rajpurkar.github.io/SQuAD-explorer/) or NQ
(https://ai.google.com/research/NaturalQuestions), which also include
the option to "abstain".

*FIRST STEP*: Train an existing system on new data, analyze how it
 works on the original task.  E.g. train an abstain classifier on NQ,
 apply it to Quiz Bowl.

*KEY RESULTS*: Expected wins metric (https://arxiv.org/abs/1904.04792)
 for quiz bowl, abstention F1 for other datasets.

Convert between question formats using machine translation
----------------------------------------------------------

Many different QA datasets are similar, but have slightly differences
in phrasing and resources.  For example:
*Jeopardy*: A state since the 1700s but not in the original 13, it
ends with its own 2-letter postal abbreviation
*Natural Questions* / *PAQ* (https://github.com/facebookresearch/PAQ): What state postal code KY?
*Quiz Bowl*: For ten points, the Mammoth Cave system in what U.S. state is the longest cave in the world?

Being able to convert between datasets or generate new data that looks
like a datset has multiple uses:
* Providing additional training data for machines
* Getting around copyright restrictions
* Providing additional training data for humans (so they can practice
playing Jeopardy! or Quizbowl)

*FIRST STEP (Generation)*: Given a dataset, treat (evidence, question)
 pairs as MT data and train a simple MT system
 (https://github.com/joeynmt/joeynmt) to take the evidence as input
 and generate the question.  If your dataset doesn't have evidence, do
 a search over Wikipedia.

*FIRST STEP (Conversion)*: Given a dataset, match its questions with
 another question (with the same answer) in another data.  An easy way
 of doing this would be to have a threshold on word overlap.  Then use
 those as MT pairs with a simple MT system.

*NEXT STEPS*: Generating better examples, seeing if they're useful.
1. Include evidence during conversion.
2. See if systems / humans can answer the generated questions (they
   should) or can tell if they're generated automatically (they shouldn't).
3. If you train a system on the original domain with synthetic
question, does accuracy improve?

*KEY RESULTS*: Quality of generated questions / improvement in QA
 accuracy using augmented training data.

More Resources
==================
We are using infrastructure for a public competition:
* [Competition](https://sites.google.com/view/qanta/2024-competition/for-computer-teams)
* [Tutorial](https://colab.research.google.com/drive/1bCt2870SdY6tI4uE3JPG8_3nLmNJXX6_?usp=sharing)
* [Leaderboard](https://huggingface.co/spaces/umdclip/advcalibration)
* [General Information Webpage](http://qanta.org)
