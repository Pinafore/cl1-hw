
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
any point by "buzzing in"; if the answer is correct, the team gets
points and the next question is read.  Otherwise, the team loses
points and the other team can answer.

There are several ways of doing a fun and interesting project in this space as we detail below.

Possible Project Ideas
============================

Create a better system for answering questions
----------------------------------------------
*GOAL*: The is most straightforward.  Given some text, predict what the answer is.  If more than a couple teams do this project (or the next), we can create a leaderboard to compare systems on heldout systems.

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
5. Adding additional features for a reranker (https://arxiv.org/abs/2102.03016), potentially using adversarial data (https://sites.google.com/view/qanta/projects/adversarial)

*FIRST STEP*: Train an existing system on new data, analyze how it works on the original task.  For example, train and deploy a DPR / ORQA system on quiz bowl and see how well it does.

*KEY RESULTS*: Accuracy / F1 

Create a better system for knowing when to signal to answer
-----------------------------------------------------------

*GOAL*: Unlike other datasets, a challenging aspect of playing quiz bowl competitively is knowing when your system is confident enough to answer, not just selecting the best answer.

You can improve this by transfer learning going to/from SQuAD 2.0 (https://rajpurkar.github.io/SQuAD-explorer/) or NQ (https://ai.google.com/research/NaturalQuestions), which also include the option to "abstain".

*FIRST STEP*: Train an existing system on new data, analyze how it works on the original task.  E.g. train an abstain classifier on NQ, apply it to Quiz Bowl.

*KEY RESULTS*: Expected wins metric (https://arxiv.org/abs/1904.04792)

Convert between question formats using machine translation
----------------------------------------------------------

Model when an answer would be asked
-----------------------------------
Some questions are timeless: "When was the Magna Carta signed?".  Other questions have a four-year expiration date: "Who won the last world cup?" or "What was the tipping point in the electoral college?".  One preprocessing step might be needing to extract the year the question was asked, it might not be easily extractable from the dataset as is.  Once that's done, we can try to figure out the effect between when questions are asked and the QA pair.

*FIRST STEP*: Compute a distribution over answers, excluding rare ones, and find the ones that are most temporally constrained.  Those that appear only for a brief period, those that appear and persist, and those that stopped being asked.

*NEXT STEPS*: Getting a better understanding of the temporal effects of QA:
1. Create a predictor of when an answer is timely.
2. Create a predictor of when a QA pair depends on when it is asked.
3. Try to improve QA accuracy for those questions.
4. Given news / Wikipedia edits predict what questions / topics will be asked in the future.

Predict the human difficulty of a question
------------------------------------------



More Resources
==================
We are using infrastructure for a public competition:
* [Baseline system](https://github.com/Pinafore/qanta-codalab)
* [Codalab tutorial worksheet](https://worksheets.codalab.org/worksheets/0x26df3339cf734a9cace34abc800defd2/)
* [General Information Webpage](http://qanta.org)

