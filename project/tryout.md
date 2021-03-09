QA for Speech Input
-----------------------------------

This paper makes an argument for improving the underlying algorithms for ASR for QA (https://arxiv.org/abs/2102.08345).  To do this realistically, we need a more comprehensive dataset.

The big picture is that we need to create an incentive system to get people to read questions out loud, so we need an overall system to read questions out loud for fun so people can answer them.  The cost of admission is that to hear questions (and have fun) they need to record a few questions of their own.

*FIRST STEP*: Update a server (https://github.com/ihsgnef/qb_interface) to do the following:
1. Upload a sound file for a question via the browser
2. Do a forced alignment (https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html)
3. Have the words appear as the words are said

*NEXT STEPS*: Create a schedule for getting the recordings and improving the experience of those playing the game.  Focus on questions that will be most challenging for ASR.  Let people compete against each other while hearing the same question online.

*KEY RESULTS*: See how much fine tuning on the newly collected data helps full pipeline QA.

Identifying Bad Evaluation Questions via Dataset Cartography
------------------------------------------------------------

The goal of [this paper introducing dataset cartography](https://www.aclweb.org/anthology/2020.emnlp-main.746/) is to identify training examples that are *easy-to-learn*, *hard-to-learn*, or *ambiguous*.
This paper identifies them through the training dynamics from epoch to epoch. For example, examples which a model confidently answers correctly from the very first epoch are likely to be ``easy'' while predictions that flip back and forth are likely to be flawed in some way.
While this paper focuses on training data, with adaptation its possible to apply to evaluation data as well.


The big picture of this project would be to combine methods from several papers that identify bad evaluation examples and report on their agreement/disagreement/propose new ideas.
The first step towards this is adapting the method from the Dataset Cartography paper to work for *evaluation* data instead of *training* data.

*FIRST STEP*: Create a fork of https://github.com/allenai/cartography that does the following
1. Their method uses a static development/test set. Change the evaluation so that it uses K-fold validation and that the development set is in one of these folds
2. Rerun their analysis, but this time only show results for the development set.
3. Manually annotate 100 examples to evaluate how well their method identifies problematic examples (e.g., for the least variable and least confident examples, why are they hard to learn)

*NEXT STEPS*: Is performance when an example is in the validation K-fold predictive of its properties when its in the training fold?

*KEY_RESULTS*: Do the results of the paper generalize to the development data? How do the results differ on the development data?