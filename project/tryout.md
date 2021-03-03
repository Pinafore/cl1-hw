QA for Speech Input
-----------------------------------

This paper makes an argument for improving the underlying algorithms for ASR for QA (https://arxiv.org/abs/2102.08345).  To do this realistically, we need a more comprehensive dataset.

The big picture is that we need to create an incentive system to get people to read questions out loud, so we need an overall system to read questions out loud for fun so people can answer them.  The cost of admission is that to hear questions (and have fun) they need to record a few questions of their own.

*FIRST STEP*: Update a server (https://github.com/ihsgnef/qb_interface) to do the following:
1. Upload a sound file for a question via the browser
2. Do a forced alignment (https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html)
3. Have the words appear as the words are said

*NEXT STEPS*: Create a schedule for getting the recordings and improving the experience of those playing the game.  Focus on questions that will be most challenging for ASR.

*KEY RESULTS*: See how much fine tuning on the newly collected data helps full pipeline QA.

