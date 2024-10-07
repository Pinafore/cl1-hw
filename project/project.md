Subject
==================

You may either enter a shared project on 
[question answering](qa.md) or
choose a project of your own.

If you choose to work on a project of your own choosing, it must:
* Have readily available data
* Be of general interest (there should be published guidelines for what constitutes "standard" performance) and should be a problem worth tackling
* Be of specific interest to computational linguistics: there should be clear and obvious application of the techniques we have used in the course
* Have a baseline that you can implement (or run yourself) easily within a week

If you choose to work on a project of your own choosing, you will be
held to a higher standard.  You must clearly document baselines and
show improvement over those baselines.  You'll also need to convince
us to make sure you clearly convey why it's an interesting problem.

What Makes a Valid Project
===========================

You do not have to make a publishable piece of work.  It's fine to:
* Replicate a technique that does not have published source code
* Run an existing method on new data
* Make a minor modification to existing work
* Do a thorough analysis of existing work
* Create a visualization or interface for existing work
* Do a literature review of an established subarea: but you should
  still be running some of the experiments, analyzing the data

The bar is considerably lower than for a publication!

Groups
==================

You must form a group of 3-6 students enrolled in this course.  If you have a project idea and you're not able to convince two other people to work on it, it's probably not that interesting.  You should instead join another group.  

Proposal
==================

The project proposal can be turned in 14. October.  This one page PDF document
should describe:

* What project you're working on.  If you're not doing the shared competition, you must describe the task, data, and baselines clearly.

* Who is on your team

* What techniques you will explore 

* Your timeline for completing the project (be realistic; you should
  have your first results by the first deliverable deadline)

Have the person in your group whose name is alphabetically first
submit your assignment on Piazza (we're doing this on Piazza so that people can see projects and see if there are potential synergies / best practice that can be shared).  Late days cannot be used on this
assignment.  (We'll create a thread for this you can add to as a followup.)

First Deliverable
======================

Your first deliverable is due 22. November.  This is a one page writeup detailing what you've done thus far.  It should prove that your project idea is sound and that you've made progress.  Good indications of this are:
* You have your data
* You've implemented baseline techniques that work
* You've made some progress on the overall goal of your project

Post your first deliverable report publicly on Piazza, we will create
a thread for you add your deliverable as a followup.

Final Presentation
======================

The final presentation will be a poster session.  Your poster should:

* Explain what you did

* Who did what

* What challenges you had

* Review how well you did 

* Provide an error analysis or otherwise do a qualitative examination of your data.  For example, in a traditional supervised setup, this would contain examples from the
  development set that you get wrong.  You should show those sentences
  and explain why (in terms of features or the model) they have the
  wrong answer.  You should have been doing this all along, but this is your final inspection of
  your errors. The features or model problems you discover should not
  be trivial features you could add easily.  Instead, these should be
  features or models that are difficult to correct.  An error analysis
  is not the same thing as simply presenting the error matrix, as it
  does not inspect any individual examples.

* The linguistic motivation for your features.  This is a
  computational linguistics class, so you should give precedence to
  features / techniques that we use in this class (e.g., syntax,
  morphology, part of speech, word sense, etc.).  Given two features
  that work equally well and one that is linguistically motivated,
  we'll prefer the linguistically motivated one.

* Presumably you did many different things; how did they each
  individually contribute to your final result?


Project Writeup
======================

By 11:59 19. Dec (the final exam time as set by the registrar), have the person in your group whose last name
is alphabetically last submit their project writeup explaining what
you did and what results you achieved on Gradescope.  This document should
make it clear:

* Why this is a good idea
* What you did
* Who did what
* Whether your technique worked or not

Please do not go over 2500 words unless you have a really good reason.
Images are a much better use of space than words, usually (there's no
limit on including images, but use judgement and be selective).

I'd suggest that you use either the ACL template:
* https://github.com/acl-org/acl-style-files


Grade
======================

The grade will be out of 25 points, broken into five areas:

* _Presentation_: For your poster presentation, do you highlight what
  you did and make people care? 

* _Writeup_: Does the writeup explain what you did in a way that is
  clear and effective?

* _Technical Soundness_: Did you use the right tools for the job, and
  did you use them correctly?  Were they relevant to this class?

* _Effort_: Did you do what you say you would, and was it the right
  ammount of effort?

* _Results_:  How well did your techniques perform, and how thorough and insightful is your analysis?

Computational Resources
=============================

Check Piazza for Google Cloud Credits you can use for your project.
In addition, the University of Maryland Nexus Cluster provides
computational resources for project-based machine learning projects in
graduate courses.  To request access, send an e-mail to
aseth125@umd.edu and ssakshi@umd.edu with the subject line "CMSC 723 Project
Computational Resources" (regardless of what course you're registered
in) with the contact information for everyone in
your group.  Please also direct any technical issues to the TA, who
can loop in Nexus staff as needed.
