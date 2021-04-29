Subject
==================

You may either enter a shared project on 
[question answering](qa.md), a shared project on [social media](https://github.com/Pinafore/cl1-hw/blob/master/project/sw_project.pdf), or
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

Groups
==================

You must form a group of 4-6 students enrolled in this course.  If you have a project idea and you're not able to convince two other people to work on it, it's probably not that interesting.  You should instead join another group.  

Proposal
==================

The project proposal can be turned in 13. March or 24. March (we're giving two due dates because some people are raring to get started with the project and want to run ideas by us ASAP and other people need more time).  This one page PDF document
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

Your first deliverable is due 16. April.  This is a one page writeup detailing what you've done thus far.  It should prove that your project idea is sound and that you've made progress.  Good indications of this are:
* You have your data
* You've implemented baseline techniques that work
* You've made some progress on the overall goal of your project

Post your first deliverable report publicly on Piazza.

Final Presentation
======================

The final presentation will be an eight minute video.  In the final presentation you will:

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

It is okay to go under time, but
_do not_ go over time.  It will negatively impact your grade. 

We'll treat this as a flipped classroom session (as usual).  You'll post your videos by May 9, we'll watch them before the last class, and then we'll ask questions.

Project Writeup
======================

By 10:00 13. May (the final exam time as set by the registrar), have the person in your group whose last name
is alphabetically last submit their project writeup explaining what
you did and what results you achieved on GradeScope.  This document should
make it clear:

* Why this is a good idea
* What you did
* Who did what
* Whether your technique worked or not

Please do not go over 2500 words unless you have a really good reason.
Images are a much better use of space than words, usually (there's no
limit on including images, but use judgement and be selective).

I'd suggest that you use either the ACL or ICML template:
* https://icml.cc/Conferences/2018/StyleAuthorInstructions
* https://acl2018.org/downloads/acl18-latex.zip


Grade
======================

The grade will be out of 25 points, broken into five areas:

* _Presentation_: For your oral presentation, do you highlight what
  you did and make people care?  Did you use time well during the
  presentation?

* _Writeup_: Does the writeup explain what you did in a way that is
  clear and effective?

* _Technical Soundness_: Did you use the right tools for the job, and
  did you use them correctly?  Were they relevant to this class?

* _Effort_: Did you do what you say you would, and was it the right
  ammount of effort?

* _Results_:  How well did your techniques perform, and how thorough and insightful is your analysis?

Computational Resources
=============================
Check Piazza for Google Cloud Credits you can use for your project.  In addition, the University of Maryland Center for Machine Learning (CML) provides
computational resources for project-based machine learning projects in
graduate courses.  To request access, send an e-mail to
 kxnguyen@umd.edu with the subject line "CMSC 723 Project
Computational Resources" with the contact information for everyone in
your group.  Please also direct any technical issues to the TA, who
can loop in CML staff as needed.
