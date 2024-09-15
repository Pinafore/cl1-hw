# There once was a Python warmup from ...

First, check out the Github repository with the course homework templates:

[https://github.com/Pinafore/cl1-hw](https://github.com/Pinafore/cl1-hw)

The goal of this assignment is to create a piece of code that will
determine whether a poem is a
[limerick](http://en.wikipedia.org/wiki/Limerick\_(poetry)) or
not.  To do this, we will be using the
[CMU pronunciation dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) (which is covered in the [second chapter of the NLTK book](http://www.nltk.org/book/ch02.html)).

A limerick is defined as a poem with the form AABBA, where the A lines
rhyme with each other, the B lines rhyme with each other (and not the
A lines).  (English professors may disagree with this definition, but
that's what we're using here to keep it simple.  There are also
constraints on how many syllables can be in a line.)

## Programming

Look at the file limerick.py in the limerick folder.  Your job is to fill
in the missing functions in that file so that it does what it’s
supposed to do.
* `num_syllables`: look at how many syllables a word has.  Implement this first, as it is easy and will help you understand what the CMU dictionary data looks like.  You don't *need* to use it in implementing subsequent functions, but you can.  (Historically, this homework also had you check whether the meter/syllable constraints of limericks held and it was essential for that, but we removed that to make the homework easier.  This is necessary for the extra credit, though.  We left it in because it's a good introduction to syllables are represented.)
* `after_stressed`: Get all the sounds after the stressed vowel.
* `rhymes`: detect whether two words rhyme or not
* `last_words`: Get the last words in a line.
* `is_limerick`: given a candidate limerick, return whether it meets the constraint or not.
More requirements / information appear in the source files.

**What does it mean for two words to rhyme?** They should share the
  same sounds in their pronunciation starting with the last stressed
  vowel.  (This is a very strict definition of rhyming.  This makes
  the assignment easier.)  If one word is longer than the other, then
  the sounds of shorter word (except for anything before the stressed
  vowel) should be a suffix of the sounds of the longer.  To further
  clarify, when we say "one word is longer than the other", we are
  using number of phonemes as the metric, not number of syllables or
  number of characters.

**How do I know if my code is working?**  Run the provided unit tests (python tests.py) in the homework directory.  Initially, many of them will fail.  If your program is working correctly, all of them will pass.  However, the converse is not true: passing all the tests is not sufficient to demonstrate that your code is working.  *This is strongly encouraged, as it will be similar to how your code is graded.*

**How do I separate words from a string of text?**  Use the `word_tokenize` function.

**What if a word isn’t in the pronouncing dictionary?** Assume it doesn’t rhyme with anything and only has one syllable.

**How ``hardened'' should the code be?** It should handle ASCII text with punctuation and whitespace in upper or lower case.

**What if a word has multiple pronunciations?**  If a word like “fire” has multiple pronunciations, then you should say
that it rhymes with another word if any of the pronunciations rhymes.

**What if a word starts with a vowel?**  If its first stressed vowel is the first sound then the entire word should be a
suffix of the other word.

**What about end rhymes?**  End rhymes are indeed a rhyme, but they make for less interesting
limericks.  So "conspire" and "fire" do rhyme.

**What about stress?**  The stresses of rhymes should be consistent.

**How does the dictionary represent stress?** Either look at words you know and see if you can figure it out or read the documentation.

**What if a word has no vowels?** Then it doesn't rhyme with anything.

**What does this error mean?** If you see the error
```
  Resource cmudict not found.
  Please use the NLTK Downloader to obtain the resource:

  >>> import nltk
  >>> nltk.download('cmudict')
```
Then you need to open up python and then run the download command above.  You can also download the data manually from NLTK, but that's more work.

## Extra Credit

Extra Credit (create new functions for these features; don’t put them
in the required functions that will be run by the autograder):
* Create a new function called
 `apostrophe_tokenize` that handles apostrophes in words correctly so
  that "can’t" would rhyme with "pant" if the `is_limerick` function used `apostrophe\_tokenize` instead of `word_tokenize`.
* Make reasonable guesses about the number of syllables in unknown words in a function called `guess_syllables`.
* Make a new function called `syllable_limerick` that also checks if the stress pattern of limerics matches: three feet of three syllables each; and the shorter third and fourth lines also rhyming with each other, but having only two feet of three syllables.
* Compose a funny original limerick about
  computational linguistics, natural language processing, or machine
  learning (add it to your submission as `limerick.txt`).
Add extra credit code as functions to the `LimerickDetector`
class, but don't interfere with required functionality.
