# Author: YOUR NAME HERE
# Date: DATE SUBMITTED

# Use word_tokenize to split raw text into words
from string import punctuation
import sys

import nltk
from nltk.tokenize import word_tokenize


class LimerickDetector:

    def __init__(self):
        """
        Initializes the object to have a pronunciation dictionary available
        """
        self._pronunciations = nltk.corpus.cmudict.dict()

    def shortest_phones(self, word):
        phones = None
        if word in self._pronunciations:
            phones = self._pronunciations[word]
        else:
            return None

        shortest_num_phone = sys.maxint
        shortest_index = 0
        for index, item in enumerate(phones):
            if len(item) < shortest_num_phone:
                shortest_num_phone = len(item)
                shortest_index = index

        return phones[shortest_index]

    def num_syllables(self, word):
        """
        Returns the number of syllables in a word.  If there's more than one
        pronunciation, take the shorter one.  If there is no entry in the
        dictionary, return 1.
        """
        phones = self.shortest_phones(word)
        if phones is None:
            return 1

        phone_count = 0
        for phone in phones:
            if phone[-1].isdigit():
                phone_count += 1

        return phone_count

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        

        return False

    def is_limerick(self, text):
        """
        Takes text where lines are separated by newline characters.  Returns
        True if the text is a limerick, False otherwise.

        A limerick is defined as a poem with the form AABBA, where the A lines
        rhyme with each other, the B lines rhyme with each other (and not the A
        lines).

        (English professors may disagree with this definition, but that's what
        we're using here.)
        """

        return False

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = raw_input()

    ld = LimerickDetector()
    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))
