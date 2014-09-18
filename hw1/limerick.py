# Author: Sanghee Kim
# Date: September 14, 2014

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
            if len(item) <= shortest_num_phone:
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

    def guess_syllables(self, word):
        vowels = 'aeiouy'
        count = 0
        if word[0] in vowels:
            count += 1
        for index in range(1,len(word)):
            if (word[index] in vowels) and (word[index-1] not in vowels):
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count

    def rhyme(self, word):
        if word in self._pronunciations:
            phones_list = self._pronunciations[word]
        else:
            return None

        rhyme_list = []
        for phones in phones_list:
            result = ""
            for index, phone in enumerate(phones):
                if phone[-1].isdigit():
                    result = "".join(phones[index:])
                    break
            rhyme_list.append(''.join([i for i in result if not i.isdigit()]))

        return rhyme_list

    def compare_rhymes(self, rhyme_a, rhyme_b):
        if len(rhyme_a) == len(rhyme_b):
            if rhyme_a == rhyme_b:
                return True
        elif len(rhyme_a) < len(rhyme_b):
            if rhyme_a == rhyme_b[-len(rhyme_a):]:
                return True
        else:
            if rhyme_b == rhyme_a[-len(rhyme_b):]:
                return True

        return False

    def rhymes(self, a, b):
        """
        Returns True if two words (represented as lower-case strings) rhyme,
        False otherwise.
        """
        rhyme_a = self.rhyme(a)
        rhyme_b = self.rhyme(b)
        for r_a in rhyme_a:
            for r_b in rhyme_b:
                if self.compare_rhymes(r_a, r_b):
                    return True
        return False

    def apostrophe_tokenize(self, text):
        exclude = set(punctuation)
        exclude.remove(r"'")
        text = ''.join(ch for ch in text if ch not in exclude)
        tokenizer = nltk.RegexpTokenizer('\s+', gaps=True)
        return tokenizer.tokenize(text)

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
        text = text.strip()
        lines = text.split('\n')
        last_words = []
        for line in lines:
            last_words.append(self.apostrophe_tokenize(line)[-1])

        if len(last_words) == 5:
            if self.rhymes(last_words[0], last_words[1]) and \
               self.rhymes(last_words[1], last_words[4]):
                   if self.rhymes(last_words[2], last_words[3]) and \
                      not self.rhymes(last_words[0], last_words[2]):
                          return True
        return False

if __name__ == "__main__":
    buffer = ""
    inline = " "
    while inline != "":
        buffer += "%s\n" % inline
        inline = raw_input()

    ld = LimerickDetector()
    print("%s\n-----------\n%s" % (buffer.strip(), ld.is_limerick(buffer)))
