import re

def iterateRules():
    # the first rule looks for STUFF followed by "e+e" or "e+i"
    # followed by STUFF and removes the initial e.
    yield ("(.*)e\+([ei].*)", "\\1\\2")

    # the next rule to remove a + if it's between an e and something
    # that neither an e nor an i
    yield ("(.+)e\+([^ei].*)", "\\1e\\2")

    # your task is to write the next rule, which should map "_c+ed"
    # and "_c+ing" to a version with "ck" instead, so long as "_" is a
    # vowel.
    #
    # if you need help with python regular expressions, see:
    #   http://docs.python.org/library/re.html


def generate(analysis):
    word = analysis
    # apply all rules in sequence
    for (ruleLHS, ruleRHS) in iterateRules():
        word = re.sub("^" + ruleLHS + "$", ruleRHS, word)

    # remove any remaining boundaires.  you may wish to comment this
    # out for debugging purposes
    word = re.sub("\+","", word)
    return word

if __name__ == '__main__':
    user_input = raw_input()
    if user_input:
        print user_input, '-->',
        print generate(user_input)
