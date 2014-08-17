import sys, re
from fst import FST
from fsmutils import composewords

kFRENCH_TRANS = {0: "zero", 1: "un", 2: "deux", 3: "trois", 4:
                 "quatre", 5: "cinq", 6: "six", 7: "sept", 8: "huit",
                 9: "neuf", 10: "dix", 11: "onze", 12: "douze", 13:
                 "treize", 14: "quatorze", 15: "quinze", 16: "seize",
                 20: "vingt", 30: "trente", 40: "quarante", 50:
                 "cinquante", 60: "soixante", 100: "cent"}

kFRENCH_AND = 'et'

def prepare_input(integer):
    assert isinstance(integer, int) and integer < 1000 and integer >= 0, \
      "Integer out of bounds"
    return list("%03i" % integer)

def french_count():
    f = FST('french')

    f.add_state('start')
    f.initial_state = 'start'

    for ii in xrange(10):
        f.add_arc('start', 'start', str(ii), [kFRENCH_TRANS[ii]])

    f.set_final('start')

    return f

if __name__ == '__main__':
    user_input = int(raw_input())
    f = french_count()
    if user_input:
        print user_input, '-->',
        print " ".join(f.transduce(prepare_input(user_input)))
