import FSM
import util

vocabulary = ['panic', 'picnic', 'ace', 'pack', 'pace', 'traffic', 'lilac', 'ice', 'spruce', 'frolic']
suffixes   = ['', '+ed', '+ing', '+s']

def buildSourceModel(vocabulary, suffixes):
    # we want a language model that accepts anything of the form
    # *   w
    # *   w+s
    fsa = FSM.FSM()
    fsa.setInitialState('start')
    fsa.setFinalState('end')
    
    ### TODO: YOUR CODE HERE
    util.raiseNotDefined()

    return fsa

def buildChannelModel():
    # this should have exactly the same rules as englishMorph.py!
    fst = FSM.FSM(isTransducer=True)
    fst.setInitialState('start')
    fst.setFinalState('end')

    # we can always get from start to end by consuming non-+
    # characters... to implement this, we transition to a safe state,
    # then consume a bunch of stuff
    fst.addEdge('start', 'safe', '.', '.')
    fst.addEdge('safe',  'safe', '.', '.')
    fst.addEdge('safe',  'safe2', '+', None)
    fst.addEdge('safe2', 'safe2', '.', '.')
    fst.addEdge('safe',  'end',  None, None)
    fst.addEdge('safe2',  'end',  None, None)
    
    # implementation of rule 1
    fst.addEdge('start' , 'rule1' , None, None)   # epsilon transition
    fst.addEdge('rule1' , 'rule1' , '.',  '.')    # accept any character and copy it
    fst.addEdge('rule1' , 'rule1b', 'e',  None)   # remove the e
    fst.addEdge('rule1b', 'rule1c', '+',  None)   # remove the +
    fst.addEdge('rule1c', 'rule1d', 'e',  'e')    # copy an e ...
    fst.addEdge('rule1c', 'rule1d', 'i',  'i')    #  ... or an i
    fst.addEdge('rule1d', 'rule1d', '.',  '.')    # and then copy the remainder
    fst.addEdge('rule1d', 'end' , None, None)   # we're done

    # implementation of rule 2
    fst.addEdge('start' , 'rule2' , '.', '.')     # we need to actually consume something
    fst.addEdge('rule2' , 'rule2' , '.', '.')     # accept any character and copy it
    fst.addEdge('rule2' , 'rule2b', 'e', 'e')     # keep the e
    fst.addEdge('rule2b', 'rule2c', '+', None)    # remove the +
    for i in range(ord('a'), ord('z')):
        c = chr(i)
        if c == 'e' or c == 'i':
            continue
        fst.addEdge('rule2c', 'rule2d', c, c)     # keep anything except e or i
    fst.addEdge('rule2d', 'rule2d', '.', '.')     # keep the rest
    fst.addEdge('rule2d', 'end' , None, None)   # we're done

    # implementation of rule 3
    ### TODO: YOUR CODE HERE
    util.raiseNotDefined()

    return fst

def simpleTest():
    fsa = buildSourceModel(vocabulary, suffixes)
    fst = buildChannelModel()

    print "==== Trying source model on strings 'ace+ed' ===="
    output = runFST([fsa], ["ace+ed"])
    print "==== Result: ", str(output), " ===="

    print "==== Trying source model on strings 'panic+ing' ===="
    output = runFST([fsa], ["panic+ing"])
    print "==== Result: ", str(output), " ===="
    
    print "==== Generating random paths for 'aced', using only channel model ===="
    output = runFST([fst], ["aced"], maxNumPaths=10, randomPaths=True)
    print "==== Result: ", str(output), " ===="

    print "==== Disambiguating a few phrases: aced, panicked, paniced, sprucing ===="
    output = runFST([fsa,fst], ["aced", "paniced", "panicked", "sprucing"])
    print "==== Result: ", str(output), " ===="

    
