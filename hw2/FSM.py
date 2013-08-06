import re
import os

carmelPath = '~hal/bin/carmel'

class FSM:        
    def __init__(self, isTransducer=False, isProbabilistic=False):
        self.reset(isTransducer, isProbabilistic)

    def reset(self, isTransducer, isProbabilistic):
        self.nodes = {}
        self.edges = {}
        self.initialState = None
        self.finalState   = None
        self.N = 0
        self.M = 0
        self.isTransducer = isTransducer
        self.isProbabilistic = isProbabilistic

    def readFromFile(self, filename):
        self.reset(True, True)
        def parseChar(s):
            if s == '*e*': return None
            match = re.match('^"(.+)"$', s)
            if match is None: return None
            return match.group(1)
    
        h = open(filename, 'r')
        finalState = h.readline().strip();
        self.setFinalState(finalState)
        isFirstLine = True
        for l in h.readlines():
            match = re.match("^\(([^ ]+) \(([^ ]+) ([^ ]+) ([^ ]+) ([^ ]+)\)\)$", l.strip())
            if match is not None:
                fromState = match.group(1)
                toState   = match.group(2)
                fromChar  = parseChar(match.group(3))
                toChar    = parseChar(match.group(4))
                prob      = float(match.group(5))
                self.addEdge(fromState, toState, fromChar, toChar, prob)
                if isFirstLine:
                    self.setInitialState(fromState)
                    isFirstLine = False
        h.close()

    def addNode(self, u):
        if not self.nodes.has_key(u):
            self.N = self.N + 1
            self.nodes[u] = self.N

    def addEdge(self, u, v, inputChar, outputChar=None, prob=None):
        self.addNode(u)
        self.addNode(v)
        if not self.edges.has_key(u):
            self.edges[u] = {}
        if not self.edges[u].has_key(v):
            self.edges[u][v] = []
        if self.isProbabilistic and prob is None: prob=1
        if inputChar == '.':
            for c in range(ord('a'), ord('z')):
                if outputChar == '.':
                    self.edges[u][v].append( (chr(c), chr(c), prob) )
                else:
                    self.edges[u][v].append( (chr(c), outputChar, prob) )
                self.M = self.M+1
        else:
            self.edges[u][v].append( (inputChar, outputChar, prob) )
            self.M = self.M+1

    def addEdgeSequence(self, u, v, inputCharSeq):
        self.addNode(u)
        self.addNode(v)
        prev = u
        for i in range(len(inputCharSeq)-1):
            this = "*tmp*" + str(self.N+1)
            if self.isTransducer:
                self.addEdge(prev, this, inputCharSeq[i], inputCharSeq[i])
            else:
                self.addEdge(prev, this, inputCharSeq[i])
            prev = this
        if self.isTransducer:
            self.addEdge(prev, v, inputCharSeq[-1], inputCharSeq[-1])
        else:
            self.addEdge(prev, v, inputCharSeq[-1])

    def setInitialState(self, u):
        self.addNode(u)
        self.initialState = u

    def setFinalState(self, u):
        self.addNode(u)
        self.finalState = u

    def escape(self, char):
        if char is None:
            return "*e*"
        return '"' + re.sub('"', '\\"', char) + '"'

    def writeEdges(self, h, u):
        if self.edges.has_key(u):
            i = self.nodes[u]
            for (v, charList) in self.edges[u].iteritems():
                j = self.nodes[v]
                for (inputChar, outputChar, prob) in charList:
                    if self.isTransducer:
                        h.write("(" + str(i) + " (" + str(j) + " " + self.escape(inputChar) + " " + self.escape(outputChar))
                    else:
                        h.write("(" + str(i) + " (" + str(j) + " " + self.escape(inputChar))
                    if self.isProbabilistic:
                        h.write(" " + str(prob))
                    h.write("))\n")

    def writeToFile(self, filename):
        if self.N == 0 or self.M == 0:
            raise Exception("FST.writeToFile error: empty FST")
        if self.initialState is None:
            raise Exception("FST.writeToFile error: no initial state")
        if self.finalState is None:
            raise Exception("FST.writeToFile error: no final state")

        h = open(filename, 'w')
        h.write( str(self.nodes[self.finalState]) + "\n")
        self.writeEdges(h, self.initialState)
        for u in self.nodes.iterkeys():
            if u != self.initialState:
                self.writeEdges(h, u)
        h.close()


    def trainFST(self, inputs, outputs):
        interleaved = []
        for i in range(len(inputs)):
            interleaved.append(outputs[i])
            interleaved.append(inputs[i])

        self.writeToFile(".tmp.fst")
        writeStringFile('.tmp.fst.strings', interleaved)

        cmd = carmelPath + ' -rtDHJ .tmp.fst .tmp.fst.strings > .tmp.output.fst'
        print "executing: ", cmd
        p = os.system(cmd)

        if p != 0:
            raise Exception("execution of carmel failed!  return value=" + str(p))

        self.readFromFile('.tmp.output.fst')



def writeStringFile(filename, strings):
    h = open(filename, 'w')
    for string in strings:
        first = True
        for c in string:
            if not first: h.write(' ')
            h.write('"' + c + '"')
            first = False
        h.write('\n')
    h.close()

def runFST(fstList, strings, maxNumPaths=1, randomPaths=False, quiet=False):
    cmd = carmelPath + ' -rIQE'
    if randomPaths:
        if len(strings) > 1:
            raise Exception("cannot generate random paths for >1 input string")
        cmd = cmd + 'i -G ' + str(maxNumPaths)
    else:
        cmd = cmd + 'b -k ' + str(maxNumPaths)
        
    for i in range(len(fstList)):
        fstList[i].writeToFile(".tmp.fst." + str(i))
        cmd = cmd + ' .tmp.fst.' + str(i)
    writeStringFile('.tmp.fst.strings', strings)
    cmd = cmd + ' .tmp.fst.strings > .tmp.output'
    print "executing: ", cmd
    if quiet: cmd = "( " + cmd + " ) > /dev/null 2>&1"
    p = os.system(cmd)
    if p != 0:
        raise Exception("execution of carmel failed!  return value=" + str(p))

    outputs = []
    h = open('.tmp.output', 'r')
    for i in range(len(strings)):
        this = []
        for j in range(maxNumPaths):
            l = h.readline()
            if l == '':
                raise Exception("error: not enough lines in carmel output")
            chars = l.split()
            if len(chars) > 1:   # this properly analyzed
                this.append(''.join(chars[0:-1]))
        outputs.append(this)
    h.close()
    return outputs

