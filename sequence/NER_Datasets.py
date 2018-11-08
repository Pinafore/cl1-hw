
# Utility functions
def tags(corpus):
    return sorted(list(set(x[1] for x in corpus.tagged_words())))

def normalize(word):
    return word.lower()

def vocabulary(corpus):
    return sorted(list(set(x for x in corpus.words())))

def load_conll_file(path):
    sents = []
    tags = []
    curr_sent = []
    curr_tags = []
    with open(path) as inh:
        for ln in inh:
            ln = ln.strip()
            if ln.startswith('-DOCSTART-'): continue
            if ln == '': 
                if len(curr_sent) == 0: continue
                #end of sentence
                sents.append(curr_sent)
                tags.append(curr_tags)
                curr_sent = []
                curr_tags = []
            else:
                parts = ln.split()
                curr_sent.append(parts[0])
                curr_tags.append(parts[-1])
    return sents, tags


class ToyDataset:
    def __init__(self):
        self._sents = ["College Park is one hour away from Baltimore".split(),
                       "Adobe opens a new office in College Park".split(),
                       "Amazon Baltimore has a hiring event in College Park".split(),
                       "Baltimore Aircoil parteners with Amazon".split(),
                       "A new office of Baltimore Aircoil opens in College Park".split()]

        self._tags = ["B-LOC I-LOC O O O O O B-LOC".split(),
                      "B-ORG O O O O O B-LOC I-LOC".split(),
                      "B-ORG I-ORG O O O O O B-LOC I-LOC".split(),
                      "B-ORG I-ORG O O B-ORG".split(),
                      "O O O O B-ORG I-ORG O O B-LOC I-LOC".split()]

    def tagged_sents(self):
        for sent, tag in zip(self._sents, self._tags):
            yield zip(sent, tag)

    def words(self):
        for ii in self._sents:
            for jj in ii:
                yield jj

    def tagged_words(self):
        for sent, tag in zip(self._sents, self._tags):
            for ii, jj in zip(sent, tag):
                yield ii, jj

class CoNLL2003_Train(ToyDataset):
    def __init__(self):
        self._sents, self._tags = load_conll_file('conll2003/train.txt')

class CoNLL2003_Valid(ToyDataset):
    def __init__(self):
        self._sents, self._tags = load_conll_file('conll2003/valid.txt')

class CoNLL2003_Test(ToyDataset):
    def __init__(self):
        self._sents, self._tags = load_conll_file('conll2003/test.txt')

def dataset_to_sents_and_tags(dataset):
    sents = [(lambda x: [y[0] for y in x])(pair) for
                   pair in dataset.tagged_sents()]
    tags = [(lambda x: [y[1] for y in x])(pair) for
                  pair in dataset.tagged_sents()]
	
    return sents, tags