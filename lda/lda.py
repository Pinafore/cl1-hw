from collections import defaultdict
from random import random, randint
from glob import glob
from math import log
import argparse

from nltk.corpus import stopwords
from nltk.probability import FreqDist

from nltk.tokenize import TreebankWordTokenizer
kTOKENIZER = TreebankWordTokenizer()
kDOC_NORMALIZER = True

import time

def dict_sample(d, cutoff=-1):
    """
    Sample a key from a dictionary using the values as probabilities (unnormalized)
    """
    if cutoff==-1:
        cutoff = random()
    normalizer = float(sum(d.values()))

    current = 0
    for i in d:
        assert(d[i] > 0)
        current += float(d[i]) / normalizer
        if current >= cutoff:
            return i
    print("Didn't choose anything: %f %f" % (cutoff, current))


def lgammln(xx):
    """
    Returns the gamma function of xx.
    Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
    Usage: lgammln(xx)
    Copied from stats.py by strang@nmr.mgh.harvard.edu
    """

    assert xx > 0, "Arg to gamma function must be > 0; got %f" % xx
    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516,
             0.120858003e-2, -0.536382e-5]
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
        x = x + 1
        ser = ser + coeff[j] / x
    return -tmp + log(2.50662827465 * ser)


class RandomWrapper:
    """
    Class to wrap a random number generator to facilitate deterministic testing.
    """

    def __init__(self, buff):
        self._buffer = buff
        self._buffer.reverse()

    def __call__(self):
        val = self._buffer.pop()
        print("Using random value %0.2f" % val)
        return val

class VocabBuilder:
    """
    Creates a vocabulary after scanning a corpus.
    """

    def __init__(self, lang="english", min_length=3, cut_first=100):
        """
        Set the minimum length of words and which stopword list (by language) to
        use.
        """
        self._counts = FreqDist()
        self._stop = set(stopwords.words(lang))
        self._min_length = min_length
        self._cut_first = cut_first

        print("Using stopwords: %s ... " % " ".join(list(self._stop)[:10]))

    def scan(self, words):
        """
        Add a list of words as observed.
        """

        for ii in [x.lower() for x in words if x.lower() not in self._stop \
                       and len(x) >= self._min_length]:
            self._counts[ii] += 1

    def vocab(self, size=5000):
        """
        Return a list of the top words sorted by frequency.
        """
        keys = list(self._counts.keys())
        if len(self._counts) > self._cut_first + size:
            return keys[self._cut_first:(size + self._cut_first)]
        else:
            return keys[:size]

class LdaTopicCounts:
    """
    This class works for normal LDA.  There is no correlation between words,
    although words can have an aysymmetric prior.
    """

    def __init__(self, beta=0.01):
        """
        Create a topic count with the provided Dirichlet parameter
        """
        self._beta = {}
        self._beta_sum = 0.0

        # Maintain a count for each word
        self._normalizer = FreqDist()
        self._topic_term = defaultdict(FreqDist)
        self._default_beta = beta

        self._finalized = False

    def set_vocabulary(self, words):
        """
        Sets the vocabulary for the topic model.  Only these words will be
        recognized.
        """
        for ii in range(len(words)):
            self._beta_sum += self._default_beta

    def change_prior(self, word, beta):
        """
        Change the prior for a single word.
        """
        
        assert not self._finalized, "Priors are fixed once sampling starts."

        self._beta[word] = beta
        self._beta_sum += (beta - self._default_beta)

    def initialize(self, word, topic):
        """
        During initialization, say that a word token with id ww was given topic
        """
        self._topic_term[topic][word] += 1
        self._normalizer[topic] += 1

    def change_count(self, topic, word, delta):
        """
        Change the topic count associated with a word in the topic
        """
        
        self._finalized = True

        self._topic_term[topic][word] += delta
        self._normalizer[topic] += delta

    def get_normalizer(self, topic):
        """
        Return the normalizer of this topic
        """
        return self._beta_sum + self._normalizer[topic]

    def get_prior(self, word):
        """
        Return the prior probability of a word.  For tree-structured priors,
        return the probability marginalized over all paths.
        """
        return self._beta.get(word, self._default_beta)
    
    def get_observations(self, topic, word):
        """
        Return the number of occurences of a combination of topic, word, and
        path.
        """
        return self._topic_term[topic][word]

    def word_in_topic(self, topic, word):
        """
        Return the probability of a word type in a topic
        """
        val = self.get_observations(topic, word) + self.get_prior(word)
        val /= self.get_normalizer(topic)
        return val
    
    def report(self, vocab, handle, limit=25):
        """
        Create a human readable report of topic probabilities to a file.
        """
        for kk in self._normalizer:
            normalizer = self.get_normalizer(kk)
            handle.write("------------\nTopic %i (%i tokens)\n------------\n" % \
                      (kk, self._normalizer[kk]))

            word = 0
            for ww in self._topic_term[kk]:
                handle.write("%0.5f\t%0.5f\t%0.5f\t%s\n" % \
                             (self.word_in_topic(kk, ww),
                              self.get_observations(kk, ww),
                              self.get_prior(ww),
                              vocab[ww]))
                      
                word += 1
                if word > limit:
                    break


class Sampler:
    def __init__(self, num_topics, vocab, alpha=0.1, beta=0.01, rand_stub=None):
        """
        Create a new LDA sampler with the provided characteristics
        """
        self._num_topics = num_topics
        self._doc_counts = defaultdict(FreqDist)
        self._doc_tokens = defaultdict(list)
        self._doc_assign = defaultdict(list)
        self._alpha = [alpha for x in range(num_topics)]
        self._sample_stats = defaultdict(int)
        self._vocab = vocab
        self._topics = LdaTopicCounts(beta)
        self._topics.set_vocabulary(vocab)
        self._lhood = []
        self._time = []
        self._rand_stub = rand_stub

    def change_alpha(self, idx, val):
        """
        Update the alpha value; note that this invalidates precomputed values.
        """
        self._alpha[idx] = val

    def get_doc(self, doc_id):
        """
        Get the data associated with an individual document
        """
        return self._doc_tokens[doc_id], self._doc_assign[doc_id], \
            self._doc_counts[doc_id]

    def add_doc(self, doc, vocab, doc_id = None, 
                token_limit=-1):
        """
        Add a document to the corpus.  If a doc_id is not supplied, a new one
        will be provided.
        """
        temp_doc = [vocab.index(x) for x in doc if x in vocab]

        if not doc_id:
            doc_id = len(self._doc_tokens)
        assert not doc_id in self._doc_tokens, "Doc " + str(doc_id) + \
            " already added"

        if len(temp_doc) == 0:
            print("WARNING: empty document (perhaps the vocab doesn't make sense?)")
        else:
            self._doc_tokens[doc_id] = temp_doc

        token_count = 0
        for ww in temp_doc:
            assignment = randint(0, self._num_topics - 1)
            self._doc_assign[doc_id].append(assignment)
            self._doc_counts[doc_id][assignment] += 1
            self._topics.initialize(ww, assignment)

            token_count += 1
            if token_limit > 0 and token_count > token_limit:
                break

        assert len(self._doc_assign[doc_id]) == len(temp_doc), \
               "%s != %s" % (str(self._doc_assign[doc_id]), str(temp_doc))
                                                                     
        return doc_id

    def change_topic(self, doc, index, new_topic):
        """
        Change the topic of a token in a document.  Update the counts
        appropriately.  -1 is used to denote "unassigning" the word from a topic.
        """
        assert doc in self._doc_tokens, "Could not find document %i" % doc
        assert index < len(self._doc_tokens[doc]), \
            "Index %i out of range for doc %i (max: %i)" % \
            (index, doc, len(self._doc_tokens[doc]))
        term = self._doc_tokens[doc][index]
        alpha = self._alpha
        assert index < len(self._doc_assign[doc]), \
               "Bad index %i for document %i, term %i %s" % \
               (index, doc, term, str(self._doc_assign[doc]))
        old_topic = self._doc_assign[doc][index]

        if old_topic != -1:
            assert new_topic == -1
            
            # TODO: Add code here to keep track of the counts and
            # assignments

        if new_topic != -1:
            assert old_topic == -1

            # TODO: Add code here to keep track of the counts and
            # assignments

    def run_sampler(self, iterations = 100):
        """
        Sample the topic assignments of all tokens in all documents for the
        specified number of iterations.
        """
        for ii in range(iterations):
            start = time.time()
            for jj in self._doc_assign:
                self.sample_doc(jj)

            total = time.time() - start
            lhood = self.lhood()
            print("Iteration %i, likelihood %f, %0.5f seconds" % (ii, lhood, total))
            self._lhood.append(lhood)
            self._time.append(total)

    def report_topics(self, vocab, outputfilename, limit=10):
        """
        Produce a report to a file of the most probable words in a topic, a
        history of the sampler, and the state of the Markov chain.
        """
        topicsfile = open(outputfilename + ".topics", 'w')
        self._topics.report(vocab, topicsfile, limit)

        statsfile = open(outputfilename + ".stats", 'w')
        tmp = "iter\tlikelihood\ttime(s)\n"
        statsfile.write(tmp)
        for it in range(0, len(self._lhood)):
            tmp = str(it) + "\t" + str(self._lhood[it]) + "\t" + str(self._time[it]) + "\n"
            statsfile.write(tmp)
        statsfile.close()

        topicassignfile = open(outputfilename + ".topic_assign", 'w')
        for doc_id in self._doc_assign.keys():
            tmp = " ".join([str(x) for x in self._doc_assign[doc_id]]) + "\n"
            topicassignfile.write(tmp)
        topicassignfile.close()

        doctopicsfile = open(outputfilename + ".doc_topics", 'w')
        for doc_id in self._doc_counts.keys():
            tmp = ""
            for tt in range(0, self._num_topics):
                tmp += str(self._doc_counts[doc_id][tt]) + " "
            tmp = tmp.strip()
            tmp += "\n"
            doctopicsfile.write(tmp)
        doctopicsfile.close()

    def sample_probs(self, doc_id, index):
        """
        Create a dictionary storing the conditional probability of this token being assigned to each topic.
        """
        assert self._doc_assign[doc_id][index] == -1, \
          "Sampling doesn't make sense if this hasn't been unassigned."
        sample_probs = {}
        term = self._doc_tokens[doc_id][index]
        for kk in range(self._num_topics):

            # TODO: Compute the conditional probability of
            # sampling a topic; at the moment it's just the
            # uniform probability.
            sample_probs[kk] = 1.0 / float(self._num_topics)

        return sample_probs
        

    def sample_doc(self, doc_id, debug=False):
        """
        For a single document, compute the conditional probabilities and
        resample topic assignments.
        """

        one_doc_topics = self._doc_assign[doc_id]
        
        topics = self._topics

        for index in range(len(one_doc_topics)):
            self.change_topic(doc_id, index, -1)
            sample_probs = self.sample_probs(doc_id, index)

            if self._rand_stub:
                cutoff = self._rand_stub()
            else:
                cutoff = random()
            new_topic = dict_sample(sample_probs, cutoff)

            self.change_topic(doc_id, index, new_topic)

        return self._doc_assign[doc_id]


    def lhood(self):
        val = self.doc_lhood() + self.topic_lhood()
        return val

    def doc_lhood(self):
        doc_num = len(self._doc_counts)
        alpha_sum = sum(self._alpha)

        val = 0.0
        val += lgammln(alpha_sum) * doc_num
        tmp = 0.0
        for tt in range(0, self._num_topics):
            tmp += lgammln(self._alpha[tt])
        val -= tmp * doc_num
        for doc_id in self._doc_counts:
            for tt in range(0, self._num_topics):
                val += lgammln(self._alpha[tt] + self._doc_counts[doc_id][tt])
            val -= lgammln(alpha_sum + len(self._doc_assign[doc_id]))

        return val

    def topic_lhood(self):
        val = 0.0
        vocab_size = len(self._vocab)

        val += lgammln(self._topics._beta_sum) * self._num_topics
        val -= lgammln(self._topics._default_beta) * vocab_size * self._num_topics
        for tt in range(0, self._num_topics):
            for ww in self._vocab:
                val += lgammln(self._topics._default_beta + self._topics._topic_term[tt][ww])
            val -= lgammln(self._topics.get_normalizer(tt))
        return val
    


def tokenize_file(filename):
    contents = open(filename).read()
    for ii in kTOKENIZER.tokenize(contents):
        yield ii
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--doc_dir", help="Where we read the source documents",
                           type=str, default=".", required=False)
    argparser.add_argument("--language", help="The language we use",
                           type=str, default="english", required=False)
    argparser.add_argument("--output", help="Where we write results",
                           type=str, default="result", required=False)    
    argparser.add_argument("--vocab_size", help="Size of vocabulary",
                           type=int, default=1000, required=False)
    argparser.add_argument("--num_topics", help="Number of topics",
                           type=int, default=5, required=False)
    argparser.add_argument("--num_iterations", help="Number of iterations",
                           type=int, default=100, required=False)    
    args = argparser.parse_args()

    vocab_scanner = VocabBuilder(args.language)

    # Create a list of the files
    search_path = u"%s/*.txt" % args.doc_dir
    files = glob(search_path)
    assert len(files) > 0, "Did not find any input files in %s" % search_path
    
    # Create the vocabulary
    for ii in files:
        vocab_scanner.scan(tokenize_file(ii))

    # Initialize the documents
    vocab = vocab_scanner.vocab(args.vocab_size)
    print(len(vocab), vocab[:10]) 
    lda = Sampler(args.num_topics, vocab)
    for ii in files:
        lda.add_doc(tokenize_file(ii), vocab)

    lda.run_sampler(args.num_iterations)
    lda.report_topics(vocab, args.output)

