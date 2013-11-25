from collections import defaultdict
from string import lower
import gzip
import itertools
from collections import defaultdict
from math import log

from nltk.probability import FreqDist
from nltk.model.ngram import NgramModel
from nltk.util import ingrams
from nltk.probability import LidstoneProbDist

class StubLanguageModel:
    def logprob(self, context, word):
        return 0.0

class BiCorpus:
    """
    A class to provide an iterator over aligned sentences in two languages.
    """

    def __iter__(self):
        for ii, jj in self.raw_sentences():
            yield ii, jj


class GzipAlignedCorpus(BiCorpus):
    """
    A class to read from gzipped files (e.g. en.corpus.gz as provided by this
    assignment)
    """

    def __init__(self, base, langs, limit):
        self._base = base
        self._langs = langs
        self._limit = limit

    def raw_sentences(self):
        sentence_count = 0
        for langs in zip(gzip.open("%s.%s.gz" % (self._base, self._langs[0])),
                         gzip.open("%s.%s.gz" % (self._base, self._langs[1]))):
            sentence_count += 1
            yield [x.split() for x in langs]

            if limit > 0 and sentence_count >= limit:
                break


class ToyCorpus(BiCorpus):
    """
    A toy corpus you can use for debugging.  It contains lyrics from German
    songs
    """

    def raw_sentences(self):
        data = [("von 99 Luftballons",
                 "about 99 balloons"),
                ("99 dusenjaeger",
                 "99 fighter jets"),
                ("wo auch der Himmel weint",
                 "where the heaven also cries"),
                ("der siebte Himmel ist noch weit",
                 "the seventh heaven is far"),
                ("von hier an blind",
                 "blind about now"),
                ("der Himmel waer dann rosa",
                 "the heaven would be pink"),
                ("mein Irisch Kind wo wilest du",
                 "where are you my Irish child"),
                ("mein Fahrrad ist nicht rosa",
                 "my bike is not pink"),
                ("du bist nicht blind",
                 "you are not blind"),
                ("bleibe ruhig mein Kind",
                 "stay quiet my child")]

        for ii, jj in [map(lower, x) for x in data]:
            yield jj.lower().split(), ii.lower().split()


class Translation:
    """
    A class that provides translation probabilities from English to foreign
    words.
    """

    def __init__(self):
        # Initialize the data structure
        self._counts = defaultdict(FreqDist)

    def score(self, word_e, word_f):
        """
        Returns the MLE probability of an foreign word given English word
        """

        return self._counts[word_e].freq(word_f)

    def get_count(self, word_e, word_f):
        """
        Return the number of times an English word was translated into a foreign
        word
        """

        return self._counts[word_e][word_f]

    def collect_count(self, count, word_e, word_f):
        """
        Increment the count of the number of times a foreign word was translated
        into an English word.
        """

        self._counts[word_e].inc(word_f, count)

    def vocab(self):
        """
        Return all english words
        """

        return self._counts.keys()

    def status(self):
        """
        Return a brief string summarizing the state of the translations
        """

        keys = self._counts.keys()[:5]
        s = ""
        for ee in keys:
            s += "%s:%s\n" % (ee, "\t".join("%s:%f" % \
                                                (ff, self.score(ee, ff)) \
                                                for ff in \
                                                self._counts[ee].keys()[:10]))
        return s


class UniformTranslation:
    """
    Dummy class to provide initial translations
    """

    def score(self, word_e, word_f):
        return 1.0


class ModelOne:

    def __init__(self):
        # The language model is initially undefined
        self._lm = None
        self._lm_order = -1
        self._trans = UniformTranslation()

    def initial_translation(self, corpus):
        """
        Create an initial translation probability; translate all foreign words
        into any English word with uniform probability.
        """
        return UniformTranslation()

    def sentence_counts(self, sentence_e, sentence_f, translation):
        """
        Marginalizing over all alignments, generate an iterator over expected
        counts of translations.

        Every time word e_i is aligned with f_j in alignment a, that should be
        weighted be the probability of p(a | E, F), where E and F are the whole
        sentences given as input.  You should yield

        e_i, f_j, p(a | E, F)

        For all word pairs.  It is efficient to marginalize over alignments
        mathematically than to try to generate them all.
        """

        # Fill this in!  You should have a big for loop that surrounds
        # something that looks like:
        #
        # yield english_word, foreign_word, expected_count

        return []

    def accumulate_counts(self, corpora, translation):
        """
        Given a corpus, add the counts into the new translation model (this is
        the M step of EM).
        """

        new_trans = Translation()
        sentence_count = 0

        for e_sent, f_sent in corpora:
            if sentence_count % 100 == 0:
                print("LM Sentence %i" % sentence_count)

            e_sent = [None] + e_sent
            for e_ii, f_jj, cc in self.sentence_counts(e_sent, f_sent,
                                                       translation):
                new_trans.collect_count(cc, e_ii, f_jj)

            sentence_count += 1

        return new_trans

    def report(self, words, top_words=5):
        """
        Return the most likely translations of English words
        """

        for ii in [lower(x).strip() for x in words]:
            probability = FreqDist()
            for jj in self._trans.vocab():
                probability.inc(jj, self._trans.score(ii, jj))

            for jj in probability.keys()[:top_words]:
                yield ii, jj, probability[jj]

    def best_alignment(self, sentence_e, sentence_f):
        """
        Return a list of index tuples (e, f) that encodes the most likely
        alignment between the English sentence and the foreign sentence.
        """

        # Extra credit!

        return []

    def translate_score(self, sentence_e, sentence_f):
        """
        Return the noisy channel *log* probability of this score using a
        language model.  Computes p(e) using the logprob method of the language
        model.

        Caution: the nltk logprob function returns the negative logprob, so
        you'll need to negate at some point.
        """

        assert self._lm, "Language model must be loaded"
        assert sentence_e[0] != None, \
            "Score function adds None so you don't have to"

        # Fill this in!  Compute the noisy channel score of this translation!

        return 0.0

    def build_lm(self, corpus, order=2):
        """
        Create a reasonable English language model on your training data.
        """

        self._lm_order = order
        if order > 0:
            tokens = []
            sentence_count = 0
            for e_sent, f_sent in corpus:
                if sentence_count % 100 == 0:
                    print("LM Sentence %i" % sentence_count)
                    sentence_count += 1

                # Each sentence starts with an empty string
                tokens += [''] + e_sent

                estimator = lambda fdist, bins: \
                    LidstoneProbDist(fdist, 0.1)
                self._lm = NgramModel(order, tokens, pad_left=False,
                                      pad_right=False,
                                      estimator=estimator)
        else:
            self._lm = StubLanguageModel()

    def em(self, corpus, iters):
        """
        Runs EM on the specified corpus for a set number of iterations
        """

        self._trans = self.initial_translation(corpus)

        for ii in xrange(iters):
            print("Model1 Iteration %i ... " % ii)
            self._trans = self.accumulate_counts(corpus, self._trans)
            print(self._trans.status())
            print("done")


if __name__ == "__main__":
    import sys

    # Print usage information
    if len(sys.argv) < 4:
        print("""USAGE: ibm_trans.py MODEL1_ITERATIONS CORPUS TEST

MODEL1_ITERATIONS -> number of iterations to run Model 1
CORPUS            -> gzipped file or "toy"
TEST              -> corpus of words you want to run on
LIMIT             -> how many sentences to read [default=all]
""")
    else:


        if len(sys.argv) == 5:
            model1_iters, corpus, test, limit = sys.argv[1:]
            limit = int(limit)
        else:
            model1_iters, corpus, test = sys.argv[1:]
            limit = -1
        model1_iters = int(model1_iters)

        # Load the corpus
        tests = []
        if corpus == "toy":
            corpus = ToyCorpus()
            tests.append(("my bike cries 99 fighter jets".split(),
                          "mein fahrrad weint 99 dusenjaeger".split()))
            tests.append(("the heaven is pink about now".split(),
                          "von hier an bleibe der himmel rosa".split()))

        else:
            corpus = GzipAlignedCorpus("corpus", ['en', 'de'], limit)
            tests.append(("i cannot say anything at this stage".split(),
                          "das kann ich so aus dem stand nicht sagen".split()))
            tests.append(("i can say anything at this stage".split(),
                          "das kann ich nicht sagen".split()))

        print("Corpus %s, %s m1 iters" % (corpus, model1_iters))

        # Run inference
        m = ModelOne()
        m.em(corpus, model1_iters)


        # Display word translations
        print("========COMPUTING WORD TRANSLATIONS=======")

        for ee, ff, pp in m.report(open(test)):
            print("%s\t%s\t%f" % (ee, ff, pp))
        print("")

        m.build_lm(corpus)

        # Display sentence translations
        print("========COMPUTING SENTENCE TRANSLATIONS=======")

        for ee, ff in tests:
            print("ENGLISH:\t %s" % " ".join(ee))
            print("FOREIGN:\t %s" % " ".join(ff))
            print("SCORE:\t %f" % m.translate_score(ee, ff))
            print("")
