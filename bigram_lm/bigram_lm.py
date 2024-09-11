from math import log, exp
from collections import defaultdict
import argparse
import typing

from numpy import mean

import nltk
from nltk import FreqDist
from nltk.util import bigrams

kLM_ORDER = 2
kUNK_CUTOFF = 3
kNEG_INF = -1e6

kSTART = "<S>"
kEND = "</S>"

token = int | str

def lower(str):
    return str.lower()
    
def lg(x):
    return log(x) / log(2.0)

def dict_sample(d: typing.Dict, cutoff: int =-1):
    """
    Sample a key from a dictionary using the values as probabilities
    (unnormalized)

    cutoff -- Give up after encountering this much probability mass; set to
    something like 0.9 if sampling is taking too long.
    """

    #TODO(only for extra credit): Implement this function
    
    from random import random



class BigramLanguageModel:

    def __init__(self, unk_cutoff: int, jm_lambda: float=0.6, dirichlet_alpha: float=0.1,
                 kn_discount: float=0.1, kn_concentration: float=1.0,
                 standardize_function: typing.Callable=lambda x: x.lower()):
        """Initialize our bigram language model.

        jm_lambda -- Parameter that controls interpolation between unigram and
        bigram: this is the bigram weight.

        dirichlet_alpha -- The pseudocount added to every observation (even
        zero count ones) when estimating the probability (thus in total, we
        add dirichlet_alpha times the vocabulary size)

        kn_concentration -- How much we weight the backoff distribution (same
        for all contexts) for Knesser-Ney, theta in the equation
        
        kn_discount -- How much we subtract from every count and then add to
        the backoff distribution (same for all contexts) for Knesser-Ney,
        delta in the equation.

        standardize_function -- Function to turn words into a simplified base
        form.

        """
        self._unk_cutoff = unk_cutoff
        self._jm_lambda = jm_lambda
        self._dirichlet_alpha = dirichlet_alpha
        self._kn_discount = kn_discount
        self._kn_concentration = kn_concentration
        self._vocab_final = False
        self._vocab = set()

        self._standardizer = standardize_function
        
        self._training_counts = FreqDist()

        # Add your code here!        
        # Bigram counts        

        # Unigram counts

        # Prefix counts

    def train_seen(self, word: str, count: int=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.

        word -- The word we saw
        count -- How many times we saw the word
        """
        
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        self._training_counts[word] += count

            
    def sample(self, sample_size: int) -> typing.Iterator[str]:
        """
        Extra Credit: Generate from the language model
        
        sample_size -- How many tokens to generate from the model
        """
        return ""                                                      
            
    def vocab_lookup(self, word: str) -> token:
        """
        Given a word, provides a vocabulary representation.  Words under the
        cutoff threshold shold have the same value.  All words with counts
        greater than or equal to the cutoff should be unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        if word in self._vocab:
            return word
        else:
            return None

    def finalize(self):
        """
        Fixes the vocabulary as static, prevents keeping additional vocab from
        being added
        """
        
        self._vocab_final = True
        self._vocab = set(x for x in self._training_counts if \
                          self._training_counts[x] >= self._unk_cutoff)
        self._vocab.add(kSTART)
        self._vocab.add(kEND)
        
        # -------------------------------------------------------------------
        # You may want to add code here because it's at this point you know
        # the vocab size
        # -------------------------------------------------------------------

        assert self.vocab_lookup(kSTART) is not None, "Missing start"
        assert self.vocab_lookup(kEND) is not None, "Missing end"
        
    def censor(self, sentence: typing.Iterable[str]) -> typing.Iterator[str] | typing.Iterator[int]:
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, replace words not in the vocabulary with
        <UNK>, and end the sentence with </s>.

        sentence -- The original sentence
        """
        yield self.vocab_lookup(kSTART)
        for word in sentence:
            yield self.vocab_lookup(self.standardize(word))
        yield self.vocab_lookup(kEND)

    def standardize(self, word: str) -> int | str:
        """
        Standardize a word: if it's not in the vocabulary, replace it with the unknown token

        word -- The original word
        """
        if word is None:
            return None
        else:
            return self._standardizer(word)

    def mle(self, context: token, word: token) -> float:
        """
        Return the log MLE estimate of a word given a context.  If the MLE would
        be negative infinity, use kNEG_INF

        word -- The word we compute the probability for
        context -- The context that we condition on
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.        
        return 0.0

    def laplace(self, context: token, word: token):
        """
        Return the log MLE estimate of a word given a context.

        word -- The word we compute the probability for
        context -- The context that we condition on
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.                    
        return 0.0

    def jelinek_mercer(self, context, word):
        """
        Return the Jelinek-Mercer log probability estimate of a word
        given a context; interpolates context probability with the
        overall corpus probability.

        word -- The word we compute the probability for
        context -- The context that we condition on
        """

        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.                
        return 0.0

        
          
    def kneser_ney(self, context: token, word: token) -> float:
        """
        Return the log probability of a word given a context given Kneser
        Ney backoff, as approximated by the minimal seating assumption of the
        Chinese restaurant process.

        word -- The word we compute the probability for
        context -- The context that we condition on
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    
    def dirichlet(self, context: token, word: token) -> float:
        """
        Additive smoothing, assuming independent Dirichlets with fixed
        hyperparameter.

        word -- The word we compute the probability for
        context -- The context that we condition on
        """
        # This initially return 0.0, ignoring the word and context.
        # Modify this code to return the correct value.
        return 0.0

    def vocab_size(self) -> int:
        """
        Return the size of the vocabulary.
        """
        return len(self._vocab)
    
    def add_train(self, sentence: typing.Iterable[str]):
        """
        Add the counts associated with a sentence.

        sentence -- The sentence we extract the counts from
        """

        # You'll need to complete this function, but here's a line of code that
        # will hopefully get you started.
        for context, word in bigrams(self.censor(sentence)):
            None
            # ---------------------------------------

    def perplexity(self, sentence: str, method: typing.Callable) -> float:
        """
        Compute the perplexity of a sentence given an estimation technique

        sentence -- The sentence we compute the perplexity for
        method -- The estimator function
        """

        # You don't have to modify this function
        return 2.0 ** (-1.0 * mean([method(context, word) for context, word in \
                                    bigrams(self.censor(sentence))]))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--jm_lambda", help="Parameter that controls " + \
                           "interpolation between unigram and bigram: " + \
                           "this is the bigram weight.",
                           type=float, default=0.6, required=False)
    argparser.add_argument("--dir_alpha", help="The pseudocount added to " + \
                           "every observation (even zero count ones) when" + \
                           "estimating the probability (thus in total, we" + \
                           "add dirichlet_alpha times the vocabulary size)",
                           type=float, default=0.1, required=False)
    argparser.add_argument("--unk_cutoff", help="How many times must a word " + \
                           "be seen before it enters the vocabulary",
                           type=int, default=2, required=False)    
    argparser.add_argument("--lm_type", help="Which smoothing technique to use",
                           type=str, default='mle', required=False)
    argparser.add_argument("--train_limit", help="How many sentences to add " + \
                           "from the training corpus.",
                           type=int, default=-1, required=False)
    argparser.add_argument("--test_limit", help="How many test sentences from " + \
                           "the test corpus (Gutenberg)", type=int, default=1, required=False)
    argparser.add_argument("--kn_concentration", help="How much we weight the " + \
                           "backoff distribution (same for all contexts) for " + \
                           "Knesser-Ney, theta in the equation.",
                           type=float, default=1.0, required=False)
    argparser.add_argument("--kn_discount", help="How much we subtract from " + \
                           "every count and then add to the backoff distribution " + \
                           "(same for all contexts) for Knesser-Ney, delta in the " + \
                           "equation.",
                           type=float, default=0.1, required=False)
    
    args = argparser.parse_args()    
    lm = BigramLanguageModel(kUNK_CUTOFF, jm_lambda=args.jm_lambda,
                             dirichlet_alpha=args.dir_alpha,
                             kn_discount=args.kn_discount)

    for sent in nltk.corpus.brown.sents():
        for word in sent:
            lm.train_seen(lm.standardize(word))

    print("Done looking at all the words, finalizing vocabulary")
    lm.finalize()

    sentence_count = 0
    for sent in nltk.corpus.brown.sents():
        sentence_count += 1
        lm.add_train(sent)

        if args.train_limit > 0 and sentence_count >= args.train_limit:
            break

    print("Trained language model with %i sentences from Brown corpus." % sentence_count)

    # Build the test corpus
    num_sentences = len(nltk.corpus.gutenberg.sents())

    if args.test_limit > 0:
        from random import sample
        sentence_indices = sample(range(num_sentences), args.test_limit)
    else:
        from random import shuffle
        sentence_indices = list(range(num_sentences))
        shuffle(sentence_indices)        
    
    for method_name in ['kneser_ney', 'mle', 'dirichlet', 'jelinek_mercer', 'laplace']:
        print("======================")
        print("      %s" % method_name)
        print("======================")        
        sentence_count = 0
        method = getattr(lm, method_name)

        from random import shuffle
        sentences = nltk.corpus.gutenberg.sents()

        for sent_index in sentence_indices:
            sent = sentences[sent_index]
            original = list(sent)
            censored = list(lm.censor(original))

            for ii, jj, kk in zip([""] + original, censored, [0.0] + [method(censored[ii-1], censored[ii]) for ii in range(1, len(censored))]):
                print("%10s\t%10s\t%03.4f" % (ii, jj, kk))
            print("Perplexity: %0.4f" % lm.perplexity(censored, method))
            print("----------------")
