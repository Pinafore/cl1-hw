import nltk

kLM_ORDER = 2
kUNK_CUTOFF = 3


class LanguageModel:
    def train_seen(self, word, count=1):
        """
        Tells the language model that a word has been seen @count times.  This
        will be used to build the final vocabulary.
        """
        assert not self._vocab_final, \
            "Trying to add new words to finalized vocab"

        raise NotImplementedError

    def vocab_lookup(self, word):
        """
        Given a word, provides a vocabulary representation.  All words below the
        cutoff threshold shold have the same value.  All other words should be
        unique and consistent.
        """
        assert self._vocab_final, \
            "Vocab must be finalized before looking up words"

        raise NotImplementedError

    def censor(self, sentence):
        """
        Given a sentence, yields a sentence suitable for training or testing.
        Prefix the sentence with <s>, replace words not in the vocabulary with
        <UNK>, and end the sentence with </s>.
        """
        yield self.vocab_lookup("<s>")
        for ii in sentence:
            yield self.vocab_lookup(ii)
        yield self.vocab_lookup("</s>")

    def

if __name__ == "__main__":
    lm = LanguageModel(kLM_ORDER, kUNK_CUTOFF)

    for ii in nltk.corpus.brown.words():
        lm.train_seen(ii)

    lm.fix_vocab()

    for ii in nltk.corpus.brown.sentences():
        for jj in ii:
            for kk, ww in lm.generate_contexts(jj):
                lm.add_context(kk, ww)

    for ii in nltk.corpus.treebank.sentences():
        scores = (lm.perplexity(ii, lm.mle),
                  lm.perplexity(ii, lm.laplace),
                  lm.perplexity(ii, lm.katz))
        print(scores, " ".join(ii))
