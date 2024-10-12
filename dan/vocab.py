
class Vocab:
    def __init__(self, counts, specials=[], max_tokens=-1, vocab_cutoff=0):
        self.default_index = None        
        self.lookup = {}
        self.reverse_lookup = {}
        
        for idx, word in enumerate(specials):
            self.lookup[word] = idx
            self.reverse_lookup[idx] = word

        if max_tokens <= 0:
            max_tokens = len(counts)

        lookup = max(self.lookup.values())
        valid_tokens = list(counts.most_common(max_tokens))
        # Sort so that it's 
        valid_tokens.sort(key=lambda x: (-x[1], x[0]))
        for word, count in valid_tokens:
            if vocab_cutoff <= 0 or (vocab_cutoff > 0 and count > vocab_cutoff):
                lookup += 1
                self.lookup[word] = lookup
                self.reverse_lookup[lookup] = word

    def set_default_index(self, idx):
        self.default_index = idx

    def __len__(self):
        return len(self.lookup)

    def __contains__(self, key):
        return key in self.lookup

    def __getitem__(self, key):
        if key in self.lookup:
            return self.lookup[key]
        elif self.default_index is not None:
            return self.default_index

    def lookup_token(self, word):
        return self.reverse_lookup[word]
    
    @staticmethod
    def build_vocab_from_iterator(iterator, specials=[], max_tokens=-1, vocab_cutoff=2):
        from nltk.lm import Vocabulary
        from collections import Counter

        counts = Counter()
        for doc in iterator:
            counts.update(doc)

        return Vocab(counts, specials=specials, max_tokens=max_tokens, vocab_cutoff=vocab_cutoff)

if __name__ == "__main__":
    from guesser import kTOY_DATA
    from nltk.tokenize import word_tokenize

    tokenizer = word_tokenize

    vocab = Vocab.build_vocab_from_iterator([word_tokenize(doc["text"]) for doc in kTOY_DATA["mini-train"]],
                                            specials=["<unk>"], max_tokens=10)
    print("Unk:", vocab['<unk>'])
    vocab.set_default_index(vocab["<unk>"])
        
    for doc in kTOY_DATA["mini-dev"]:
        print(" ".join("%s_%i" % (x, vocab[x]) for x in word_tokenize(doc["text"])))
