from time import sleep
from guesser import Guesser
from parameters import GuesserParameters
from string import ascii_lowercase
from math import floor
from collections import defaultdict

import tarfile
import json
from json import JSONDecodeError
from tqdm import tqdm
import logging


from baseconv import BaseConverter


kCACHE_MISS = "CACHE_MISS"
from nltk.corpus import stopwords
from guesser import alphanum

def clean_probs(log_probs):
    probs = [y for x, y in log_probs]
    return sum(probs) / len(probs)

class GprGuesser(Guesser):
    """
    Class that uses the OpenAI API to generate answers to questions with hints
    from our own guessers.  Cache the results because we're cheap and want
    reproducability.
    """

    def __init__(self, cache_filename, num_examples=2, num_shards=9999, shard_vocab=ascii_lowercase, shard_prefix_length=5, save_every=50):
        """

        @param num_examples: How many retrieval results to include in GPT prompt
        """
        
        self.client = None
        self.retrievers = {}
        self.cache = {}
        self.num_queries = 0
        self.save_every = save_every
        self.last_save = -1
        self.num_shards = num_shards
        self.num_examples = num_examples
        self.cache_filename = cache_filename
        self.stopwords = set(stopwords.words("english"))
        for ii in ["one", "man"]:
            self.stopwords.add(ii)

        self.num_shards = num_shards
        self.shard_vocab = shard_vocab
        self.shard_prefix_length = shard_prefix_length
        self.shard_converter = BaseConverter(shard_vocab)

    def clean_for_shard(self, query):
        words = alphanum.split(query.lower())
        nospace = "".join(x for x in words if not x in self.stopwords)
        clean = "".join(x for x in nospace if x in self.shard_vocab)
        
        if len(clean) < self.shard_prefix_length:
            clean += "".join([self.shard_vocab[0]] * (self.shard_prefix_length - len(clean)))
        
        return clean[:self.shard_prefix_length]

    def shard(self, string):
        """
        Shard the string based on prefix of the string.  
        """

        clean_version = self.clean_for_shard(string)            
        value = int(self.shard_converter.decode(clean_version))
        shard = floor((value / (len(self.shard_vocab) ** self.shard_prefix_length + 1)) * self.num_shards)

        assert shard >= 0
        assert shard < self.num_shards, "%i outside of range" % shard

        return shard

              



    @staticmethod
    def clean(raw_question):
        clean = raw_question
        for old, new in [("\xa0", " ")]:
            clean = clean.replace(old, new)
        return clean
    
    def __call__(self, question, n_guesses=1):
        """
        Generate a guess, but grab from the cache first if it's available
        """
        # Remove non-breaking spaces
        question = self.clean(question)

        if self.num_queries % self.save_every == 0:
            self.save()
        
        # Check the cache, return it from there if we have it
        if question not in self.cache:
            result = None
            while result is None:
                    result = kCACHE_MISS
                    # logging.info("No cache found for: |%s|" % question)                                
                    
            if result != kCACHE_MISS:
                self.cache[question] = result

        if question in self.cache:
            return [{"guess": self.cache[question]["guess"],
                     "confidence": clean_probs(self.cache[question]["confidence"])}]
        else:  # If we get here, this means that we couldn't query GPT and it's not cached
            assert result == kCACHE_MISS
            return [{"guess": "", "confidence": 0.0}]

    def save(self, cache_filename=None, queries=None, suffix=".json", force=False):
        """
        Save the API results to a file to save money and time for the future
        """

        force_write = (cache_filename is not None and queries is not None) or force

        if cache_filename is None:
            cache_filename=self.cache_filename
        if queries is None:
            queries=self.cache
        
        # Save if we have something to save and we haven't already saved it
        # alternately, if we have specified a different filename, always save
        if (self.num_queries > 0 and self.last_save != self.num_queries) or force_write:
            logging.info("Made %i new queries, saving to %s [cf: %s / queries: %s]" % (self.num_queries, cache_filename, cache_filename, str(queries is None)))
            shards = defaultdict(dict)
            for ii in queries:
                shard = self.shard(ii)
                shards[shard][ii] = self.cache[ii]

            for shard in tqdm(shards):
                filename = "%s%05i%s" % (cache_filename, shard, suffix)
                with open(filename, 'w') as outfile:
                    json_object = json.dumps(shards[shard], indent=2)
                    outfile.write(json_object)

            tar = tarfile.open("%s.tar.gz" % cache_filename, 'w:gz')
            for shard in tqdm(shards):
                filename = "%s%05i%s" % (cache_filename, shard, suffix)
                tar.add(filename)
            tar.close()
        

        self.last_save = self.num_queries
        

            
    def load(self):
        """
        Load the cache of search results from a file
        """

        clean = {}

        try:
            tar = tarfile.open("%s.tar.gz" % self.cache_filename, 'r:gz')
        except tarfile.ReadError:
            logging.debug("Empty cache, creating empty")
            return self.cache
            
        for ii in tar.getmembers():
            if ii.name.endswith(".pkl") or "/._" in ii.name:
                logging.debug("Skipping %s" % ii)
                continue
            else:
                logging.debug("Reading from %s" % ii)
            try:
                with tar.extractfile(ii) as infile:
                    raw_text = infile.read().decode('utf-8', errors='ignore')
                    json_object = json.loads(raw_text)
                    clean = {}
                    # Deal with non-breaking space issue
                    for ii in json_object:
                        clean[ii.replace("\xa0", " ")] = json_object[ii]
            except (IOError, JSONDecodeError) as e:
                json_object = {}
                print("Failed to load cache")

            self.cache.update(json_object)
            self.cache.update(clean)
            logging.debug("Reading %09i entries from %s, cache size is now %09i" %
                          (len(json_object), ii, len(self.cache)))

        tar.close()
        logging.info("%i entries added to cache" % len(self.cache))
        return self.cache



        prompt = [instructions] + prompt
    

if __name__ == "__main__":
    import argparse
    import gzip
    import logging
    from buzzer import runs

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default="buzztrain")
    
    # parser.add_argument('--source_jsongz', type=str, default="../data/qanta.buzztrain.json.gz")
    parser.add_argument('--build_cache', action="store_true", help="Save cache", default=False)
    # parser.add_argument('--cache', type=str, default="../models/buzztrain_gpr_cache")
    parser.add_argument('--cache_copy', type=str, default=None)
    parser.add_argument('--run_length', type=int, default=100)
    parser.add_argument('--limit', type=int, default=-1)
    flags = parser.parse_args()
    
    gg = GprGuesser(cache_filename="../models/%s_gpr_cache" % flags.fold)
    gg.load()

    with gzip.open("../data/qanta.%s.json.gz" % flags.fold) as infile:
        questions = json.load(infile)

    print("Loaded %i question" % len(questions))
        
    misses = 0
    hits = 0
    print("Generating runs of length %i" % flags.run_length)

    if flags.limit > 0:
        questions = questions[:flags.limit]

    queries = set()
    for qq in tqdm(questions):
        for rr in runs(qq["text"], flags.run_length):
            clean = gg.clean(rr)
            hit = clean in gg.cache
            if hit:
                queries.add(clean)
                hits += 1
            else:
                if flags.build_cache:
                    gg(rr)
                misses += 1

    gg.save(force=flags.build_cache)
    print("---------------------")
    print(hits / (hits + misses))

    logging.info("Hit ratio: %f" % (hits / (hits + misses)))

    if flags.cache_copy:
        print("Saving to %s" % flags.cache_copy)
        gg.save(cache_filename=flags.cache_copy, queries=queries)
