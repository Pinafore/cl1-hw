import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.corpus.reader.wordnet import Lemma

from typing import Union

class Oracle:
    def __init__(self, synset: Synset):
        self._num_queries = 0

        self._synset = synset

    def num_queries(self) -> int:
        return self._num_queries
        
    def for_all(self, relationship, arguments) -> bool:
        self._num_queries += 1

        relationship_handle = getattr(self._synset, relationship)
        result = relationship_handle()

        return all(x in result for x in arguments)

    def there_exists(self, relationship: str, arguments) -> bool:
        self._num_queries += 1

        relationship_handle = getattr(self._synset, relationship)
        result = relationship_handle()

        return any(x in result for x in arguments)

    def cnf_eval(self, relationships: str, arguments) -> bool:
        self._num_queries += 1

        final = True
        for clause_relationships, clause_arguments in zip(relationships, arguments):

            intermediate = False
            for relationship, argument in zip(clause_relationships, clause_arguments):
                handle = getattr(self._synset, relationship)
                result = handle()

                intermediate = intermediate or (argument in result)
            final = final and intermediate
        return final


if __name__ == "__main__":
    dog_oracle = Oracle(wn.synset('dog.n.01'))
    all_tests = [[wn.synset(x) for x in ['corgi.n.01', 'pug.n.01', 'dalmatian.n.02']],
                     [wn.synset(x) for x in ['man.n.01', 'pug.n.01']]]
    for test in all_tests:
        print("Are %s all dogs?" % str(test))
        print(dog_oracle.for_all('hyponyms', test))

    for ll in [wn.lemma('dog.n.01.Canis_familiaris'), wn.lemma('man.n.01.adult_male')]:
        print("Is %s a lemma of dog?" % str(ll))
        print(dog_oracle.there_exists('lemmas', [ll]))

    # True CNF
    relationships = [["member_holonyms", "member_holonyms"], ["part_meronyms"]]
    arguments = [[wn.synset('anatidae.n.01'), wn.synset('pack.n.06')], [wn.synset('flag.n.07')]]
    print("Does a dog belong in a pack or a flock and have a poofy tail?")
    print(dog_oracle.cnf_eval(relationships, arguments))
  
    relationships.append(['instance_hypernyms'])
    arguments.append([wn.synset('president_of_the_united_states.n.01')])
    print("Does a dog belong in a pack or a flock and have a poofy tail and is a president of the US?")
    print(dog_oracle.cnf_eval(relationships, arguments))
                        
