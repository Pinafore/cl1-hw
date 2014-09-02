from collections import defaultdict
from csv import DictReader, DictWriter

import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import TreebankWordTokenizer

kTOKENIZER = TreebankWordTokenizer()

def morphy_stem(word):
    """
    Simple stemmer
    """
    stem = wn.morphy(word)
    if stem:
        return stem.lower()
    else:
        return word.lower()

class FeatureExtractor:
    def __init__(self):
        """
        You may want to add code here
        """
        
        None
    
    def features(self, text):
        d = defaultdict(int)
        for ii in kTOKENIZER.tokenize(text):
            d[morphy_stem(ii)] += 1
        return d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='subsample this amount')
    args = parser.parse_args()
    
    # Create feature extractor (you may want to modify this)
    fe = FeatureExtractor()
    
    # Read in training data
    train = DictReader(open("train.csv", 'r'))
    
    # Split off dev section
    dev_train = []
    dev_test = []
    full_train = []

    for ii in train:
        if args.subsample < 1.0 and int(ii['id']) % 100 > 100 * args.subsample:
            continue
        feat = fe.features(ii['text'])
        if int(ii['id']) % 5 == 0:
            dev_test.append((feat, ii['cat']))
        else:
            dev_train.append((feat, ii['cat']))
        full_train.append((feat, ii['cat']))

    # Train a classifier
    print("Training classifier ...")
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train)
    # classifier = nltk.classify.MaxentClassifier.train(dev_train, 'IIS', trace=3, max_iter=1000)

    right = 0
    total = len(dev_test)
    for ii in dev_test:
        prediction = classifier.classify(ii[0])
        if prediction == ii[1]:
            right += 1
    print("Accuracy on dev: %f" % (float(right) / float(total)))

    # Retrain on all data
    classifier = nltk.classify.NaiveBayesClassifier.train(dev_train + dev_test)
    
    # Read in test section
    test = {}
    for ii in DictReader(open("test.csv")):
        test[ii['id']] = classifier.classify(fe.features(ii['text']))

    # Write predictions
    o = DictWriter(open('pred.csv', 'w'), ['id', 'pred'])
    o.writeheader()
    for ii in sorted(test):
        o.writerow({'id': ii, 'pred': test[ii]})
