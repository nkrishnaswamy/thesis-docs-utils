import sys,os
from optparse import OptionParser
import pickle
from nltk.classify import maxent
from nltk.metrics import *
import random

def main():
    parser = OptionParser()
    parser.add_option("-t", "--training", dest = "training", default = "", help = "Training data", metavar = "TRAINING")
    parser.add_option("-T", "--testing", dest = "testing", default = "", help = "Testing data", metavar = "TESTING")
    parser.add_option("-s", "--save", dest = "save", default = "", help = "Save classifier destination", metavar = "SAVE")
    parser.add_option("-k", "--kfactor", dest = "kfactor", default = "", help = "Exclude every kth entry (training), include every kth entry (testing)", metavar = "KFACTOR")
    (options, args) = parser.parse_args()
    
    '''train: python MaxEntBaseline.py -t ../preprocessing/het-features.pickle -s maxent-baseline-classifier.pickle -k 10'''
    '''test: python MaxEntBaseline.py -t maxent-baseline-classifier.pickle -T ../preprocessing/het-testing.pickle -k 10'''
    
    training = open(options.training, "rb")
    kfactor = int(options.kfactor)
    
    if options.save is '':
        if options.testing is not '':
            testing = open(options.testing, "rb")
            classifier = pickle.load(training)
            training.close()
            testing_data = pickle.load(testing)
            testing.close()
            separated_testing_data = [testing_data[i] for i in range(len(testing_data)) if i % kfactor == 0]
            #random.shuffle(testing_data)
            evaluate(classifier,separated_testing_data)
    else:
        save = open(options.save, "wb")
        training_data = pickle.load(training)
        training.close()
        separated_training_data = [training_data[i] for i in range(len(training_data)) if i % kfactor != 0]
        #random.shuffle(training_data)
        train_classifier(separated_training_data,save)

def train_classifier(training_data,save_file):
    classifier = maxent.MaxentClassifier.train(training_data[:200], 'GIS', min_lldelta=.0001,max_iter=1000)
    pickle.dump(classifier,save_file)
    save_file.close()

def evaluate(classifier,testing_data):
    features = [f[0] for f in testing_data]
    candidates = [f[1] for f in testing_data]
    labels = [f[2] for f in testing_data]
    
    reference = []
    restricted_test = []
    unrestricted_test = []
    for i in range(len(testing_data[:20])):
        pdist = classifier.prob_classify(features[i])
        restricted_best_prob = pdist.prob(candidates[i][0])
        restricted_best_match = candidates[i][0]
        print "\nCandidates in restricted choice set:"
        for candidate in candidates[i]:
            print candidate, "\t", pdist.prob(candidate)
            if pdist.prob(candidate) > restricted_best_prob:
                restricted_best_prob = pdist.prob(candidate)
                restricted_best_match = candidate
        unrestricted_best_match = classifier.classify(features[i])
        print "Prediction from restricted choice set: %s" % (restricted_best_match,)
        print "Prediction from unrestricted choice set: %s" % (unrestricted_best_match,)
        print "True label: %s" % (labels[i],)
        restricted_test.append(restricted_best_match == labels[i])
        unrestricted_test.append(unrestricted_best_match == labels[i])
        reference.append(True)

    print "\nChoice set restricted:\n", ConfusionMatrix(reference,restricted_test)
    print "\nChoice set unrestricted:\n", ConfusionMatrix(reference,unrestricted_test)

if __name__ == "__main__":
    main()
