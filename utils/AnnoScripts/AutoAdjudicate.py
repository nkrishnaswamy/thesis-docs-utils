# Given three input XML files, writes the sentences accepted or rejected by majority rule

import sys,os
from optparse import OptionParser
import xml.etree.ElementTree as elemTree

values = {'ungrammatical':0,
                'nonsense':1,
                'awkward':2,
                'fine':3}

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest = "input", default = "", help = "Input text file", metavar = "INPUT", nargs = 3)
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output XML file", metavar = "OUTPUT", nargs = 2)
    (options, args) = parser.parse_args()

    anno1 = open(options.input[0])
    anno2 = open(options.input[1])
    anno3 = open(options.input[2])

    accepted = open(options.output[0], "a")
    rejected = open(options.output[1], "a")

    sentencesList = []
    
    for i in range(3):
        tree = elemTree.parse(options.input[i])
        root = tree.getroot()
        for child in root:
            if child.tag == "TAGS":
                sentencesList.append(list(child.iter("SENTENCE")))

    numAccepted = 0
    numRejected = 0
    for i in range(len(sentencesList[0])):
        scores = []
        for j in range(len(sentencesList)):
            scores.append(values[sentencesList[j][i].attrib['judgement']])
        if sum(scores) >= 5 and scores.count(1) < 2:
            print "accept", sentencesList[j][i].attrib['text'], scores, sum(scores)
            accepted.write("%s %s\n" % (sentencesList[j][i].attrib['text'], str(scores)))
            numAccepted += 1
        else:
            print "reject", sentencesList[j][i].attrib['text'], scores, sum(scores)
            rejected.write("%s %s\n" % (sentencesList[j][i].attrib['text'], str(scores)))
            numRejected += 1

    accepted.close()
    rejected.close()

    accepted = open(options.output[0])
    rejected = open(options.output[1])

    numAccepted = len(accepted.readlines())-1
    numRejected = len(rejected.readlines())-1
    print "Total accepted: ", numAccepted
    print "Total rejected: ", numRejected

if __name__ == "__main__":
    main()