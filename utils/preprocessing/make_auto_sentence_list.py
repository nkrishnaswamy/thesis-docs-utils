import sys,os
import re
from optparse import OptionParser

import change_to_forms_test

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest = "input", default = "", help = "Input text file", metavar = "INPUT")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output code file", metavar = "OUTPUT")
    parser.add_option("-r", "--repeat", dest = "repeat", default = "", help = "# times to repeat each input", metavar = "REPEAT")
    (options, args) = parser.parse_args()
    
    test_sentences = open(options.input)
    sentence_list = open(options.output, "w")
    r = int(options.repeat)
    
    s = 0
    
    sentence_list.write("sentences = [\n")

    lines = test_sentences.readlines()
    for i in range(len(lines)-1):
        line = lines[i].split('[')[0].replace("\n","").strip()
        if verify_sent(line):
            for j in range(r):
                sentence_list.write('\t"' + line + '",' + "\n")
            s += 1

    line = lines[-1].split('[')[0].replace("\n","").strip()
    if verify_sent(line):
        for j in range(r-1):
            sentence_list.write('\t"' + line + '",' + "\n")
        sentence_list.write('\t"' + line + '"' + "\n")
        s += 1

    sentence_list.write("\t]")

    print "%s\t%s sentences" % (options.input, s)

def verify_sent(sent):
    arity = {"grasp" : 1,
        "hold" : 1,
        "touch" : 1,
        "move" : 1,
        "turn" : 1,
        "roll" : 1,
        "slide" : 1,
        "spin" : 1,
        "lift" : 1,
        "stack" : 2,
        "put" : 2,
        "lean" : 2,
        "flip" : 1,
        "close" : 1,
        "open" : 1
    }
    
    valid = True
    
    logical_form = change_to_forms_test.parse_sent(sent)

    if logical_form.count('(') == logical_form.count(')'):
        pred = logical_form.split('(')[0]
        match_objs = re.compile(r"(?<=\()[^()]*(?=[\),])")
        if arity[pred] > 1 and match_objs.findall(logical_form)[1:] == match_objs.findall(logical_form)[:-1]:
            valid = False

    return valid


if __name__ == "__main__":
    main()
