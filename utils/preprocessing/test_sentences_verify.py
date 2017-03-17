import sys,os
from optparse import OptionParser
import change_to_forms_test

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest = "input", default = "", help = "Input text file", metavar = "INPUT")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output text file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()
    
    test_sentences = open(options.input)

    logical_forms = open(options.output, "w")

    for line in test_sentences.readlines():
        logical_forms.write(change_to_forms_test.parse_sent(line.split('[')[0]) + "\n")

if __name__ == "__main__":
    main()
