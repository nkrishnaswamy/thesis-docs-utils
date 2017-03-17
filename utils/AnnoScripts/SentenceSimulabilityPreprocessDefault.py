import sys,os
from optparse import OptionParser
import xml.etree.ElementTree as elemTree

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest = "input", default = "", help = "Input text file", metavar = "INPUT")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output XML file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()

    inFile = open(options.input)
    outFile = open(options.output, "w")
    
    outFile.write("""<?xml version="1.0" encoding="UTF-8" ?>\n""")
    outFile.write("<SentenceSimulabilityTask>\n<TEXT><![CDATA[")
    #print file.readlines()
    lines = inFile.readlines()
    for i in range(len(lines)):
        outFile.write(lines[i])
    outFile.write("]]></TEXT>\n<TAGS>\n")
    startIndex = 0
    endIndex = 0
    for i in range(len(lines)):
        endIndex += len(lines[i])
        outFile.write("""<SENTENCE id="S%i" spans="%i~%i" text="%s" judgement="fine" />\n""" % (i,startIndex,endIndex-1,lines[i][:-1]))
        startIndex += len(lines[i])
    outFile.write("</TAGS>\n</SentenceSimulabilityTask>")

if __name__ == "__main__":
    main()