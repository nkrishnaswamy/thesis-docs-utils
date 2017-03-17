# Given an XML template of data formatted for the SentenceSimulabilityTask DTD for MAE and a file of raw data of the form <sentence>\n<judgement>, copies the judgements onto the template for any sentences that exist in both the raw data file and the template file

import sys,os
from optparse import OptionParser
import xml.etree.ElementTree as elemTree

def main():
    parser = OptionParser()
    parser.add_option("-i", "--input", dest = "input", default = "", help = "Input text file", metavar = "INPUT", nargs = 2)
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output XML file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()

    #print options.input[0],options.input[1]

    templateFile = open(options.input[0])
    rawFile = open(options.input[1])
    
    data = {}
    templateLines = templateFile.readlines()
    rawLines = rawFile.readlines()
    for i in range(0,len(rawLines),3):
        if rawLines[i] in templateLines:
            data[rawLines[i][:-1]] = rawLines[i+1][:-1].lower().split()[0]

    #print data
    
    outFile = open(options.output, "w")
    
    outFile.write("""<?xml version="1.0" encoding="UTF-8" ?>\n""")
    outFile.write("<SentenceSimulabilityTask>\n<TEXT><![CDATA[")
    #print file.readlines()
    for line in templateLines:
        outFile.write(line)
    outFile.write("]]></TEXT>\n<TAGS>\n")
    startIndex = 0
    endIndex = 0
    for i in range(len(templateLines)):
        endIndex += len(templateLines[i])
        outFile.write("""<SENTENCE id="S%i" spans="%i~%i" text="%s" judgement="%s" />\n""" % (i,startIndex,endIndex-1,templateLines[i][:-1],data[templateLines[i][:-1]]))
        startIndex += len(templateLines[i])
    outFile.write("</TAGS>\n</SentenceSimulabilityTask>")

if __name__ == "__main__":
    main()