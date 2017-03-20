import sys,os
import sqlite3
from optparse import OptionParser
import json
import read_database
import urllib2

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-u", "--url", dest = "url", default = "", help = "URL prefix", metavar = "URL PREFIX")
    (options, args) = parser.parse_args()
    
    database = options.database
    url = options.url
    
    connection = sqlite3.connect(database)
    
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM VideoDBEntry')
    results = cursor.fetchall()

    connection.close()

    for i in range(len(results)):
        if i % 10 == 0:
            print i
    #filePath is [1]
        try:
            result = urllib2.urlopen(url+results[i][1]+".mp4")
        except urllib2.HTTPError, e:
            print e.code, e.msg
            print "Error: %s ID %s" % (url+results[i][1]+".mp4", results[i][0])

if __name__ == "__main__":
    main()
