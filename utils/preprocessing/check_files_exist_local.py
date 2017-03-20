import sys,os
import sqlite3
from optparse import OptionParser
import json
import read_database

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    (options, args) = parser.parse_args()
    
    database = options.database
    
    connection = sqlite3.connect(database)
    
    cursor = connection.cursor()

    cursor.execute('SELECT * FROM VideoDBEntry')
    results = cursor.fetchall()

    connection.close()

    for result in results:
    #filePath is [1]
        if os.path.isfile("data/video/" + result[1]+ ".mp4") == False:
            print "Missing file: %s ID %s" % (result[1], result[0])

if __name__ == "__main__":
    main()
