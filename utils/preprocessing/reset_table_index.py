import sys,os
import sqlite3
from optparse import OptionParser
import json

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-t", "--table", dest = "table", default = "", help = "Table to reset", metavar = "TABLE")
    parser.add_option("-i", "--index", dest = "index", default = "", help = "Index to reset to", metavar = "INDEX")
    (options, args) = parser.parse_args()
    
    database = options.database
    table = options.table
    index = int(options.index)
    
    connection = sqlite3.connect(database)
    
    cursor = connection.cursor()

    cursor.execute('UPDATE SQLITE_SEQUENCE SET seq = ? WHERE name = ?',(index,table))

    connection.commit()

    connection.close()

if __name__ == "__main__":
    main()
