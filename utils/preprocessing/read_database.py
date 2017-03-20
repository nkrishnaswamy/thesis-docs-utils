import sys,os
import sqlite3
from optparse import OptionParser
import json

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output text file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()
    
    database = options.database
    #outfile = open(options.output, "w")
    
    read_database(database)

def read_database(database):
    print database
    connection = sqlite3.connect(database)

    cursor = connection.cursor()

    cursor.execute('SELECT * FROM VideoDBEntry')
    results = cursor.fetchall()
    
    connection.close()

    output = []
    for result in results:
#        print result[1] # filename
#        print result[2] # input string
#        print result[3] # parse
#        print result[4] # object-resolved parse
#        print result[5] # event predicate
#        print result[6] # objects
#        print filter_dict(result[7])    # param values
        output.append((result[1],result[2],result[3],
                       result[4],result[5],result[6],
                       filter_dict(result[7])))
                      
    return output

def filter_dict(db_value):
    if db_value is not None:
        dict = json.loads(db_value)
        dict = {str(k): str(v) for k, v in dict.iteritems() if len(v) > 0}
    else:
        dict = {}


    return dict

if __name__ == "__main__":
    main()
