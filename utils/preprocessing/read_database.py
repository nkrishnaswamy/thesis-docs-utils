import sys,os
import sqlite3
from optparse import OptionParser
import json

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-t", "--table", dest = "table", default = "", help = "Table", metavar = "TABLE")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output text file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()
    
    database = options.database
    table = options.table
    #outfile = open(options.output, "w")
    
    read_database(database,table,True)

def read_database(database,table,filter):
    print database, table
    connection = sqlite3.connect(database)

    cursor = connection.cursor()

    cursor.execute('SELECT * FROM ' + table)
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
        entry = ()
        if len(result) > 7:
            for field in result:
                #if result[7] is not None:
                try:
                    entry = entry + (filter_dict(field,filter),)
                except:
                    entry = entry + (field,)
    
            output.append(entry)
        #   if entry is not ():
        #       output.append(entry)
        else:
            for field in result:
                entry = entry + (field,)

            output.append(entry)
        #    if entry is not ():
        #       output.append(entry)

    return output

def filter_dict(db_value,filter):
    if db_value is not None:
        dict = json.loads(db_value)
        if filter == True:
            dict = {str(k): str(v) for k, v in dict.iteritems() if len(v) > 0}
        else:
            dict = {str(k): str(v) for k, v in dict.iteritems()}
    else:
        dict = {}


    return dict

if __name__ == "__main__":
    main()
