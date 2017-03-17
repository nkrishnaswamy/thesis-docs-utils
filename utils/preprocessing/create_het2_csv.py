import sys,os
import sqlite3
from optparse import OptionParser
import read_database

forms_file = None

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-p", "--urlPrefix", dest = "url_prefix", default = "", help = "URL prefix", metavar = "PREFIX")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output text file", metavar = "OUTPUT")
   
    (options, args) = parser.parse_args()
    
    database = options.database
    url_prefix = options.url_prefix
    output = open(options.output,"w")

    alternates = read_alternate_sentences(database)

    output.write("VIDEO,DESCRIPTION1,DESCRIPTION2,DESCRIPTION3\n")
    for alternate in alternates:
       output.write("%s,%s,%s,%s\n" % tuple([url_prefix + alternate[0] + ".mp4"] + list(alternate[1:])))

def read_alternate_sentences(database):
    connection = sqlite3.connect(database)
    
    cursor = connection.cursor()
    
    cursor.execute('SELECT * FROM AlternateSentences')
    results = cursor.fetchall()
    
    connection.close()
    
    output = []
    for result in results:
        output.append((result[1],result[2],result[3],result[4]))
    
    return output

#    contents = read_database.read_database(database)
#
#    descriptions = []
#    for entry in contents:
#        if entry[1] not in descriptions:
#            descriptions.append(entry[1])
#
#    output.write("DESCRIPTION,VIDEOA,VIDEOB,VIDEOC\n")
#    for description in descriptions:
#        files = [url_prefix + entry[0]+".mp4" for entry in contents if entry[1] == description]
#        if len(files) == 3:
#            output.write("%s,%s,%s,%s\n" % tuple([description] + files))

if __name__ == "__main__":
    main()