import sys,os
import argparse
import sqlite3
import json
import math
import re
import aenum
import itertools

def main():
    parser = argparse.ArgumentParser(description='TF-IDF calculator for underspecified motion features')
    parser.add_argument('--corpus', '-c', help='features database')
    parser.add_argument('--feature', '-f', help='feature label')
    args = parser.parse_args()
    
    global corpus
    corpus = args.corpus
    global feature
    feature = args.feature
    
    print compute_tfidf(corpus,feature)
    
def compute_tfidf(corpus,feature):
    tf = {}
    idf = 1
    N = 0
    
    tfidf = {}

    connection = sqlite3.connect(corpus)

    with connection:
        cursor = connection.cursor()
    
        cursor.execute("SELECT * FROM VideoDBEntry")
        results = cursor.fetchall()
        N = len(results)
    
        for result in results:
            tf[result[1]] = 0
            if result[7] is not None:
                dict = {str(k): str(v) for k, v in json.loads(result[7]).iteritems() if len(v) > 0}
#                print dict
                for label in dict:
                    if label == feature:
                        tf[result[1]] = 1
                        idf += 1

    connection.close()

#    print idf

    idf = math.log10(float(N)/float(idf))

#    pretty_print_dict(tf)
#    print idf

    sum_tfidf = 0.0

    for key in tf:
        tfidf[key] = tf[key] * idf
        sum_tfidf += tfidf[key]

#    pretty_print_dict(tfidf)

    avg_tfidf = float(sum_tfidf)/len(tfidf)
    return idf

def pretty_print_dict(dictionary):
    for key in sorted(dictionary):
        print key, " : ", dictionary[key]
    
    print "\n"

if __name__ == "__main__":
    main()
