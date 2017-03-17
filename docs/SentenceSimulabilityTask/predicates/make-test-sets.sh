#!/bin/bash 
for f in *.txt;
	do python ../../make_auto_sentence_list.py -i $f -o ../test-sets/$f -r 3; 
done; 
cd ../test-sets/; 
for old in *.txt; 
	do mv $old `basename $old .txt`.py; 
done