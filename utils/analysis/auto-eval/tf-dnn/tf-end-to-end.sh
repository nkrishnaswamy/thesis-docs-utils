#!/bin/bash
STARTTIME=$(date +%s)
for i in {3..3};
do
    for j in {6..9};
    do
        for k in {1000,2000,3000,4000,5000}
        do
            python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -cw > output-g$i-n$j-s$k-cw.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -cwd > output-g$i-n$j-s$k-cwd.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -c > output-g$i-n$j-s$k-c.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -co > output-g$i-n$j-s$k-co.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -w > output-g$i-n$j-s$k-w.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -wd > output-g$i-n$j-s$k-wd.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k > output-g$i-n$j-s$k-0.txt; python dnn.py -f ../../../../docs/UnderspecificationAnalysis/het-test-set.db -g $i -k 10 -n $j -s $k -o > output-g$i-n$j-s$k-o.txt;
        done
    done
done
ENDTIME=$(date +%s)
echo "Elapsed time: $(($ENDTIME - $STARTTIME)) seconds"
