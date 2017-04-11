#!/bin/bash
STARTTIME=$(date +%s)
for i in {1..3};
do
    for j in {5..7};
    do
        cd ../preprocessing/; python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -o het-features.pickle -e -g $i; python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -C AlternateSentences -o het-testing.pickle -c -e -b -g $i; cd ../analysis/; python MaxEntBaseline.py -t ../preprocessing/het-features.pickle -s maxent-baseline-classifier-g$i-n$j.pickle -k 10 -n $j; python MaxEntBaseline.py -t maxent-baseline-classifier-g$i-n$j.pickle -T ../preprocessing/het-testing.pickle -k 10 -n $j > output-g$i-n$j.txt;
    done
done
ENDTIME=$(date +%s)
echo "Elapsed time: $(($ENDTIME - $STARTTIME)) seconds"
