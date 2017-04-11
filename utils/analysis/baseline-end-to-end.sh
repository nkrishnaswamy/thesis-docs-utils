#!/bin/bash
STARTTIME=$(date +%s)
for i in {1..3};
do
	cd ../preprocessing/; python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -o het-features.pickle -e -g $i; python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -C AlternateSentences -o het-testing.pickle -c -e -b -g $i; cd ../analysis/; python MaxEntBaseline.py -t ../preprocessing/het-features.pickle -s maxent-baseline-classifier-g$i.pickle -k 10; python MaxEntBaseline.py -t maxent-baseline-classifier-g$i.pickle -T ../preprocessing/het-testing.pickle -k 10 > output-g$i.txt;
done
ENDTIME=$(date +%s)
echo "Elapsed time: $(($ENDTIME - $STARTTIME)) seconds"
