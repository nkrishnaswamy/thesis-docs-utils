import sys,os
from optparse import OptionParser
import pickle

import read_database

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-F", "--features_table", dest = "features_table", default = "", help = "Table containing event features", metavar = "FEATURESTABLE")
    parser.add_option("-e", "--filter_features", action = "store_true", dest = "filter_features", default = False, help = "filter out non-occurring features in training data")
    parser.add_option("-C", "--candidates_table", dest = "candidates_table", default = "", help = "Table containing candidate sentences", metavar = "CANDIDATESTABLE")
    parser.add_option("-b", "--filter_candidates", action = "store_true", dest = "filter_candidates", default = False, help = "filter out non-occurring features in testing data")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output pickle file", metavar = "OUTPUT")
    parser.add_option("-c", "--as_candidates", action = "store_false", dest = "as_features", default = True,
                      help = "vectorize as candidates as opposed to features")
    parser.add_option("-g", "--granularity", dest = "granularity", default = 3,
                      help = "granularity level (1 = coarse, label with predicate only; 2 = middle, label with predicate and adjunct if available; 3 = fine, label with full parse)")
    (options, args) = parser.parse_args()
    
    '''make features:
        python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -o het-features.pickle -e -g 1'''
    '''make candidates:
        python vectorize.py -d ../../../../Underspecification\ Analysis/data/het-test-set.db -F VideoDBEntry -C AlternateSentences -o het-testing.pickle -c -e -b -g 1'''
    
    database = options.database
    features_table = options.features_table
    filter_features = options.filter_features
    candidates_table = options.candidates_table
    filter_candidates = options.filter_candidates
    outfile = open(options.output,"w")
    as_candidates = not options.as_features
    granularity = int(options.granularity)
    
    features_data = read_database.read_database(database,features_table,filter_features)

    if as_candidates:
        candidates_data = read_database.read_database(database,candidates_table,filter_candidates)

    vectors = []

    errors = 0
    error_list = []
    for i in range(len(features_data)):
        print i, features_data[i][7]
        if features_data[i][7] is not {}:
            if granularity == 3:
                label = features_data[i][2]
            elif granularity == 2:
                label = features_data[i][5]
                if " on " in features_data[i][2]:
                    label += " on"
                elif " in " in features_data[i][2]:
                    label += " in"
                elif " against " in features_data[i][2]:
                    label += " against"
                elif " touching " in features_data[i][2]:
                    label += " touching"
                elif " near " in features_data[i][2]:
                    label += " near"
            elif granularity == 1:
                label = features_data[i][5]
            #print entry
            features = type_dict(features_data[i][7])
            
            if as_candidates:
                candidates = list(candidates_data[i][2:])

                for j in range(len(candidates)):
                    if granularity == 1:
                        candidates[j] = candidates[j].split()[0]
                    elif granularity == 2:
                        if "on" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " on"
                        elif "in" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " in"
                        elif "against" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " against"
                        elif "touching" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " touching"
                        elif "left" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " left"
                        elif "right" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " right"
                        elif "in_front" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " in front"
                        elif "behind" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " behind"
                        elif "near" in candidates[j].split():
                            candidates[j] = candidates[j].split()[0] + " near"
                        else:
                            candidates[j] = candidates[j].split()[0]
                    
                    candidates[j] = candidates[j].replace("in_front","in front of")
                    candidates[j] = candidates[j].replace("left","left of")
                    candidates[j] = candidates[j].replace("right","right of")
                    
                print i, (features, tuple(candidates), label)
                vectors.append((features, tuple(candidates), label))
                if label not in candidates:
                    errors += 1
                    error_list.append("vectors[%s]: '%s' not in %s" % (len(vectors)-1, label, candidates))
            else:
                print i, (features,label)
                vectors.append((features,label))

    print "%s errors" % errors
    print error_list

    pickle.dump(vectors,outfile)
    outfile.close()

def type_dict(dictionary):
    features = {}
    #num_features = 0;
    for key in dictionary:
        if key == "MotionSpeed":
            features[key] = float(dictionary[key]) if len(dictionary[key]) > 0 else 0
        elif key == "MotionManner":
            features[key] = dictionary[key];
        elif key == "TranslocSpeed":
            features[key] = float(dictionary[key]) if len(dictionary[key]) > 0 else 0
        elif key == "TranslocDir":
            features[key] = parsable_vec_to_tuple(dictionary[key]) if len(dictionary[key]) > 0 else (0,0,0)
        elif key == "RotSpeed":
            features[key] = float(dictionary[key]) if len(dictionary[key]) > 0 else 0
        elif key == "RotAngle":
            features[key] = float(dictionary[key]) if len(dictionary[key]) > 0 else 0
        elif key == "RotAxis":
            features[key] = dictionary[key];
        elif key == "RotDir":
            if dictionary[key] == "+":
                features[key] = 1;
            elif dictionary[key] == "-":
                features[key] = -1;
            elif len(dictionary[key]) > 0:
                features[key] = int(dictionary[key]);
            else:
                features[key] = 0
        elif key == "SymmetryAxis":
            features[key] = dictionary[key];
        elif key == "PlacementOrder":
            features[key] = dictionary[key];
        elif key == "RelOrientation":
            features[key] = dictionary[key];
        elif key == "RelOffset":
            features[key] = parsable_vec_to_tuple(dictionary[key]) if len(dictionary[key]) > 0 else (0,0,0)

        #num_features += 1

    #features["NumFeatures"] = num_features

    return features

def parsable_vec_to_tuple(parseable):
    return tuple([float(n) for n in parseable.replace('<','').replace('>','').split(';')])

if __name__ == "__main__":
    main()
