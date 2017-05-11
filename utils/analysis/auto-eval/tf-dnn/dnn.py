from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys,os
import argparse
import sqlite3
import json
import aenum
import itertools
import re
import urllib
import math
from nltk.classify import maxent
from nltk.metrics import *

import tfidf

class ParamType(aenum.Enum):
    discrete = 1
    continuous = 2

discrete_possible_values = {
    "MotionManner" : ["turn","roll",
                      "slide","spin",
                      "lift","put touching",
                      "put on","put in",
                      "put near","lean on",
                      "lean against","flip on edge",
                      "flip at center"],
    "RotAxis" : ["X","Y","Z"],
    "RotDir" : ["1","-1"],
    "SymmetryAxis" : ["X","Y","Z"],
    "PlacementOrder" : ["1,2","2,1"],
    "RelOrientation" : ["left","right",
                      "behind","in_front",
                      "on"]
}

param_type = {
    "MotionSpeed" : ParamType.continuous,
    "MotionManner" : ParamType.discrete,
    "TranslocSpeed" : ParamType.continuous,
    "TranslocDir" : ParamType.continuous,
    "RotSpeed" : ParamType.continuous,
    "RotAngle" : ParamType.continuous,
    "RotAxis" : ParamType.discrete,
    "RotDir" : ParamType.discrete,
    "SymmetryAxis" : ParamType.discrete,
    "PlacementOrder" : ParamType.discrete,
    "RelOrientation" : ParamType.discrete,
    "RelOffset" : ParamType.continuous
}

def main():
    parser = argparse.ArgumentParser(description='DNN')
    parser.add_argument('--features', '-f', help='features database')
    parser.add_argument('--granularity', '-g', help = 'granularity level (1 = coarse, label with predicate only; 2 = middle, label with predicate and adjunct if available; 3 = fine, label with full parse)')
    parser.add_argument('--kfactor', '-k', help = 'Exclude every kth entry (training), include every kth entry (testing)')
    parser.add_argument('--offset', '-n', help = 'Exclude every kth entry (training), include every kth entry (testing), offset by n')
    parser.add_argument('--steps', '-s', help = '# training steps')
    parser.add_argument('--combined', '-c', action='store_true', help = 'Use combined classifier')
    parser.add_argument('--weighted', '-w', action='store_true', help = 'Weight features')
    parser.add_argument('--discrete_weights_only', '-d', action='store_true', help = 'Only weight discrete features')
    parser.add_argument('--omit_features', '-o', action='store_true', help = 'Omit feature information (use weights only as features)')
    args = parser.parse_args()
    
    global features_db
    features_db = args.features
    
    global granularity
    granularity = int(args.granularity)

    global kfactor
    kfactor = int(args.kfactor)
    
    global offset
    offset = int(args.offset)
    
    global steps
    steps = int(args.steps)
    
    global combined
    combined = args.combined
    
    global weighted
    weighted = args.weighted
    
    global discrete_weights_only
    discrete_weights_only = args.discrete_weights_only
    
    global omit_features
    omit_features = args.omit_features

    slice = 10

    features_conn = sqlite3.connect(features_db)
    features_cur = features_conn.cursor()
    
    features_cur.execute('SELECT * FROM VideoDBEntry')
    results = features_cur.fetchall()[:slice]
    
    features_cur.execute('SELECT * FROM AlternateSentences')
    candidates = [list(r[2:]) for r in features_cur.fetchall()[:slice]]
    
    features_conn.close()
    
    for i in range(len(candidates)):
        for j in range(len(candidates[i])):
            candidates[i][j] = granularize(candidates[i][j],granularity)

    print candidates

    global dev_test
    dev_test = results[offset::kfactor]
    test_candidates = candidates[offset::kfactor]
    
    global train
    train = [r for r in results if r not in dev_test]
    train_canditates = [candidates[i] for i in range(len(candidates)) if candidates[i] not in dev_test]
    
    global label_set
    label_set = []
    
    true_labels = []

    dev_test = featurize(dev_test)
    train = featurize(train)
    
    for entry in dev_test:
        if granularize(entry["Input"],granularity) not in label_set:
            label_set.append(granularize(entry["Input"],granularity))
        true_labels.append(granularize(entry["Input"],granularity))
    
    for entry in train:
        if granularize(entry["Input"],granularity) not in label_set:
            label_set.append(granularize(entry["Input"],granularity))

    for candidate_set in candidates:
        for candidate in candidate_set:
            if candidate not in label_set:
                label_set.append(candidate)

    print label_set

    wide_columns = []
    deep_columns = []
    feature_columns = []

    global tfidf_bias
    tfidf_bias = {}

    for key in sorted(param_type):
        tfidf_bias[key] = tfidf.compute_tfidf(features_db,key)
        if param_type[key] == ParamType.discrete:
            if weighted:
                sparse_feature = tf.contrib.layers.sparse_column_with_keys(
                   key, discrete_possible_values[key],
                   dtype=tf.int64 if type(discrete_possible_values[key][0]) is int else tf.string)
                weighted_feature = tf.contrib.layers.weighted_sparse_column(sparse_id_column=sparse_feature, weight_column_name=key+"IDFWeight")
                feature_columns.append(tf.contrib.layers.one_hot_column(sparse_feature))
                deep_columns.append(tf.contrib.layers.one_hot_column(sparse_feature))
                feature_columns.append(tf.contrib.layers.embedding_column(weighted_feature, dimension=1))
                deep_columns.append(tf.contrib.layers.embedding_column(weighted_feature, dimension=1))
            else:
                feature_columns.append(tf.contrib.layers.one_hot_column(tf.contrib.layers.sparse_column_with_keys(
                    key, discrete_possible_values[key],
                    dtype=tf.int64 if type(discrete_possible_values[key][0]) is int else tf.string)))
                deep_columns.append(tf.contrib.layers.one_hot_column(tf.contrib.layers.sparse_column_with_keys(
                    key, discrete_possible_values[key],
                    dtype=tf.int64 if type(discrete_possible_values[key][0]) is int else tf.string)))
#            feature_columns.append(tf.contrib.layers.sparse_column_with_keys(key+"IDFWeight", [tfidf_bias[key]], dtype=tf.float32))
#            deep_columns.append(tf.contrib.layers.sparse_column_with_keys(key+"IDFWeight", [tfidf_bias[key]], dtype=tf.float32))
#            feature_columns.append(tf.contrib.layers.real_valued_column(key+"IDFWeight", dimension=1))
#            deep_columns.append(tf.contrib.layers.real_valued_column(key+"IDFWeight", dimension=1))
        elif param_type[key] == ParamType.continuous:
            feature_columns.append(tf.contrib.layers.real_valued_column(key, dimension=1))
            wide_columns.append(tf.contrib.layers.real_valued_column(key, dimension=1))
#            feature_columns.append(tf.contrib.layers.real_valued_column(key+"IDFWeight", dimension=1))
#            wide_columns.append(tf.contrib.layers.real_valued_column(key+"IDFWeight", dimension=1))


#    print train


    if combined:
        classifier = tf.contrib.learn.DNNLinearCombinedClassifier(
                                                      linear_feature_columns=wide_columns,
                                                      dnn_feature_columns=deep_columns,
                                                      dnn_hidden_units=[10, 20, 20, 10],
                                                      n_classes=len(label_set))
    else:
        classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                    hidden_units=[10, 20, 20, 10],
                                                    n_classes=len(label_set))



    print tfidf_bias

    classifier.fit(input_fn=train_input_fn, steps=steps)

#    print classifier.evaluate(input_fn=test_input_fn, steps=1)

    testing_data = test_input_fn
    predicted_labels = []
    probs = classifier.predict_proba(input_fn=testing_data)
    predictions = classifier.predict(input_fn=testing_data)
    pred_probs = [[i for i in predictions],[j for j in probs]]

    reference = []
    restricted_test = []
    unrestricted_test = []
    for i in range(len(pred_probs[0])):
        candidate_indices = [label_set.index(test_candidates[i][j]) for j in range(len(test_candidates[i]))]
        restricted_best_prob = pred_probs[1][i][candidate_indices[0]]
        restricted_best_match = label_set[candidate_indices[0]] if max(pred_probs[1][i]) > 0.0 else "None"
        restricted_best_match_index = candidate_indices[0] if max(pred_probs[1][i]) > 0.0 else None
        
        for k in candidate_indices:
            if pred_probs[1][i][k] > restricted_best_prob:
                restricted_best_prob = pred_probs[1][i][k]
                restricted_best_match = label_set[k]
                restricted_best_match_index = k
        
        multiple_choice = []
        multiple_choice.append(candidate_indices.index(restricted_best_match_index)+1)
        for k in candidate_indices:
            if pred_probs[1][i][k] == restricted_best_prob:
                if candidate_indices.index(k)+1 not in multiple_choice:
                    multiple_choice.append(candidate_indices.index(k)+1)

        print "\nCandidates in restricted choice set: %s (indices %s)" % (test_candidates[i],candidate_indices)
        
        multiple_choice = list(set(sorted(multiple_choice)))
        print "Prediction with multiple choice option: %s" % (sorted(multiple_choice))
        print "Prediction with restricted choice set: %s (index %s, probability %s)" % (restricted_best_match,restricted_best_match_index,restricted_best_prob)
        
        unrestricted_best_match = label_set[pred_probs[0][i]] if max(pred_probs[1][i]) > 0.0 else "None"
        unrestricted_best_match_index = pred_probs[0][i] if max(pred_probs[1][i]) > 0.0 else None
        print "Prediction from unrestricted choice set: %s (index %s, probability %s)" % (unrestricted_best_match,unrestricted_best_match_index,max(pred_probs[1][i]))
        print "True label: %s" % true_labels[i]
#        pred_probs[1][i],)
        predicted_labels.append(label_set[pred_probs[0][i]])
            
        restricted_test.append(restricted_best_match == true_labels[i])
        unrestricted_test.append(unrestricted_best_match == true_labels[i])
        reference.append(True)
    
    print "\nChoice set restricted:\n", ConfusionMatrix(reference,restricted_test)
    print "\nChoice set unrestricted:\n", ConfusionMatrix(reference,unrestricted_test)
#    accuracy_score = classifier.evaluate(input_fn=test_input_fn,
#                                     steps=1)["accuracy"]

#    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


#    raw = urllib.urlopen("http://download.tensorflow.org/data/iris_training.csv").read()
#    with open("iris_training.csv",'w') as f:
#        f.write(raw)
#
#    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
#        filename="iris_training.csv",
#        target_dtype=np.int,
#        features_dtype=np.float32)
#
#    print training_set.data

def train_input_fn():
    return input_fn(train)

def test_input_fn():
    return input_fn(dev_test)

def input_fn(df):
    print df

    labels = []
    continuous_cols = {}
    discrete_cols = {}
    feature_cols = {}
    
    labeled_features = {}
    
    for param in param_type:
        if param_type[param] == ParamType.continuous:
            feature_cols[param] = []
            continuous_cols[param] = []
        elif param_type[param] == ParamType.discrete:
            feature_cols[param] = []
            discrete_cols[param] = []


    for param in param_type:
#        continuous_cols[param+"IDFWeight"] = []
#        if param_type[param] == ParamType.continuous:
#            continuous_cols[param+"IDFWeight"] = []
        if param_type[param] == ParamType.discrete:
            feature_cols[param+"IDFWeight"] = []
            discrete_cols[param+"IDFWeight"] = []


    for d in df:
        print d
#        print "# labels = %s" % len(df)
        labels.append(label_set.index(granularize(d["Input"],granularity)))
#        feature_cols[param+"IDFWeight"].append(tfidf_bias[param])
#        continuous_cols[param+"IDFWeight"].append(tfidf_bias[param])
        for param in param_type:
            if param_type[param] == ParamType.continuous:
#                feature_cols[param+"IDFWeight"].append(tfidf_bias[param])
#                continuous_cols[param+"IDFWeight"].append(tfidf_bias[param])
                if param in d:
                    if omit_features:
                        feature_cols[param].append(tfidf_bias[param])
                        continuous_cols[param].append(tfidf_bias[param])
                    else:
                        if weighted and not discrete_weights_only:
                            feature_cols[param].append(d[param])
                            continuous_cols[param].append(d[param])
                        else:
                            feature_cols[param].append(d[param] * tfidf_bias[param])
                            continuous_cols[param].append(d[param] * tfidf_bias[param])
                else:
                    feature_cols[param].append(0.)
                    continuous_cols[param].append(0.)
            elif param_type[param] == ParamType.discrete:
                feature_cols[param+"IDFWeight"].append(tfidf_bias[param])
                discrete_cols[param+"IDFWeight"].append(tfidf_bias[param])
                if param in d:
                    if omit_features:
                        feature_cols[param].append(d[param])
                        discrete_cols[param].append(d[param])
                    else:
                        feature_cols[param].append("")
                        discrete_cols[param].append("")
                else:
                    feature_cols[param].append("")
                    discrete_cols[param].append("")

    print tfidf_bias

    print continuous_cols,discrete_cols,labels,len(labels)

#        for feature in sorted(d):
#            if feature not in labeled_features:
#                labeled_features[feature] = [d[feature]]
#            else:
#                labeled_features[feature].append(d[feature])
#
#    print labeled_features



#    for entry in labeled_features:
#        labels.append(labeled_features["Input"])
#        for param in param_type:
#            if param_type[param] == ParamType.continuous:
#                continuous_cols[param] = 0.
#            elif param_type[param] == ParamType.discrete:
#                discrete_cols[param] = ""
#
#        for param in param_type:
#            if param_type[param] == ParamType.continuous:
#                if param in labeled_features:
#                    continuous_cols[param] = labeled_features[param]
#            elif param_type[param] == ParamType.discrete:
#                if param in labeled_features:
#                    discrete_cols[param] = labeled_features[param]
#
#    print continuous_cols,discrete_cols
#
    continuous_cols = {k: tf.constant(continuous_cols[k]) for k in continuous_cols}
    print "Continuous:", continuous_cols

#    discrete_cols = {k: tf.constant(discrete_cols[k]) for k in discrete_cols}
    if combined:
        discrete_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(len(discrete_cols[k]))],
            values=discrete_cols[k],
            dense_shape=[len(discrete_cols[k]), 1])
            for k in discrete_cols}
    else:
        feature_cols = {k: tf.SparseTensor(
            indices=[[i, 0] for i in range(len(feature_cols[k]))],
            values=feature_cols[k],
            dense_shape=[len(feature_cols[k]), 1])
            for k in feature_cols}
#
##    continuous_cols = {k: tf.constant(labeled_features[k]) for k in param_type if k in labeled_features and param_type[k] == ParamType.continuous}
#
    print "Discrete:", discrete_cols
#
##    discrete_cols = {k: tf.SparseTensor(
##        indices=[[i, 0] for i in range(len(labeled_features[k]))],
##        values=labeled_features[k],
##        dense_shape=[len(labeled_features[k]), 1])
##        for k in param_type if k in labeled_features and param_type[k] == ParamType.discrete}
#
##    print discrete_cols
#
    if combined:
        feature_cols = dict(continuous_cols.items() + discrete_cols.items())
    print "len(feature_cols):", len(feature_cols)
#    for key in feature_cols:
#        tfidf_bias.append([tfidf.compute_tfidf(features_db,key)])

#    return
#
#    print feature_cols
#
    labels = tf.constant(labels)
#
    print labels
#
    return feature_cols,labels

def tfidf_bias_fn(cols):
#    tf.nn.bias_add(tf.nn.relu(cols),tf.constant(tfidf_bias))
#    for item in tf.nn.relu(cols):
#        pass

    print tfidf_bias
    print cols,tf.constant(tfidf_bias)
    return tf.nn.relu(cols)

def granularize(input,granularity):
    output = input
    if granularity == 1:
        output = input.split()[0]
    elif granularity == 2:
        if "on" in input.split():
            output = input.split()[0] + " on"
        elif "in" in input.split():
            output = input.split()[0] + " in"
        elif "against" in input.split():
            output = input.split()[0] + " against"
        elif "touching" in input.split():
            output = input.split()[0] + " touching"
        elif "left" in input.split():
            output = input.split()[0] + " left"
        elif "right" in input.split():
            output = input.split()[0] + " right"
        elif "in_front" in input.split():
            output = input.split()[0] + " in front"
        elif "behind" in input.split():
            output = input.split()[0] + " behind"
        elif "near" in input.split():
            output = input.split()[0] + " near"
        else:
            output = input.split()[0]

    output = output.replace("in_front","in front of")
    output = output.replace("left","left of")
    output = output.replace("right","right of")

    return output

def featurize(raw):
    features = []
    for result in raw:
        if result[7] is not None:
            dict = {str(k): parseable_to_vector(str(v)) for k, v in json.loads(result[7]).iteritems() if len(v) > 0}
            if "PlacementOrder" in dict and len(dict["PlacementOrder"]) > 0:
                if dict["PlacementOrder"].split(',')[0] == dict["MotionManner"].split('(')[1].split(',')[0]:
                    dict["PlacementOrder"] = "1,2"
                else:
                    dict["PlacementOrder"] = "2,1"
        
            for feature in dict:
                if len(dict[feature]) > 0:
                    if param_type[feature] == ParamType.discrete:
                        if feature == "MotionManner":
                            if ',' in dict[feature] and '(' in dict["MotionManner"].split('(')[1]:
                                dict[feature] = dict["MotionManner"].split('(')[0] + " " + dict["MotionManner"].split(',')[1].split('(')[0]
                            else:
                                dict[feature] = dict["MotionManner"].split('(')[0]
                        elif feature == "RotDir":
                            if dict[feature] == "+":
                                dict[feature] = "1"
                            elif dict[feature] == "-":
                                dict[feature] = "-1"
                            else:
                                dict[feature] = dict[feature]
                    else:
                        if feature == "TranslocDir" or feature == "RelOffset":
                            dict[feature] = vector_magnitude(dict[feature])
                        if isinstance(dict[feature],basestring):
                            dict[feature] = float(dict[feature])
    
        dict["Input"] = result[2]
        features.append(dict)

    print features
    return features

def parseable_to_vector(parseable):
    if re.match("<.*;.*;.*>",parseable) is None:
        return parseable
    else:
        vec = parseable.split(';')
        for i in range(len(vec)):
            vec[i] = float(vec[i].strip().replace('<','').replace('>',''))
        return tuple(vec)

def vector_magnitude(vec):
    if not isinstance(vec, tuple) or len(vec) != 3:
        return vec
    else:
        dist = 0
        
        for i in range(len(vec)):
            dist += float(vec[i]) * float(vec[i])

        dist = math.sqrt(dist)

    return dist

if __name__ == "__main__":
    main()
