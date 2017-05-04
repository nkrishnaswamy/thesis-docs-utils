import sys,os
import argparse
import sqlite3
import json
import math
import re
import aenum
import itertools

class ParamType(aenum.Enum):
    discrete = 1
    continuous = 2

pred_matches = [
    "move_%",
    "turn_%",
    "roll_%",
    "slide_%",
    "spin_%",
    "lift_%",
    "put_%_touching_%",
    "put_%_on_%",
    "put_%_in_%",
    "put_%_near_%",
    "lean_%_on_%",
    "lean_%_against_%",
    "flip_%_on_edge",
    "flip_%_at_center",
    "close_%",
    "open_%"
]

# predicate string : (param labels)
param_labels = [
    "MotionSpeed",
    "MotionManner",
    "TranslocSpeed",
    "TranslocDir",
    "RotSpeed",
    "RotAngle",
    "RotAxis",
    "RotDir",
    "SymmetryAxis",
    "PlacementOrder",
    "RelOrientation",
    "RelOffset"
]

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

discrete_possible_values = {
    "MotionManner" : ("turn %","roll %",
                      "slide %","spin %",
                      "lift %","put % touching %",
                      "put % on %","put % in %",
                      "put % near %","lean % on %",
                      "lean % against %","flip % on edge",
                      "flip % at center"),
    "RotAxis" : ("X","Y","Z"),
    "RotDir" : (+1,-1),
    "SymmetryAxis" : ("X","Y","Z"),
    "PlacementOrder" : ((1,2),(2,1)),
    "RelOrientation" : ("%left%","%right%",
                        "%behind%","%in_front%",
                        "%on%")
}

counts = {}
probabilities = {}
judgements = {}     # acceptability judgements, indexed by video file path
sentences = {}    # candidate sentences, indexed by video file path

def main():
    parser = argparse.ArgumentParser(description='Generic evaluator for VoxSim HET results for thesis')
    parser.add_argument('--features', '-f', help='features database')
    parser.add_argument('--judgements', '-j', help='judgements database (converted from csv)')
    parser.add_argument('--quantiles', '-q', help='q for q-quantiles for continuous variables')
    parser.add_argument('--output_db', '-o', help='output probabilities database')
    args = parser.parse_args()
    
    global features_db
    features_db = args.features
    global judgements_db
    judgements_db = args.judgements
    global q
    q = int(args.quantiles)
    global output_db
    output_db = args.output_db

    init_dicts()

    pretty_print_dict(counts)
    pretty_print_dict(probabilities)

    create_output_db()

#Looking for:
# Task 1:
# the probability that an arbitrary judge rates as acceptable a video generated for a given predicate using a given parameter set
# Task 2:
# the probability that an arbitrary judge rates as acceptable a candidate predicate for a given video generated with a given parameter set
# confusion matrix for right choice/wrong choice

#for each predicate
#what parameters need to be conditioned on?
# (only those assigned value during simulation assignment)
# (take this from the parameter list in thesis, not automatic from db)

#need numbers for:
# COUNT # examples of predicate $PRED (for all $PRED) (= N)
# COUNT predicate = $PRED AND param value = $PARAM (for all $PARAM that ever coocur with $PRED) (= C(B))
# COUNT judgement = acceptable AND predicate = $PRED AND param value = $PARAM (for all $PARAM that ever coocur with $PRED) (= C(A,B))

#Formula notes:
# P(A|B) = P(A,B)/P(B)
# P(A,B) = C(A,B)/N
# P(B) = C(B)/N

def init_dicts():
    features_conn = sqlite3.connect(features_db)
    features_cur = features_conn.cursor()
    
    judgements_conn = sqlite3.connect(judgements_db)
    judgements_cur = judgements_conn.cursor()
    
    max_dist = 3.265
    epsilon = 0.000001
    
    for pred in pred_matches:
        continuous_values = {}
        intervals = {}
        inputs = [] # this gets cleared with each new predicate
        
        # Get COUNT # videos of predicate $PRED (for all $PRED) (= N)
        judgements_cur.execute('SELECT * FROM het2_results WHERE Video LIKE "%/' + pred + '/%"')
        results = judgements_cur.fetchall()
        counts["COUNT # viewed examples of predicate '%s'" % pred] = len(results)
        
#        pretty_print_dict(counts)

        # for each HIT where input string matches $PRED, index the answers and video paths by the input string
        for result in results:
            # in inputs, only store the current predicate being examined
            if result[27] not in inputs:
                inputs.append(result[27])
            
            if result[27] not in judgements:
                judgements[result[27]] = [result[31]]
                # candidate sentences for this video path = [three sentences]
                sentences[result[27]] = list(result[28:31])
            else:
                judgements[result[27]].append(result[31])
            
#            RightAnswerCountsprint_dict(sentences)
#            pretty_print_dict(judgements)

#            print result[27].split('/')[6].split('.')[0]
            features_cur.execute('SELECT * FROM VideoDBEntry WHERE FilePath LIKE "%/' + result[27].split('/')[6].split('.')[0] + '%"')
            video = features_cur.fetchone()
            if video[7] is not None:
                dict = json.loads(video[7])
                dict = {str(k): str(v) for k, v in dict.iteritems() if len(v) > 0}
                for feature in dict:
                    if param_type[feature] == ParamType.continuous:
                        if feature not in continuous_values:
                            intervals[feature] = []
                            if isinstance(vector_magnitude(parseable_to_vector(dict[feature])), basestring):
                                continuous_values[feature] = [float(dict[feature])]
                            else:
#                                    print feature, vector_magnitude(parseable_to_vector(dict[feature]))
                                if vector_magnitude(parseable_to_vector(dict[feature])) < max_dist:
                                    continuous_values[feature] = [vector_magnitude(parseable_to_vector(dict[feature]))]
                        else:
                            if isinstance(vector_magnitude(parseable_to_vector(dict[feature])), basestring):
                                continuous_values[feature].append(float(dict[feature]))
                            else:
#                                    print feature, vector_magnitude(parseable_to_vector(dict[feature]))
                                if vector_magnitude(parseable_to_vector(dict[feature])) < max_dist:
                                    continuous_values[feature].append(vector_magnitude(parseable_to_vector(dict[feature])))

#                values = sorted(continuous_values)

        for feature in continuous_values:
            #print feature, continuous_values[feature]
            quantile = len(sorted(continuous_values[feature]))/q
            for i in range(q):
                if i == q-1:
                    #print feature, continuous_values[feature], len(continuous_values[feature]), quantile*(i+1)
                    intervals[feature].append((sorted(continuous_values[feature])[quantile*i],sorted(continuous_values[feature])[len(continuous_values[feature])-1]))
                else:
                    intervals[feature].append((sorted(continuous_values[feature])[quantile*i],sorted(continuous_values[feature])[quantile*(i+1)]-epsilon))
        print pred, intervals

        for input in inputs:
            agreements = {0 : 0, 1 : 0, 2 : 0}
            for answer in judgements[input]:
                for option in agreements:
                    if str(unichr(option+49)) in answer:
                        agreements[option] += 1
#            print input, agreements, sentences[input], sentences[input].index(input.split('/')[6].split('.')[0].split('-')[0].replace("_"," "))

            #print video_paths[input][i].split('/')[6].split('.')[0]

            features_cur.execute('SELECT * FROM VideoDBEntry WHERE FilePath LIKE "%/' + input.split('/')[6].split('.')[0] + '%"')
            video = features_cur.fetchone()
            if video [7] is not None:
                #print video[2]
                dict = json.loads(video[7])
                dict = {str(k): str(v) for k, v in dict.iteritems() if len(v) > 0}
                param_tag = "Predicate = %s;" % pred
                #print dict
                #print agreements[i]
                for feature in dict:
                    if param_type[feature] == ParamType.discrete:
                        if feature == "PlacementOrder":
                            #print dict[feature].split(',')[0],dict["MotionManner"].split('(')[1].split(',')[0]
                            if dict[feature].split(',')[0] == dict["MotionManner"].split('(')[1].split(',')[0]:
                                param_tag += " %s = %s;" % (feature, str([1,2]))
                            else:
                                param_tag += " %s = %s;" % (feature, str([2,1]))
                        else:
                            if dict[feature] == "+":
                                param_tag += " %s = %s;" % (feature, 1)
                            elif dict[feature] == "-":
                                param_tag += " %s = %s;" % (feature, -1)
                            else:
                                param_tag += " %s = %s;" % (feature, vector_magnitude(parseable_to_vector(dict[feature])).split('(')[0])
                        #print "%s : %s" % (entry, parseable_to_vector(dict[entry]))
                        #print pred, video[2]

                    if param_type[feature] == ParamType.continuous:
                        for j in range(len(intervals[feature])):
                            if isinstance(vector_magnitude(parseable_to_vector(dict[feature])), basestring):
                                if float(dict[feature]) >= intervals[feature][j][0] and float(dict[feature]) <= intervals[feature][j][1]:
                                    param_tag += " %s = I%s;" % (feature, j+1)
                                    break
                            else:
                                if vector_magnitude(parseable_to_vector(dict[feature])) >= intervals[feature][j][0] and vector_magnitude(parseable_to_vector(dict[feature])) <= intervals[feature][j][1]:
                                    param_tag += " %s = I%s;" % (feature, j+1)
                                    break
                                        
#                        print param_tag

                if "COUNT # viewed examples where %s" % (param_tag,) not in counts:
                    counts["COUNT # viewed examples where %s" % (param_tag,)] = len(judgements[input])
                else:
                    counts["COUNT # viewed examples where %s" % (param_tag,)] += len(judgements[input])

                for i in range(3):  # 3 options per video
                    for option in pred_matches:
                        r = option.replace("_"," ").replace("%",".+")
                        e = sentences[input][i]#[sentences[input].index(input.split('/')[6].split('.')[0].split('-')[0].replace("_"," "))]
                        if re.match(r,e):
                            choice = option
                            #print "COUNT # preferred predicate %s where %s" % (choice, param_tag,)
                            if "COUNT # preferred predicate %s where %s" % (choice, param_tag,) not in counts:
                                counts["COUNT # preferred predicate %s where %s" % (choice, param_tag,)] = agreements[i]
                            else:
                                counts["COUNT # preferred predicate %s where %s" % (choice, param_tag,)] += agreements[i]

                if "COUNT # picked right sentence where %s" % (param_tag,) not in counts:
                    counts["COUNT # picked right sentence where %s" % (param_tag,)] = agreements[sentences[input].index(input.split('/')[6].split('.')[0].split('-')[0].replace("_"," "))]
                else:
                    counts["COUNT # picked right sentence where %s" % (param_tag,)] += agreements[sentences[input].index(input.split('/')[6].split('.')[0].split('-')[0].replace("_"," "))]

                probabilities["PROB picked right sentence | %s" % (param_tag,)] = float(counts["COUNT # picked right sentence where %s" % (param_tag,)])/float(counts["COUNT # viewed examples where %s" % (param_tag,)])

    features_conn.close()
    judgements_conn.close()

def parseable_to_vector(parseable):
    if re.match("<.*;.*;.*>",parseable) is None:
        return parseable
    else:
        vec = parseable.split(';')
        for i in range(len(vec)):
            vec[i] = vec[i].strip().replace('<','').replace('>','')
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

def pretty_print_dict(dictionary):
    for key in sorted(dictionary):
        print key, " : ", dictionary[key]

    print "\n"

def create_output_db():
    connection = sqlite3.connect(output_db)

    with connection:
        cursor = connection.cursor()
        #Count = total count of occurances of with given predicate, parameter set
        cursor.execute("CREATE TABLE IF NOT EXISTS TotalCounts (Predicate TEXT, MotionSpeed TEXT, MotionManner TEXT, TranslocSpeed TEXT, TranslocDir TEXT, RotSpeed TEXT, RotAngle TEXT, RotAxis TEXT, RotDir TEXT, SymmetryAxis TEXT, PlacementOrder TEXT, RelOrientation TEXT, RelOffset TEXT, Count INTEGER)")
        
        #Count = total count of preferred original input sentence with given predicate, parameter set
        cursor.execute("CREATE TABLE IF NOT EXISTS RightAnswerCounts (Predicate TEXT, MotionSpeed TEXT, MotionManner TEXT, TranslocSpeed TEXT, TranslocDir TEXT, RotSpeed TEXT, RotAngle TEXT, RotAxis TEXT, RotDir TEXT, SymmetryAxis TEXT, PlacementOrder TEXT, RelOrientation TEXT, RelOffset TEXT, Count INTEGER)")
        
        #Count = total count of preferred predicate with given known input predicate, parameter set
        cursor.execute("CREATE TABLE IF NOT EXISTS PreferredPredCounts (Predicate TEXT, MotionSpeed TEXT, MotionManner TEXT, TranslocSpeed TEXT, TranslocDir TEXT, RotSpeed TEXT, RotAngle TEXT, RotAxis TEXT, RotDir TEXT, SymmetryAxis TEXT, PlacementOrder TEXT, RelOrientation TEXT, RelOffset TEXT, Choice TEXT, Count INTEGER)")
        
        #Prob = likelihood of preferring the original input sentence given predicate, parameter set
        cursor.execute("CREATE TABLE IF NOT EXISTS RightAnswerProbabilities (Predicate TEXT, MotionSpeed TEXT, MotionManner TEXT, TranslocSpeed TEXT, TranslocDir TEXT, RotSpeed TEXT, RotAngle TEXT, RotAxis TEXT, RotDir TEXT, SymmetryAxis TEXT, PlacementOrder TEXT, RelOrientation TEXT, RelOffset TEXT, Prob FLOAT)")
        
        #Prob = likelihood of preferring a predicate description given known input predicate, parameter set
        cursor.execute("CREATE TABLE IF NOT EXISTS PreferredPredProbabilities (Predicate TEXT, MotionSpeed TEXT, MotionManner TEXT, TranslocSpeed TEXT, TranslocDir TEXT, RotSpeed TEXT, RotAngle TEXT, RotAxis TEXT, RotDir TEXT, SymmetryAxis TEXT, PlacementOrder TEXT, RelOrientation TEXT, RelOffset TEXT, Choice TEXT, Prob FLOAT)")
        
        entries = []
        for key in sorted(counts):
            if "COUNT # viewed examples where" in key:
                parameter_set = key.replace("COUNT # viewed examples where","").strip().split(';')
                entry = ["","","","","","","","","","","","","",""]
                for parameter in parameter_set:
                    if parameter.split('=')[0].strip() == "Predicate":
                        entry[0] = parameter.split('=')[1].replace("_"," ").strip()
                    elif parameter.split('=')[0].strip() == "MotionSpeed":
                        entry[1] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "MotionManner":
                        entry[2] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "TranslocSpeed":
                        entry[3] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "TranslocDir":
                        entry[4] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotSpeed":
                        entry[5] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotAngle":
                        entry[6] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotAxis":
                        entry[7] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotDir":
                        entry[8] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "SymmetryAxis":
                        entry[9] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "PlacementOrder":
                        entry[10] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RelOrientation":
                        entry[11] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RelOffset":
                        entry[12] = parameter.split('=')[1].strip()

                    entry[13] = counts[key]
                            
                entries.append(entry)
                    
        cursor.executemany("INSERT INTO TotalCounts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", entries)
                  
        entries = []
        for key in sorted(counts):
            if "COUNT # picked right sentence where" in key:
                parameter_set = key.replace("COUNT # picked right sentence where","").strip().split(';')
                entry = ["","","","","","","","","","","","","",""]
                for parameter in parameter_set:
                    if parameter.split('=')[0].strip() == "Predicate":
                        entry[0] = parameter.split('=')[1].replace("_"," ").strip()
                    elif parameter.split('=')[0].strip() == "MotionSpeed":
                        entry[1] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "MotionManner":
                        entry[2] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "TranslocSpeed":
                        entry[3] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "TranslocDir":
                        entry[4] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotSpeed":
                        entry[5] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotAngle":
                        entry[6] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotAxis":
                        entry[7] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RotDir":
                        entry[8] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "SymmetryAxis":
                        entry[9] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "PlacementOrder":
                        entry[10] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RelOrientation":
                        entry[11] = parameter.split('=')[1].strip()
                    elif parameter.split('=')[0].strip() == "RelOffset":
                        entry[12] = parameter.split('=')[1].strip()

                    entry[13] = counts[key]
            
                entries.append(entry)

        cursor.executemany("INSERT INTO RightAnswerCounts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", entries)
        
        entries = []
        for key in sorted(counts):
            for pref in pred_matches:
                if "COUNT # preferred predicate %s where" % (pref,) in key:
                    parameter_set = key.replace("COUNT # preferred predicate %s where" % (pref,),"").strip().split(';')
#                    print parameter_set
                    entry = ["","","","","","","","","","","","","","",""]
                    for parameter in parameter_set:
                        if parameter.split('=')[0].strip() == "Predicate":
                            entry[0] = parameter.split('=')[1].replace("_"," ").strip()
                        elif parameter.split('=')[0].strip() == "MotionSpeed":
                            entry[1] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "MotionManner":
                            entry[2] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "TranslocSpeed":
                            entry[3] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "TranslocDir":
                            entry[4] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RotSpeed":
                            entry[5] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RotAngle":
                            entry[6] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RotAxis":
                            entry[7] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RotDir":
                            entry[8] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "SymmetryAxis":
                            entry[9] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "PlacementOrder":
                            entry[10] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RelOrientation":
                            entry[11] = parameter.split('=')[1].strip()
                        elif parameter.split('=')[0].strip() == "RelOffset":
                            entry[12] = parameter.split('=')[1].strip()
                    
                        entry[13] = pref
                        entry[14] = counts[key]

                    entries.append(entry)
        
        cursor.executemany("INSERT INTO PreferredPredCounts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", entries)

        for pred in pred_matches:
            cursor.execute("SELECT * FROM TotalCounts WHERE Predicate = ?", (pred.replace("_"," "),))
            results = cursor.fetchall()
            
            for result in results:
                print result
                params = [param_labels[p-1] for p in range(1,len(result)-2) if result[p] != ""]
                #print params
                for i in range(1,len(params)+1):
                    combinations = list(itertools.combinations(params,i))
                    print combinations
                    for combo in combinations:
                        cmd = "SELECT * FROM TotalCounts WHERE"
                        retain = ""
                        for param in combo:
                            retain += " %s = '%s' AND" % (param,result[param_labels.index(param)+1])
    #                    print cmd[:-5]
                        cursor.execute(cmd + retain[:-4])
                        total_count = 0
                        for match in cursor.fetchall():
                            total_count += match[-1]
        
                        cmd = "SELECT * FROM RightAnswerCounts WHERE"
                        cursor.execute(cmd+retain[:-4])
                        correct_count = 0
                        for match in cursor.fetchall():
                            correct_count += match[-1]
                                
                        print "PROB picked right sentence | %s = %s" % (retain[:-4], float(correct_count)/float(total_count),)

                        entry = [pred.replace("_"," "),"","","","","","","","","","","","",""]
                        parameter_set = retain.split("AND")
                        for parameter in parameter_set:
                            if parameter.split('=')[0].strip() == "Predicate":
                                entry[0] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "MotionSpeed":
                                entry[1] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "MotionManner":
                                entry[2] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "TranslocSpeed":
                                entry[3] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "TranslocDir":
                                entry[4] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RotSpeed":
                                entry[5] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RotAngle":
                                entry[6] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RotAxis":
                                entry[7] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RotDir":
                                entry[8] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "SymmetryAxis":
                                entry[9] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "PlacementOrder":
                                entry[10] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RelOrientation":
                                entry[11] = parameter.split('=')[1].replace("'","").strip()
                            elif parameter.split('=')[0].strip() == "RelOffset":
                                entry[12] = parameter.split('=')[1].replace("'","").strip()
                    
                        entry[13] = float(correct_count)/float(total_count)
#                        print entry
                        cursor.execute("INSERT INTO RightAnswerProbabilities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", entry)

                        for pref in pred_matches:
                            cmd = "SELECT * FROM PreferredPredCounts WHERE"
#                            print cmd + retain[:-4] + " AND Choice = " + pref
                            cursor.execute(cmd + retain[:-4] + " AND Choice = '" + pref + "'")
                            pref_count = 0
                            for match in cursor.fetchall():
                                print match
                                pref_count += match[-1]
                        
#                            print pref_count, total_count
                            print "PROB preferred predicate %s for %s | %s = %s" % (pref, pred.replace("_"," "), retain[:-4], float(pref_count)/float(total_count),)

                            entry = [pred.replace("_"," "),"","","","","","","","","","","","","",""]
                            parameter_set = retain.split("AND")
                            for parameter in parameter_set:
                                if parameter.split('=')[0].strip() == "Predicate":
                                    entry[0] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "MotionSpeed":
                                    entry[1] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "MotionManner":
                                    entry[2] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "TranslocSpeed":
                                    entry[3] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "TranslocDir":
                                    entry[4] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RotSpeed":
                                    entry[5] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RotAngle":
                                    entry[6] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RotAxis":
                                    entry[7] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RotDir":
                                    entry[8] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "SymmetryAxis":
                                    entry[9] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "PlacementOrder":
                                    entry[10] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RelOrientation":
                                    entry[11] = parameter.split('=')[1].replace("'","").strip()
                                elif parameter.split('=')[0].strip() == "RelOffset":
                                    entry[12] = parameter.split('=')[1].replace("'","").strip()
                        
                            entry[13] = pref
                            entry[14] = float(pref_count)/float(total_count)
#                            print entry
                            cursor.execute("INSERT INTO PreferredPredProbabilities VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", entry)

        connection.commit()

    connection.close()
if __name__ == "__main__":
    main()
