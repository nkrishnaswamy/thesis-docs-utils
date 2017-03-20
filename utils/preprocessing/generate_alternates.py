import sys,os
import sqlite3
from optparse import OptionParser
import re
import random
import read_database

forms_file = None

def main():
    parser = OptionParser()
    parser.add_option("-d", "--database", dest = "database", default = "", help = "Database", metavar = "DATABASE")
    parser.add_option("-f", "--forms", dest = "forms_file", default = "", help = "Master logical forms file", metavar = "FORMS")
    parser.add_option("-o", "--output", dest = "output", default = "", help = "Output text file", metavar = "OUTPUT")
    (options, args) = parser.parse_args()
    
    database = options.database
    
    global forms_file
    forms_file = options.forms_file

    contents = read_database.read_database(database)

    for entry in contents:
        sentence_choices = []
#        for field in entry:
#            print field
#        print "\n"
        sentence_choices.append(entry[1])
        
        other_choices = [entry[1]]
        if search_alternate_manner(entry) != "":
            if (search_alternate_manner(entry) not in other_choices):
                other_choices.append(search_alternate_manner(entry))
#        print other_choices

        if (search_rel_orientation(entry) not in other_choices):
            other_choices.append(search_rel_orientation(entry))
#        print other_choices

        if entry[4] == "put":
            if (alternate_on_in(entry) not in other_choices):
                other_choices.append(alternate_on_in(entry))
#        print other_choices

        if entry[4] == "lean":
            if (alternate_on_against(entry) not in other_choices):
                other_choices.append(alternate_on_against(entry))
#        print other_choices

        if entry[4] == "stack":
            if (alternate_placement_order(entry) not in other_choices):
                other_choices.append(alternate_placement_order(entry))
#        print other_choices

        underspecifications = search_underspecifications(entry)
        other_choices += underspecifications
        
#        print "Options: " + str(other_choices)
        if len(other_choices) > 2:
            while len(sentence_choices) < 3:
                choice = random.choice(other_choices)
                if choice not in sentence_choices:
                    sentence_choices.append(choice)
#                print sentence_choices
        else:
            for choice in other_choices:
                if choice not in sentence_choices:
                    sentence_choices.append(choice)

        while len(sentence_choices) < 3:
            choice = find_pairwise_candidate(entry)
            while choice in sentence_choices:
                choice = find_pairwise_candidate(entry)
                #print choice, sentence_choices
            sentence_choices.append(choice)
                
        random.shuffle(sentence_choices)
                
        connection = sqlite3.connect(database)
            
        cursor = connection.cursor()
            
        cursor.execute('CREATE TABLE IF NOT EXISTS AlternateSentences(Id INTEGER PRIMARY KEY, FilePath TEXT, Sentence1 TEXT, Sentence2 TEXT, Sentence3 TEXT)')

        cursor.execute('SELECT * FROM AlternateSentences WHERE FilePath = ?', (entry[0],))
        result = cursor.fetchall()
        if len(result) == 0:
            cursor.execute('INSERT INTO AlternateSentences(FilePath,Sentence1,Sentence2,Sentence3) VALUES(?,?,?,?)',
                           (entry[0],sentence_choices[0],sentence_choices[1],sentence_choices[2]))
            connection.commit()
            print "Entry added:\nVideo file: %s" % entry[0]
            for i in range(len(sentence_choices)):
                print "[%s]: %s" % (str(i+1), sentence_choices[i])
                print "\n"
        
        connection.close()

def search_alternate_manner(entry):
    alternate_manner = ""
    if ("MotionManner" in entry[6]):
        alternate_manner = entry[6]["MotionManner"]
    
    matches = re.search(r"(?P<pred>.+)\((?P<obj1>.+),(?P<rel>.+)\((?P<obj2>.+)\)\)",alternate_manner)
    if matches is not None:
        if matches.groupdict()["obj1"] == matches.groupdict()["obj2"]:
            alternate_manner = "%s(%s,%s)" % (matches.groupdict()["pred"],
                                                  matches.groupdict()["obj1"],matches.groupdict()["rel"])

    alternate_manner = alternate_manner.replace('('," the ").replace(','," ").replace("_"," ").replace(')',"").replace("edge","on edge").replace("center","at center")
    alternate_manner = re.sub(r"[0-9]+","",alternate_manner)
    return alternate_manner

def search_rel_orientation(entry):
    rel_orientation = ""
    if ("RelOrientation" in entry[6]):
        rel_orientation = entry[6]["RelOrientation"]
    
    rel_orientation = entry[1].replace("touching",rel_orientation)
    return rel_orientation

def alternate_on_in(entry):
    on_in = entry[1]
    if (" on " in entry[1]):
        on_in = on_in.replace(" on the "," in the ")
    elif (" in " in entry[1]):
        on_in = on_in.replace(" in the "," on the ")
    
    return on_in

def alternate_on_against(entry):
    on_against = entry[1]
    if (" on " in entry[1]):
        on_against = on_against.replace(" on the "," against the ")
    elif (" against " in entry[1]):
        on_against = on_against.replace(" against the "," on the ")
    
    return on_against

def alternate_placement_order(entry):
    placement_order = entry[3]
    matches = re.search(r"(?P<pred>.+)\((?P<obj1>.+),(?P<obj2>.+)\)",placement_order)
    if matches is not None:
        placement_order = "%s(%s,%s)" % (matches.groupdict()["pred"],
                                         matches.groupdict()["obj2"],matches.groupdict()["obj1"])
    placement_order = placement_order.replace('('," the ").replace(','," and the ").replace("_"," ").replace(')',"")
    placement_order = re.sub(r"[0-9]+","",placement_order)

    return placement_order

def search_underspecifications(entry):
    specified_manner_predicates = {
        r"grasp\((?P<dobj>.+)\)" : ("hold(%s)","touch(%s)"),
        r"hold\((?P<dobj>.+)\)" : ("touch(%s)",),
        r"turn\((?P<dobj>.+)\)" : ("move(%s)",),
        r"roll\((?P<dobj>.+)\)" : ("move(%s)","turn(%s)","spin(%s)"),
        r"slide\((?P<dobj>.+)\)" : ("move(%s)",),
        r"spin\((?P<dobj>.+)\)" : ("move(%s)","turn(%s)"),
        r"lift\((?P<dobj>.+)\)" : ("move(%s)",),
        r"stack\((?P<dobj>.+),(?P<iobj>.+)\)" : ("move(%s)",),
        r"put\((?P<dobj>.+),.+\((?P<iobj>.+)\)\)" : ("move(%s)",),
        r"lean\((?P<dobj>.+),.+\((?P<iobj>.+)\)\)" : ("move(%s)","turn(%s)"),
        r"flip\((?P<dobj>[^()]+)\)" : ("move(%s)","turn(%s)"),
        r"flip\((?P<dobj>.+),.+\((?P<iobj>.+)\)\)" : ("move(%s)","turn(%s)")
    }
    
    #print entry[3]
    underspecifications = []

    for key in specified_manner_predicates:
        matches = re.search(key,entry[3])
        if matches is not None:
            #for group in matches.groupdict():
            #    print matches.groupdict()[group]
            for val in specified_manner_predicates[key]:
                underspecification = val % matches.groupdict()["dobj"]
                underspecification = underspecification.replace('('," the ").replace(','," ").replace("_"," ").replace(')',"")
                underspecification = re.sub(r"[0-9]+","",underspecification)
                underspecifications.append(underspecification)


    return underspecifications

def find_pairwise_candidate(entry):
    pairwise_candidate = ""
    forms_list = open(forms_file).readlines()
    
    parse = entry[2]
    parse_args = re.search(r"\A[^,]+\(the\([^()]+\)",parse)
    exp = parse_args.group(0).split('(',1)
    exp[0] = "\A[^,]+"
    exp = "(".join(exp).replace("(","\(").replace(")","\)") + "+"
    #print exp

    choices = []
    for form in forms_list:
        matches = re.search(exp,form)
        if matches is not None:
            #print form
            choices.append(form)
#            for group in matches.groupdict():
#                if matches.groupdict()[group] == parse_args.groupdict()[group]:
#                    print matches.group(0)
    pairwise_candidate = random.choice(choices)

    if pairwise_candidate.split('(')[0] == "stack":
        pairwise_candidate = pairwise_candidate.replace(','," and ")

    pairwise_candidate = pairwise_candidate.replace('('," ").replace(','," ").replace("_"," ").replace(')',"").strip()

    return pairwise_candidate

if __name__ == "__main__":
    main()
