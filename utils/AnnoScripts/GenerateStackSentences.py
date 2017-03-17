objects = ["block",
           "grape",
           "ball",
           "apple",
           "plate",
           "banana",
           "cup",
           "table",
           "disc",
           "bowl",
           "spoon",
           "knife",
           "book",
           "pencil",
           "blackboard",
           "paper sheet",
           "bottle"
           ]

for i in range(len(objects)):
    for j in range(len(objects)):
        if (i != j):
            print "stack the %s and the %s" % (objects[i],objects[j] )
