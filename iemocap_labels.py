"""Creates a dict of lists for each emotion, containing the respective uttIDs"""

import re
import os


emodict = {"ang": [], "dis": [], "exc": [], "fea": [], "fru": [],
           "hap": [], "neu": [], "oth": [], "sad": [], "sur": [], "xxx": []}

files = os.listdir('EmoEvaluations')
# open each EmoEvaluation file and assign the uttID to one of the lists
for file in files:
    with open(os.path.join('EmoEvaluations', file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split("\t")
            if line[0][0] == "[":
                label = line[2]
                emodict[label].append(line[1])# column containing label

# output into one file for each emotion label
for key in emodict:
    with open(os.path.join('IEMOCAP_labels', key), 'w', encoding='utf-8') as out:
        for value in emodict[key]:
            out.writelines(value + "\n")




