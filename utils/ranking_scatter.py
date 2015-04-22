"""
Creates scatter plot from ranking file

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse, random, math
from sets import Set
import operator
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-rank", "--ranking_file", nargs=1, default=["ranking.txt"],
                    help="Ranking file")
parser.add_argument("-pdf", "--pdf_file", nargs=1, default=["ranking.pdf"],
                    help="Ranking file")
args = parser.parse_args()
rank_file = args.ranking_file[0]
pdf_file = args.pdf_file[0]

excluded_predictors = ["scikit_lreg", "scikit_randf"]


index_mode = "PRED"
# index_names = ["lreg", "nnet", "scikit_lreg", "scikit_dtree", "scikit_randf", "scikit_svm"]
# index_names = ["lreg"]
index_names = ["lreg", "nnet", "scikit_dtree", "scikit_svm"]
index_labels = {"lreg": "Logistic Regression", "nnet": "Neural Network", "scikit_dtree": "Decision Tree", "scikit_svm": "Support Vector Machine"}
glyph_colors = {"lreg":[171,217,233],
                "nnet":[44,123,182],
                "scikit_lreg":[178,223,138],
                "scikit_dtree":[253,174,97],
                "scikit_randf":[251,154,153],
                "scikit_svm":[215,25,28]}
opacity = 160
label_columns = 2
fixed_size = False

'''
# Algorithm
index_mode = "MODEL"
index_names = ["amelia", "mice", "hmisc"]
index_labels = {"amelia": "Amelia II", "mice": "MICE", "hmisc": "Hmisc"}
glyph_colors = {"amelia":[252,141,98],
                "mice":[102,194,165],
                "hmisc":[141,160,203]}
opacity = 160
label_columns = 3
fixed_size = True
glyph_size = 30
'''

'''
# Samples
index_mode = "MODEL"
index_names = ["df1", "df5", "df10"]
index_labels = {"df1": "1 imputation", "df5": "5 imputations", "df10": "10 imputations"}
glyph_colors = {"df1":[253,192,134],
                "df5":[190,174,212],
                "df10":[127,201,127]}
opacity = 160
label_columns = 3

fixed_size = True
glyph_size = 30
'''

'''
# Percentage
index_mode = "MODEL"
index_names = ["t80", "t65", "t50"]
index_labels = {"t80": "20% complete", "t65": "35% complete", "t50": "50% complete"}
glyph_colors = {"t80":[166,206,227],
                "t65":[31,120,180],
                "t50":[178,223,138]}
opacity = 160
label_columns = 3

fixed_size = True
glyph_size = 30
'''

plt.clf()
fig = plt.figure()

plt.xlim([0.5,1.05])
plt.ylim([-0.05,0.5])
plt.axes().set_aspect('equal')

counts = {}
xdict = {}
ydict = {}
sdict = {}
vars90 = Set([])
vars99 = Set([])
inter90 = None
vcounts = {}
with open(rank_file, "r") as rfile:
    lines = rfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")

        pred = parts[2]
        if index_mode == "PRED":
            idx = pred
        else:
            idx = ""
            mdl_str = parts[1]
            for idx in index_names:
                if idx in mdl_str:
                    break
        if not idx in index_names or pred in excluded_predictors: continue

        if idx in xdict:
            x = xdict[idx]
            y = ydict[idx]
            s = sdict[idx]
        else:
            x = []
            y = []
            s = []
            xdict[idx] = x
            ydict[idx] = y
            sdict[idx] = s
            counts[idx] = 0

        vars = parts[3]
        f1_mean = float(parts[4])
        f1_std = float(parts[5])
        if 0.9 < f1_mean: counts[idx] = counts[idx] + 1

        r1 = random.random()
        r2 = random.random()

        x.append(f1_mean)
        y.append(f1_std)

        vlist = vars.split(",")

        if fixed_size:
           s.append(glyph_size)
        else:
           # Size is calculated as a function of the number of variables in the predictor
           l = len(vlist)
           z = 30 * l /10.0
           s.append(z)

        if 0.9 <= f1_mean: 
            for v in vlist: 
                vars90.add(v)
                if not v in vcounts: vcounts[v] = 0
                vcounts[v] = vcounts[v] + 1
            if not inter90: inter90 = Set(vlist)
            else: 
                print Set(vlist)
                inter90 = inter90.intersection(Set(vlist))
        if 0.99 <= f1_mean: 
            for v in vlist: vars99.add(v)

plots = []
labels = []
for k in index_names:
    if not k in xdict: continue
    print k, counts[k]
    x = xdict[k]
    y = ydict[k]
    s = sdict[k]
    c = [e/255.0 for e in glyph_colors[k]]
    c.append(opacity/255.0)
    plots.append(plt.scatter(x, y, s=s, color=c, marker='o'))
    labels.append(index_labels[k])

print "Variables in predictors with more than 99% accuracy:", vars99
print "Variables in predictors with more than 90% accuracy:", vars90
print "Variables in all predictors with more than 90% accuracy:", inter90
print "Counts for variables in predictors with more than 90% accuracy:"
sorted_counts = reversed(sorted(vcounts.items(), key=operator.itemgetter(1)))
for v in sorted_counts: print v

plt.legend(tuple(plots),
           tuple(labels),
           loc='best',
           ncol=label_columns,
           prop={'size':9})

plt.xlabel('F1-score mean')
plt.ylabel('F1-score error')

# plt.show()
fig.savefig(pdf_file)