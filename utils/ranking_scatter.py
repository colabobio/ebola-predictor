"""
Creates scatter plot from reports

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse, random, math
from sets import Set
import operator
import matplotlib.pyplot as plt
import seaborn as sns

# parser = argparse.ArgumentParser()
# parser.add_argument('pred', nargs=1, default=["nnet"],
#                     help="Predictor to generate plot for")
# args = parser.parse_args()
# predictor = args.pred[0]
# mode = args.scatter_mode[0]

# predictors = ["lreg", "nnet", "scikit_lreg", "scikit_dtree", "scikit_randf", "scikit_svm"]
predictors = ["nnet"]
colors = {"lreg":[166,206,227], 
          "nnet":[31,120,180],
          "scikit_lreg":[178,223,138],
          "scikit_dtree":[51,160,44],
          "scikit_randf":[251,154,153],
          "scikit_svm":[227,26,28]}
opacity = 160

plt.clf()
fig = plt.figure()

plt.xlim([0.5,1.1])
plt.ylim([0.5,1.1])
plt.axes().set_aspect('equal')


counts = {}
xdict = {}
ydict = {}
sdict = {}
vars90 = Set([])
vars99 = Set([])
inter90 = None
vcounts = {}
with open("out/ranking.txt", "r") as rfile:
    lines = rfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")
        pred = parts[2]
        if not pred in predictors: continue
        
        if pred in xdict:
            x = xdict[pred]
            y = ydict[pred]
            s = sdict[pred]
        else:
            x = []
            y = []
            s = []
            xdict[pred] = x
            ydict[pred] = y
            sdict[pred] = s
            counts[pred] = 0
                    
        vars = parts[3]
        f1_mean = float(parts[4])
        f1_std = float(parts[5])
        if 0.9 < f1_mean: counts[pred] = counts[pred] + 1
        
        r1 = random.random()
        r2 = random.random()        
        if f1_mean == 1:
            f1_mean += 0.01*(1-2*r1)
            f1_std += 0.01*(1-2*r2)
                    
        x.append(f1_mean)
        y.append(1 - f1_std)
        vlist = vars.split(",")
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
for k in xdict:
    print k, counts[k]
    x = xdict[k]
    y = ydict[k]    
    s = sdict[k]
    c = [e/255.0 for e in colors[k]]
    c.append(opacity/255.0) 
    plots.append(plt.scatter(x, y, s=s, color=c, marker='o'))
    labels.append(k);


print "Variables in predictors with more than 99% accuracy:", vars99
print "Variables in predictors with more than 90% accuracy:", vars90
print "Variables in all predictors with more than 90% accuracy:", inter90
print "Counts for variables in predictors with more than 90% accuracy:"
sorted_counts = reversed(sorted(vcounts.items(), key=operator.itemgetter(1)))
for v in sorted_counts: print v

plt.legend(tuple(plots),
           tuple(labels),
           loc='best',
           ncol=3,
           prop={'size':9})

plt.xlabel('Mean F1 score')
plt.ylabel('1 - F1 score error')

# plt.show()
fig.savefig("./out/ranking.pdf")