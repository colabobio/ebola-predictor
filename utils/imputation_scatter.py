"""
Creates the F1 mean/error scatter plots for all combinations of imputation parameters
(percentage of complete records used for training, number of imputed frames)

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

predictors = ["lreg", "nnet", "scikit_dtree", "scikit_svm"]

alg_names = ["amelia", "mice", "hmisc"]
alg_labels = {"amelia": "Amelia II", "mice": "MICE", "hmisc": "Hmisc"}
alg_colors = {"amelia":[252,141,98],
              "mice":[102,194,165],
              "hmisc":[141,160,203]}
opacity = 200

df_names = ["df1", "df5", "df10"]
df_labels = {"df1": "1 imputation", "df5": "5 imputations", "df10": "10 imputations"}

tp_names = ["t80", "t65", "t50"]
tp_labels = {"t80": "20% complete", "t65": "35% complete", "t50": "50% complete"}

plt.clf()
fig = plt.figure()

with open(rank_file, "r") as rfile:
    ranking_data = rfile.readlines()

for df in df_names:
    for tp in tp_names:
        print df, tp
        xdict = {}
        ydict = {}
        for alg in alg_names:
            xdict[alg] = []
            ydict[alg] = []
        for line in ranking_data:
            line = line.strip()
            parts = line.split(" ")
            pred = parts[2]
            mdl_str = parts[1]
            if not ((df + "/") in mdl_str and (tp  + "/") in mdl_str and pred in predictors): continue
            alg = ""
            for name in alg_names:
                if name in mdl_str:
                    alg = name
                    break
            if not alg: continue
            
            
            f1_mean = float(parts[4])
            f1_std = float(parts[5])
            print "  ",pred,alg,f1_mean,f1_std
            x = xdict[alg]
            y = ydict[alg]
            x.append(f1_mean)
            y.append(f1_std)

        plt.xlim([0.5,1.05])
        plt.ylim([-0.05,0.5])
        plt.axes().set_aspect('equal')

        plots = []
        labels = []
        for alg in alg_names:
            x = xdict[alg]
            y = ydict[alg]
            c = [e/255.0 for e in alg_colors[alg]]
            c.append(opacity/255.0)
            plots.append(plt.scatter(x, y, s=30, color=c, marker='o'))
            labels.append(alg_labels[alg])

        plt.legend(tuple(plots),
                   tuple(labels),
                   loc='best',
                   ncol=3,
                   prop={'size':9})

        plt.xlabel('F1-score mean')
        plt.ylabel('F1-score error')

        fig.savefig("./out/ranking-pcr-" + df + "-" + tp + ".pdf")
        plt.clf()
