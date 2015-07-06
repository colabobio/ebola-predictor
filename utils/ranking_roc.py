"""
Creates ROC plots for all predictors over a set accuracy level

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--index_mode", nargs=1, default=["PRED"],
                    help="Indexing mode, either PRED or IMP")
parser.add_argument("-B", "--base_dir", nargs=1, default=["./"],
                    help="Base folder")
parser.add_argument("-rank", "--ranking_file", nargs=1, default=["./out/ranking.txt"],
                    help="Ranking file")
parser.add_argument("-pdf", "--pdf_file", nargs=1, default=["./out/roc-over90.pdf"],
                    help="Pdf file")
parser.add_argument("-extra", "--extra_tests", nargs=1, default=[""],
                    help="Extra tests to include a prediction in the plot, comma separated")
parser.add_argument("-x", "--exclude", nargs=1, default=["lreg,scikit_randf"],
                    help="Predictors to exclude from plots")
parser.add_argument("-op", "--opacity", type=int, nargs=1, default=[160],
                    help="Opacity of data points")
parser.add_argument("-col", "--columns", type=int, nargs=1, default=[2],
                    help="Number of label columns, 0 for no labels")

args = parser.parse_args()
index_mode = args.index_mode[0]
base_dir = args.base_dir[0]
rank_file = args.ranking_file[0]
pdf_file = args.pdf_file[0]
extra_tests = args.extra_tests[0].split(",")
excluded_predictors = args.exclude[0].split(",")
opacity = args.opacity[0]
label_columns = args.columns[0]

var_labels = {}
with open("./data/alias.txt", "r") as afile:
    lines = afile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ", 1)
        var_labels[parts[0]] = parts[1]

if index_mode == "PRED":
    options_file = "./data/predictors.txt"
else:
    options_file = "./data/imputation.txt"

index_names = []
index_labels = {}
index_acron = {}
glyph_colors = {}
with open(options_file, "r") as ofile:
    lines = ofile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(',')
        index_names.append(parts[0])
        index_labels[parts[0]] = parts[1]
        index_acron[parts[0]] = parts[2]
        glyph_colors[parts[0]] = [int(x) for x in parts[3].split()]

plt.clf()
fig = plt.figure()

print "Saving scatter plot to " + pdf_file + "..."
plots = []
with open(rank_file, "r") as rfile:
    lines = rfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")

        mdl_str = parts[1]
        pred = parts[2]
        if pred in excluded_predictors:
            continue
        if index_mode == "PRED":
            idx = pred
        else:
            idx = ""
            for idx in index_names:
                if idx in mdl_str:
                    break
        if not idx in index_names:
            continue
        if extra_tests:
            missing = False
            path_pieces =  parts[1].split(os.path.sep)
            for ex in extra_tests:
                if not ex in path_pieces:
                    missing = True
            if missing: 
                 continue

        vars = parts[3]
        f1_mean = float(parts[4])
        f1_std = float(parts[5])
        vlist = vars.split(",")

        if 0.9 <= f1_mean and 0.05 <= f1_std:
            id = os.path.split(mdl_str)[1]
            os.system("python eval.py -B " + base_dir + " -N " + id + " -p " + pred + " -m roc > ./out/roc.tmp")
            df = pd.read_csv("./out/roc.csv", delimiter=",")
            y = df["Y"]
            p = df["P"]
            c = [e/255.0 for e in glyph_colors[pred]]
            c.append(opacity/255.0)
            fpr, tpr, _ = roc_curve(y, p)
            roc = plt.plot(fpr, tpr, color=c, linewidth=1.0,)
            plots.append(roc)

plt.plot([0, 1], [0, 1], 'k--', c='grey', linewidth=0.8)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc='lower right', ncol=label_columns)

# plt.show()
fig.savefig(pdf_file)
print "Done."
