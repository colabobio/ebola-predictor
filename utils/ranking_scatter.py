"""
Creates scatter plot from ranking file generated from rank_models

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse, random, math
from sets import Set
import operator
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-mode", "--index_mode", nargs=1, default=["PRED"],
                    help="Indexing mode, either PRED or IMP")
parser.add_argument("-rank", "--ranking_file", nargs=1, default=["./out/ranking.txt"],
                    help="Ranking file")
parser.add_argument("-pdf", "--pdf_file", nargs=1, default=["./out/ranking.pdf"],
                    help="Output pdf file")
parser.add_argument("-extra", "--extra_tests", nargs=1, default=[""],
                    help="Extra tests to include a prediction in the plot, comma separated")
parser.add_argument("-pred", "--pred_file", nargs=1, default=["./out/predictors.tsv"],
                    help="Predictors file")
parser.add_argument("-count", "--count_file", nargs=1, default=["./out/varcounts.csv"],
                    help="Counts file")
parser.add_argument("-op", "--opacity", type=int, nargs=1, default=[160],
                    help="Opacity of data points")
parser.add_argument("-col", "--columns", type=int, nargs=1, default=[2],
                    help="Number of label columns, 0 for no labels")
parser.add_argument("-r", "--radius", type=float, nargs=1, default=[0],
                    help="Radius of data points, 0 for calculation based on number of variables")
parser.add_argument("-x", "--exclude", nargs=1, default=["lreg,scikit_randf"],
                    help="Predictors to exclude from plots")

args = parser.parse_args()
index_mode = args.index_mode[0]
rank_file = args.ranking_file[0]
pdf_file = args.pdf_file[0]
extra_tests = args.extra_tests[0].split(",")
pred_file = args.pred_file[0]
count_file = args.count_file[0]
excluded_predictors = args.exclude[0].split(",")

opacity = args.opacity[0]
label_columns = args.columns[0]
glyph_size = args.radius[0]
fixed_size = 0 < glyph_size

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

plt.xlim([0.5,1.05])
plt.ylim([-0.05,0.5])
plt.axes().set_aspect('equal')

top_models = []
counts = {}
xdict = {}
ydict = {}
sdict = {}
vars90 = Set([])
inter90 = None
vcounts = {}
with open(rank_file, "r") as rfile:
    lines = rfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")

        pred = parts[2]
        if pred in excluded_predictors:
            continue
        if index_mode == "PRED":
            idx = pred
        else:
            idx = ""
            mdl_str = parts[1]
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

        if 0.05 <= f1_std:
            if 0.9 <= f1_mean: counts[idx] = counts[idx] + 1
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

        if 0.9 <= f1_mean and 0.05 <= f1_std:
            top_line = index_acron[idx] + '\t' + ', '.join([var_labels[v] for v in vlist]) + '\t' + ("%.2f" % f1_mean) + '\t' + ("%.2f" % f1_std)
            top_models.append(top_line)
            for v in vlist:
                vars90.add(v)
                if not v in vcounts: vcounts[v] = 0
                vcounts[v] = vcounts[v] + 1
            if not inter90: inter90 = Set(vlist)
            else:
                inter90 = inter90.intersection(Set(vlist))

print "Number of models with mean F1-score above 90%:"
for k in index_names:
    if not k in xdict: continue
    print index_labels[k] + ": " + str(counts[k])

print ""
print "Saving scatter plot to " + pdf_file + "..."
plots = []
labels = []
for k in index_names:
    if not k in xdict: continue
    x = xdict[k]
    y = ydict[k]
    s = sdict[k]
    c = [e/255.0 for e in glyph_colors[k]]
    c.append(opacity/255.0)
    plots.append(plt.scatter(x, y, s=s, color=c, marker='o'))
    labels.append(index_labels[k])

if 0 < label_columns:
    plt.legend(tuple(plots),
               tuple(labels),
               loc='best',
               ncol=label_columns,
               prop={'size':9})

plt.xlabel('F1-score mean')
plt.ylabel('F1-score error')
# plt.show()
fig.savefig(pdf_file)
print "Done."

print ""
print "Saving list of predictors to " + pred_file + "..."
with open(pred_file, "w") as pfile:
    pfile.write("Predictor\tVariables\tF1 mean\tF1 error\n")
    for line in top_models:
        pfile.write(line + "\n")
print "Done."

print ""
print "Saving variable counts to " + count_file + "..."
sorted_counts = reversed(sorted(vcounts.items(), key=operator.itemgetter(1)))
with open(count_file, "w") as pfile:
    pfile.write("Variable,Count\n")
    for v in sorted_counts: 
        pfile.write(var_labels[v[0]] + "," + str(v[1]) + "\n")
print "Done."
