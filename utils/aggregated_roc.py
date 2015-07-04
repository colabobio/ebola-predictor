"""
Creates ROC curve aggregating all models available inside the base directory

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, sys, random, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Directory to look for models")
parser.add_argument('-pdf', '--pdf_file', nargs=1, default=["./out/aggregated-roc.pdf"],
                    help="pdf file to save ROC curve to")
parser.add_argument("-x", "--exclude", nargs=1, default=["lreg,scikit_randf"],
                    help="Predictors to exclude from plots")
parser.add_argument("-op", "--opacity", type=int, nargs=1, default=[255],
                    help="Opacity of curves")
parser.add_argument("-col", "--columns", type=int, nargs=1, default=[2],
                    help="Number of label columns, 0 for no labels")
parser.add_argument("-i", "--interval", type=int, nargs=1, default=[1],
                    help="Interval between models")

args = parser.parse_args()
base_dir = args.base_dir[0]
pdf_file = args.pdf_file[0]
excluded_predictors = args.exclude[0].split(",")
opacity = args.opacity[0]
label_columns = args.columns[0]
interval = args.interval[0]

pred_names = []
pred_labels = {}
curve_colors = {}
options_file = "./data/predictors.txt"
with open(options_file, "r") as ofile:
    lines = ofile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(',')
        pred_names.append(parts[0])
        pred_labels[parts[0]] = parts[1]
        curve_colors[parts[0]] = [int(x) for x in parts[3].split()]

plt.clf()
fig = plt.figure()

roc_ydata = {}
roc_pdata = {}
for pred_name in pred_names:
    if pred_name in excluded_predictors:
        continue
    roc_ydata[pred_name] = []
    roc_pdata[pred_name] = []

count = 0
for dir_name, subdir_list, file_list in os.walk(os.path.join(base_dir, "models")):
    if file_list:
        train_files = glob.glob(dir_name + "/training-data-completed-*.csv")
        var_file = os.path.exists(dir_name + "/variables.txt")
        if train_files or var_file:
            count += 1
            if not count % interval == 0:
                continue
            _, mdl_id = os.path.split(dir_name)
            print "Adding ROC data from", dir_name
            for pred_name in pred_names:
                if pred_name in excluded_predictors:
                    continue
                ydata = roc_ydata[pred_name]
                pdata = roc_pdata[pred_name]
                roc_cmd = "python eval.py -B " + base_dir + " -N " + mdl_id + " -p " + pred_name + " -m roc > ./out/roc.tmp"
                os.system(roc_cmd)
                with open("./out/roc.csv", "r") as roc:
                    lines = roc.readlines()
                    for line in lines[1:]:
                        y, p = [float(x) for x in line.split(",")]
                        ydata.append(y)
                        pdata.append(p)

print "AUCs..."
plots = []
for pred_name in pred_names:
    if pred_name in excluded_predictors:
        continue
    ydata = roc_ydata[pred_name]
    pdata = roc_pdata[pred_name]
    fpr, tpr, _ = roc_curve(ydata, pdata)
    auc = roc_auc_score(ydata, pdata)
    c = [e/255.0 for e in curve_colors[pred_name]]
    c.append(opacity/255.0)
    plots.append(plt.plot(fpr, tpr, color=c, linewidth=1.0, label=pred_labels[pred_name]))
    print "Predictor",pred_name,"=",auc

plt.plot([0, 1], [0, 1], 'k--', c='grey', linewidth=0.8)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc='lower right', ncol=label_columns)

print "Saving ROC plot..."
# plt.show()
fig.savefig(pdf_file)
print "Done."
