import os, sys, random, glob, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Directory to look for models")
parser.add_argument('-o', '--output_pdf', nargs=1, default=["./out/aggregated-roc.pdf"],
                    help="pdf file to save ROC curve to")

args = parser.parse_args()
base_dir = args.base_dir[0]
pdf_file = args.output_pdf[0]

pred_names = ["lreg", "nnet", "scikit_dtree", "scikit_svm"]
pred_labels = {"lreg": "Logistic Regression", "nnet": "Neural Network", "scikit_dtree": "Decision Tree", "scikit_svm": "Support Vector Machine"}

curve_colors = {"lreg":[171,217,233],
                "nnet":[44,123,182],
                "scikit_lreg":[178,223,138],
                "scikit_dtree":[253,174,97],
                "scikit_randf":[251,154,153],
                "scikit_svm":[215,25,28]}
opacity = 255

plt.clf()
fig = plt.figure()

roc_ydata = {}
roc_pdata = {}
for pred_name in pred_names: 
    roc_ydata[pred_name] = []
    roc_pdata[pred_name] = []

count = 0
for dir_name, subdir_list, file_list in os.walk(os.path.join(base_dir, "models")):
    if file_list:
        train_files = glob.glob(dir_name + "/training-data-completed-*.csv")
        var_file = os.path.exists(dir_name + "/variables.txt")
        if train_files or var_file:
            count += 1
            _, mdl_id = os.path.split(dir_name)
            print "Adding ROC data from", dir_name
            for pred_name in pred_names:
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
            if count == 4: break

print "Generating ROC plot..."
plots = []
for pred_name in pred_names:
    ydata = roc_ydata[pred_name]
    pdata = roc_pdata[pred_name]
    fpr, tpr, _ = roc_curve(ydata, pdata)
    c = [e/255.0 for e in curve_colors[pred_name]]
    c.append(opacity/255.0)
    plots.append(plt.plot(fpr, tpr, color=c, linewidth=1.0, label=pred_labels[pred_name]))

plt.plot([0, 1], [0, 1], 'k--', c='grey', linewidth=0.8)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend(loc='lower right', ncol=2)

# plt.show()
fig.savefig(pdf_file)
print "Done."
