"""
Creates ROC plots for all predictors over 90% of accuracy

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
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
opacity = 255
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

plots = []
with open(rank_file, "r") as rfile:
    lines = rfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ")

        pred = parts[2]
        mdl_str = parts[1]        
        if index_mode == "PRED":
            idx = pred
        else:
            idx = ""
            for idx in index_names:
                if idx in mdl_str:
                    break
        if not idx in index_names or pred in excluded_predictors: continue

        f1_mean = float(parts[4])
        f1_std = float(parts[5])

        if 0.9 <= f1_mean: 
            id = os.path.split(mdl_str)[1]
            print id
            os.system("python eval.py -N " + id + " -p " + pred + " -m roc > ./out/roc.tmp")
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

plt.legend(loc='lower right', ncol=2)

# plt.show()
fig.savefig(pdf_file)


# plt.legend(tuple(plots),
#            tuple(labels),
#            loc='best',
#            ncol=label_columns,
#            prop={'size':9})
# 
# plt.xlabel('F1-score mean')
# plt.ylabel('F1-score error')
# 
# # plt.show()
# fig.savefig(pdf_file)