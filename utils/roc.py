"""
Computes and plots the ROC Curve from a predictive model.
ROC code from http://scikit-learn.org/0.11/auto_examples/plot_roc.html.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, random
import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc(probs, y_test, **kwparams):
    # Compute ROC curve and area the curve
#     r = [random.random() for x in probs]
    roc_fpr, roc_tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(roc_fpr, roc_tpr)
    print "Area under the ROC curve : %f" % roc_auc
#     print y_test, probs
#     print roc_fpr, roc_tpr, thresholds

    if "color" in kwparams:
        color = kwparams["color"]
    else:
        color = "grey"
        
    if "label" in kwparams:
        label = kwparams["label"]
    else:
        label="ROC curve"

    pltshow = kwparams["pltshow"]

    # Plot ROC curve
#     plt.plot(fpr, tpr, label=label, marker='o', c=color)
    if pltshow:
        fig = plt.figure()
        plt.plot(roc_fpr, roc_tpr, c=color)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
        if not os.path.exists("./out"): os.makedirs("./out")
        fig.savefig('./out/roc.pdf')
        print "Saved ROC curve to ./out/roc.pdf"

    return roc_fpr, roc_tpr, roc_auc