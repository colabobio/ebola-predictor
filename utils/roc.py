"""
Computes and plots the ROC Curve from a predictive model.
ROC code from http://scikit-learn.org/0.11/auto_examples/plot_roc.html.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc(probs, y_test, **kwparams):
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    print "Area under the ROC curve : %f" % roc_auc

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
    plt.plot(fpr, tpr, c=color)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")

    if pltshow:
        plt.show()

    return roc_auc