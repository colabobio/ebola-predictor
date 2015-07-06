"""
Builds a text report showing precision, recall, F1 score.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

label_file = "./data/outcome.txt"

def report(probs, y_test):
    preds = [int(0.5 < p) for p in probs]

    target_names = []
    with open(label_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            target_names.append(line.split()[1])

    report = classification_report(y_test, preds, target_names=target_names)

    print report

    return precision_recall_fscore_support(y_test, preds)