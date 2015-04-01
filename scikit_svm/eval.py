"""
Run variety of evaluation metrics on support vector machine classifier.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, sys, os
from utils import gen_predictor
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix, run_eval, get_misses

def prefix():
    return "scikit_svm"

def title():
    return "Support Vector Machine from scikit-learn"

def pred(test_filename, train_filename, param_filename):
    X, y = design_matrix(test_filename, train_filename)
    predictor = gen_predictor(param_filename)
    probs = predictor(X)
    return probs, y

def eval(test_filename, train_filename, param_filename, method, **kwparams):
    X, y = design_matrix(test_filename, train_filename)
    predictor = gen_predictor(param_filename)
    probs = predictor(X)
    return run_eval(probs, y, method, **kwparams)

def miss(test_filename, train_filename, param_filename):
    fn = test_filename.replace("-data", "-index")
    meta = None
    if os.path.exists(fn):
        with open(fn, "r") as idxfile:
            meta = idxfile.readlines()

    X, y, df = design_matrix(test_filename, train_filename, get_df=True)
    predictor = gen_predictor(param_filename)
    probs = predictor(X)
    indices = get_misses(probs, y)
    for i in indices:
        print "----------------"
        if meta: print "META:",",".join(meta[i].split(",")).strip()
        print df.ix[i]
    return indices

def evaluate(test_filename, train_filename, param_filename, method):
    # Average calibrations and discriminations
    if method == "caldis":
        eval(test_filename, train_filename, param_filename, 1)
    # Plot each method on same calibration plot
    elif method == "calplot":
        eval(test_filename, train_filename, param_filename, 2, test_file=test_filename)
    # Average precision, recall, and F1 scores
    elif method == "report":
        eval(test_filename, train_filename, param_filename, 3)
    # Plot each method on same ROC plot
    elif method == "roc":
        eval(test_filename, train_filename, param_filename, 4, pltshow=True)
    # Average confusion matrix
    elif method == "confusion":
        eval(test_filename, train_filename, param_filename, 5)
    # Method not defined:
    elif method == "misses":
        miss(test_filename, train_filename, param_filename)
    else:
        raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs=1, default=["./data/training-data-completed.csv"],
                        help="Filename for training set")
    parser.add_argument('-T', '--test', nargs=1, default=["./data/testing-data.csv"],
                        help="Filename for testing set")
    parser.add_argument('-p', '--param', nargs=1, default=["./data/scikit_svm-params"],
                        help="Filename for decision tree parameters")
    parser.add_argument('-m', '--method', nargs=1, default=["report"],
                        help="Evaluation method: caldis, calplot, report, roc, confusion, misses")
    args = parser.parse_args()
    evaluate(args.test[0], args.train[0], args.param[0], args.method[0])
