"""
Run variety of evaluation metrics on eps.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, sys, os
from utils import load_dataframe, design_matrix ,gen_predictor
sys.path.append(os.path.abspath('./utils'))
from evaluate import run_eval, get_misses
from evaluate import design_matrix as design_matrix_full

def prefix():
    return "eps"

def title():
    return "Ebola Prognosis Score"

def eval(test_filename, train_filename, param_filename, method, cutoff=0, **kwparams):
    X, y = design_matrix(load_dataframe(test_filename))
    predictor = gen_predictor(cutoff)
    probs = predictor(X)
    return run_eval(probs, y, method, **kwparams)

def miss(test_filename, train_filename, param_filename, cutoff=0):
    _, _, df0 = design_matrix_full(test_filename=test_filename, train_filename="", get_df=True)

    fn = test_filename.replace("-data", "-index")
    meta = None
    if os.path.exists(fn):
        with open(fn, "r") as idxfile:
            meta = idxfile.readlines()

    df = load_dataframe(test_filename)
    X, y = design_matrix(df)
    predictor = gen_predictor(cutoff)
    probs = predictor(X)
    indices = get_misses(probs, y)
    for i in indices:
        print "----------------"
        if meta: print "META:",",".join(meta[i].split(",")).strip()
        print "SCORE:", df.ix[i][1]
        print df0.ix[i]
    return indices

def evaluate(test_filename, train_filename, param_filename, method, cutoff=0):
    # Average calibrations and discriminations
    if method == "caldis":
        eval(test_filename, train_filename, param_filename, 1, cutoff)
    # Plot each method on same calibration plot
    elif method == "calplot":
        eval(test_filename, train_filename, param_filename, 2, cutoff, test_file=test_filename)
    # Average precision, recall, and F1 scores
    elif method == "report":
        eval(test_filename, train_filename, param_filename, 3, cutoff)
    # Plot each method on same ROC plot
    elif method == "roc":
        eval(test_filename, train_filename, param_filename, 4, cutoff, pltshow=True)
    # Average confusion matrix
    elif method == "confusion":
        eval(test_filename, train_filename, param_filename, 5, cutoff)
    # Method not defined:
    elif method == "misses":
        miss(test_filename, train_filename, param_filename, cutoff)
    else:
        raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs=1, default=[""],
                        help="Filename for training set, not needed in EPS")
    parser.add_argument('-T', '--test', nargs=1, default=["./data/testing-data.csv"],
                        help="Filename for testing set")
    parser.add_argument('-c', '--cutoff', nargs=1, type=int, default=[0],
                        help="Cutoff for prediction in EPS, score less than or equal to cutoff results in survival prediction")
    parser.add_argument('-m', '--method', nargs=1, default=["report"],
                        help="Evaluation method: caldis, calplot, report, roc, confusion, misses")
    args = parser.parse_args()
    evaluate(args.test[0], args.train[0], "", args.method[0], args.cutoff[0])
