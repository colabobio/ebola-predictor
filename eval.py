import argparse, glob, os
import numpy as np
import pandas as pd

from utils.calplot import calplot
from utils.evaluate import eval
from utils.roc import roc

from nnet.utils import gen_predictor as nnet_gen_predictor

predictors = ["nnet"]

def design_matrix(test_filename, train_filename):
    df0 = pd.read_csv(train_filename, delimiter=",", na_values="?")
    df = pd.read_csv(test_filename, delimiter=",", na_values="?")
    # df and df0 must have the same number of columns (N), but not necessarily the same
    # number of rows.
    M = df.shape[0]
    N = df.shape[1]
    y = df.values[:,0]
    X = np.ones((M, N))
    for j in range(1, N):
        # Computing i-th column. The pandas dataframe
        # contains all the values as numpy arrays that
        # can be handled individually:
        values = df.values[:, j]
        # Using the max/min values from the training set because those were used to 
        # train the predictor
        values0 = df0.values[:, j]
        minv0 = values0.min()
        maxv0 = values0.max()
        if maxv0 > minv0:
            X[:, j] = np.clip((values - minv0) / (maxv0 - minv0), 0, 1)
        else:
            X[:, j] = 1.0 / M
    return X, y

def avg_cal_dis():
    print "1"

def cal_plots():
    print "2"

def avg_report():
    test_files = glob.glob("./data/testing-data-*.csv")
    for pred in predictors:
        print "Calculating average report for " + pred + "..."
        count = 0
        for testfile in test_files:
            start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
            stop_idx = testfile.find('.csv')
            id = testfile[start_idx:stop_idx]
            pfile = "./data/" + pred + "-params-" + str(id)
            trainfile = "./data/training-data-completed-" + str(id) + ".csv"
            if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
                count = count + 1
                print id, testfile, pfile, trainfile
                X, y = design_matrix(testfile, trainfile)
                predictor = nnet_gen_predictor(pfile)
                probs = predictor(X)
                p, r, f, _ = eval(probs, y, 3)
#                 print X
#                 print y
    print "report"

def roc_plots():
    print "4"

def avg_conf_mat():
    print "5"

def evaluate(methods):
    for method in methods:
        # Average calibrations and discriminations
        if method == "cd":
            avg_cal_dis()
        # Plot each method on same calibration plot
        elif method == "calibration":
            cal_plots()
        # Average precision, recall, and F1 scores
        elif method == "report":
            avg_report()
        # Plot each method on same ROC plot
        elif method == "roc":
            roc_plots()
        # Average confusion matrix
        elif method == "confusion":
            avg_conf_mat()
        # Method not defined:
        else:
            raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-e', nargs='+', default=["confusion"], help="Supported evaluation methods: roc, cd, report, calibration, confusion")
    args = parser.parse_args()
    evaluate(args.e)