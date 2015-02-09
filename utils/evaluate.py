"""
Run variety of evaluation metrics on predictive model.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import pandas as pd
import numpy as np
from calibrationdiscrimination import caldis
from calplot import calplot
from classificationreport import report
from confusion import confusion
from roc import roc

def design_matrix(test_filename="", train_filename="", get_df=False):
    if test_filename and train_filename:
        # Will build a design matrix from the test data, using the training data for 
        # normalization
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
    else:
        # Will build the design matrix from either the training of testing set
        if train_filename: filename = train_filename
        else: filename = test_filename
        df = pd.read_csv(filename, delimiter=",", na_values="?")
        M = df.shape[0]
        N = df.shape[1]
        y = df.values[:,0]
        X = np.ones((M, N))
        for j in range(1, N):
            # Computing i-th column. The pandas dataframe
            # contains all the values as numpy arrays that
            # can be handled individually:
            values = df.values[:, j]
            minv = values.min()
            maxv = values.max()
            if maxv > minv:
                X[:, j] = np.clip((values - minv) / (maxv - minv), 0, 1)
            else:
                X[:, j] = 1.0 / M

    if get_df:
        return X, y, df
    else:
        return X, y

def run_eval(probs, y_test, method=1, **kwparams):
    if method == 1:
        return caldis(probs, y_test)
    elif method == 2:
        return calplot(probs, y_test, **kwparams)
    elif method == 3:
        return report(probs, y_test)
    elif method == 4:
        return roc(probs, y_test, **kwparams)
    elif method == 5:
        return confusion(probs, y_test)
    else:
        raise Exception("Invalid method argument given")

def get_misses(probs, y_test):
    miss = []
    for i in range(len(probs)):
        p = probs[i]
        pred = 0.5 < p
        if pred != y_test[i]:
            miss.append(i)
    return miss
