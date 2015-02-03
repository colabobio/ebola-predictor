""" Run variety of evaluation metrics on predictive model.
"""

import pandas as pd
import numpy as np
from calibrationdiscrimination import caldis
from calplot import calplot
from classificationreport import report
from confusion import confusion
from roc import roc

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

def run_eval(probs, y_test, method=1):
    if method == 1:
        return caldis(probs, y_test)
    elif method == 2:
        return calplot(probs, y_test)
    elif method == 3:
        return report(probs, y_test)
    elif method == 4:
        return roc(probs, y_test)
    elif method == 5:
        return confusion(probs, y_test)
    else:
        raise Exception("Invalid method argument given")