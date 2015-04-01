"""
Utility functions used in the logistic regression classifier.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import numpy as np

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

"""Computes a prediction (in the form of probabilities) for the given data vector
"""
def predict(x, theta):
    p = sigmoid(np.dot(x, theta))
    return np.array([p])

"""Return a function that gives a prediction from a design matrix row
"""
def gen_predictor(params_filename="./data/lreg-params"):
    with open(params_filename, "rb") as pfile:
        lines = pfile.readlines()
        N = len(lines)
        theta = np.ones(N)
        i = 0
        for line in lines:
            theta[i] = float(line.strip().split(' ')[1])
            i = i + 1

    def predictor(X):
        scores = []
        for i in range(0, len(X)):
            scores.extend(predict(X[i,:], theta))
        return scores
    return predictor
