"""
Utility functions used in the neural network.

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
def gen_predictor(params_filename="./data/lreg0-params"):
    with open(params_filename, "rb") as pfile:
        lines = pfile.readlines()
        N = len(lines)
        theta = np.ones(N)
        i = 0
        for line in lines:
            theta[i] = float(line.strip().split(' ')[1])
            i = i + 1
#         print theta
#             
#         i = 0
#         for line in pfile.readlines():
#             [name, value] = line.strip().split(":")
#             if i == 0:
#                 N = int(value.strip()) + 1
#             elif i == 1:
#                 L = int(value.strip()) + 1
#             elif i == 2:
#                 S = int(value.strip()) + 1
#             elif i == 3:
#                 K = int(value.strip())
#                 R = (S - 1) * N + (L - 2) * (S - 1) * S + K * S
#                 theta = np.ones(R)
#             else:
#                 idx = [int(s.strip().split(" ")[1]) for s in name.split(",")]
#                 n = linear_index(idx, N, L, S, K)
#                 theta[n] = float(value.strip())
#             i = i + 1

    def predictor(X):
        scores = []
#         print X
        for i in range(0, len(X)):
            scores.extend(predict(X[i,:], theta))
        return scores
    return predictor
