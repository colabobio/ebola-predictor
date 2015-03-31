"""
Utility functions used in the neural network.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import numpy as np

"""Formats the vector theta containing the neural net coefficients into matrix form
"""
def linear_index(mat_idx, N, L, S, K):
    l = mat_idx[0] # layer
    n = mat_idx[1] # node
    i = mat_idx[2] # input    
    if l < 1: 
        return n * N + i
    elif l < L - 1:
        return (S - 1) * N + (l - 1) * (S - 1) * S + n * S + i
    else:
        return (S - 1) * N + (L - 2) * (S - 1) * S + n * S + i

"""Formats the vector theta containing the neural net coefficients into matrix form
"""
def thetaMatrix(theta, N, L, S, K):
    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = [None] * L
    C = (S - 1) * N
    thetam[0] = theta[0 : C].reshape((S - 1, N))
    for l in range(1, L - 1):
        thetam[l] = theta[C : C + (S - 1) * S].reshape((S - 1, S))
        C = C + (S - 1) * S
    thetam[L - 1] = theta[C : C + K * S].reshape((K, S))
    return thetam

"""Converts the gradient matrix into array form
"""
def gradientArray(gmatrix, N, L, S, K):
    garray = np.zeros((S - 1) * N + (L - 2) * (S - 1) * S + K * S)
    C0 = (S - 1) * N
    # http://docs.scipy.org/doc/numpy/reference/generated/numpy.copyto.html
    np.copyto(garray[0 : C0], gmatrix[0].reshape(C0))
    C = C0
    for l in range(1, L - 1):
        Ch = (S - 1) * S
        np.copyto(garray[C : C + Ch], gmatrix[l].reshape(Ch))
        C = C + Ch
    Ck =  K * S
    np.copyto(garray[C : C + Ck], gmatrix[L - 1].reshape(Ck))

    return garray

"""Evaluates the sigmoid function
"""
def sigmoid(v):
    return 1 / (1 + np.exp(-v))

"""Performs forward propagation
"""
def forwardProp(x, thetam, L):
    a = [None] * (L + 1)
    a[0] = x
    for l in range(0, L):
        z = np.dot(thetam[l], a[l])
        res = sigmoid(z)
        a[l + 1] = np.insert(res, 0, 1) if l < L - 1 else res
    return a

"""Performs backward propagation
"""
def backwardProp(y, a, thetam, L, N):
    err = [None] * (L + 1)
    err[L] = a[L] - y
    for l in range(L - 1, 0, -1):  
        backp = np.dot(np.transpose(thetam[l]), err[l + 1])
        deriv = np.multiply(a[l], 1 - a[l])
        err[l] = np.delete(np.multiply(backp, deriv), 0)
    err[0] = np.zeros(N);
    return err

"""Computes a prediction (in the form of probabilities) for the given data vector
"""
def predict(x, theta, N, L, S, K):
    thetam = thetaMatrix(theta, N, L, S, K)
    a = forwardProp(x, thetam, L) 
    h = a[L]
    return h

"""Return a function that gives a prediction from a design matrix row
"""
def gen_predictor(params_filename="./data/nnet-params"):
    with open(params_filename, "rb") as pfile:
        i = 0
        for line in pfile.readlines():
            [name, value] = line.strip().split(":")
            if i == 0:
                N = int(value.strip()) + 1
            elif i == 1:
                L = int(value.strip()) + 1
            elif i == 2:
                S = int(value.strip()) + 1
            elif i == 3:
                K = int(value.strip())
                R = (S - 1) * N + (L - 2) * (S - 1) * S + K * S
                theta = np.ones(R)
            else:
                idx = [int(s.strip().split(" ")[1]) for s in name.split(",")]
                n = linear_index(idx, N, L, S, K)
                theta[n] = float(value.strip())
            i = i + 1

    def predictor(X):
        scores = []
        for i in range(0, len(X)):
            scores.extend(predict(X[i,:], theta, N, L, S, K))
        return scores
    return predictor
