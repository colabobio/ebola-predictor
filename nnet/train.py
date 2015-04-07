"""
Trains a Neural Network predictor with binary output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys
import math
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
from utils import thetaMatrix, gradientArray, sigmoid, forwardProp, backwardProp, predict

def prefix():
    return "nnet"

def title():
    return "Neural Network"

"""Evaluates the cost function
"""
def cost(theta, X, y, N, L, S, K, gamma):
    M = X.shape[0]

    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = thetaMatrix(theta, N, L, S, K)

    h = np.zeros(M)
    terms = np.zeros(M)
    for i in range(0, M):
        a = forwardProp(X[i,:], thetam, L)
        h[i] = a[L]
        t0 = -y[i] * np.log(h[i]) if 0 < y[i] else 0
        t1 = -(1-y[i]) * np.log(1-h[i]) if y[i] < 1 else 0
        if math.isnan(t0) or math.isnan(t1):
            #print "NaN detected when calculating cost contribution of observation",i
            terms[i] = 10
        else:
            terms[i] = t0 + t1 

    # Regularization penalty
    penalty = (gamma/2) * np.sum(theta * theta)

    return terms.mean() + penalty;

"""Computes the gradient of the cost function
"""
def gradient(theta, X, y, N, L, S, K, gamma):
    M = X.shape[0]

    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = thetaMatrix(theta, N, L, S, K)

    # Init auxiliary data structures
    delta = [None] * L
    D = [None] * L
    delta[0] = np.zeros((S - 1, N))
    D[0] = np.zeros((S - 1, N))
    for l in range(1, L - 1):
        delta[l] = np.zeros((S - 1, S))
        D[l] = np.zeros((S - 1, S))
    delta[L - 1] = np.zeros((K, S))
    D[L - 1] = np.zeros((K, S))

    for i in range(0, M):
        a = forwardProp(X[i,:], thetam, L)
        err = backwardProp(y[i], a, thetam, L, N)
        for l in range(0, L):
            # Notes about multiplying numpy arrays: err[l+1] is a 1-dimensional
            # array so it needs to be made into a 2D array by putting inside [],
            # and then transposed so it becomes a column vector.
            prod = np.array([err[l+1]]).T * np.array([a[l]])
            delta[l] = delta[l] + prod

    for l in range(0, L):
        D[l] = (1.0 / M) * delta[l] + gamma * thetam[l]
    
    grad = gradientArray(D, N, L, S, K)

    global gcheck
    if gcheck:
        ok = True
        size = theta.shape[0] 
        epsilon = 1E-5
        maxerr = 0.01
        grad1 = np.zeros(size);
        for i in range(0, size):
            theta0 = np.copy(theta)
            theta1 = np.copy(theta)

            theta0[i] = theta0[i] - epsilon
            theta1[i] = theta1[i] + epsilon

            c0 = cost(theta0, X, y, N, L, S, K, gamma)
            c1 = cost(theta1, X, y, N, L, S, K, gamma)
            grad1[i] = (c1 - c0) / (2 * epsilon)
            diff = abs(grad1[i] - grad[i])
            if maxerr < diff:
                print "Numerical and analytical gradients differ by",diff,"at argument",i,"/",size
                ok = False
        if ok:
            print "Numerical and analytical gradients coincide within the given precision of",maxerr

    return grad

"""Adds the current cost to the vector of values
"""
def add_value(theta):
    global params
    global values
    (X, y, N, L, S, K, gamma) = params
    value = cost(theta, X, y, N, L, S, K, gamma);
    values = np.append(values, [value])

"""
Calculating the prediction rate by applying the trained model on the remaining fraction 
of the data (the test set), and comparing with random selection
"""
def evaluate(X, y, itrain, theta):
    ntot = 0
    nhit = 0
    nnul = 0
    for i in range(0, M):
        if not i in itrain:
            ntot = ntot + 1
            p = predict(X[i,:], theta, N, L, S, K)
            if (y[i] == 1) == (0.5 < p):
                nhit = nhit + 1
            if y[i] == 1:
                nnul = nnul + 1

    rate = float(nhit) / float(ntot)
    rrate = float(nnul) / float(ntot)
    
    print ""
    print "---------------------------------------"
    print "Predictor success rate on test set:", str(nhit) + "/" + str(ntot), round(100 * rate, 2), "%"
    print "Null success rate on test set     :", str(nnul) + "/" + str(ntot), round(100 * rrate, 2), "%"
    
    return [rate, rrate]

"""Prints the neural net parameters
"""
def print_theta(theta, N, L, S, K):
    thetam = thetaMatrix(theta, N, L, S, K)

    # Coefficients of first layer
    theta0 = thetam[0]
    for i1 in range(0, S - 1):
        for i0 in range(0, N):
            print "{:5s} {:1.0f}, {:4s} {:1.0f}, {:5s} {:1.0f}: {:3.5f}".format("layer", 0, "node", i1, "input", i0, theta0[i1][i0])
            
    for l in range(1, L - 1):
        thetal = thetam[l]
        # Coefficients of l-th layer 
        for i1 in range(0, S - 1):
            for i0 in range(0, S):
                print "{:5s} {:1.0f}, {:4s} {:1.0f}, {:5s} {:1.0f}: {:3.5f}".format("layer", l, "node", i1, "input", i0, thetal[i1][i0])

    # Coefficients of output unit
    thetaf = thetam[L - 1]
    for i1 in range(0, K):
        for i0 in range(0, S):
            print "{:5s} {:1.0f}, {:4s} {:1.0f}, {:5s} {:1.0f}: {:3.5f}".format("layer", L - 1, "node", i1, "input", i0, thetaf[i1][i0])

"""Saves the neural net parameters to the specified file
"""
def save_theta(filename, theta, N, L, S, K):
    with open(filename, "wb") as pfile:
        thetam = thetaMatrix(theta, N, L, S, K)
        pfile.write("Number of independent variables : " + str(N-1) + "\n")
        pfile.write("Number of hidden layers         : " + str(L-1) + "\n")
        pfile.write("Number of units per hidden layer: " + str(S-1) + "\n")
        pfile.write("Number of output classes        : " + str(K) + "\n")

        # Coefficients of first layer
        theta0 = thetam[0]
        for i1 in range(0, S - 1):
            for i0 in range(0, N):
                pfile.write("layer 0, node " + str(i1) + ", input " + str(i0) + ": " + str(theta0[i1][i0]) + "\n")
            
        for l in range(1, L - 1):
            thetal = thetam[l]
            # Coefficients of l-th layer 
            for i1 in range(0, S - 1):
                for i0 in range(0, S):
                    pfile.write("layer " + str(l) + ", node " + str(i1) + ", input " + str(i0) + ": " + str(thetal[i1][i0]) + "\n")

        # Coefficients of output unit
        thetaf = thetam[L - 1]
        for i1 in range(0, K):    
            for i0 in range(0, S):
                pfile.write("layer " + str(L - 1) + ", node " + str(i1) + ", input " + str(i0) + ": " + str(thetaf[i1][i0]) + "\n")

"""
Trains the neural net given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting neural network parameters
: param kwparams: custom arguments for neural network: L (number of hidden layers), hf 
                  (factor to calculate number of hidden units given the number of variables),
                  inv_reg (inverse of regularization coefficient), threshold 
                  (default convergence threshold), show (show minimization plot), debug 
                  (gradient check)
"""
def train(train_filename, param_filename, **kwparams):
    if "layers" in kwparams:
        L = int(kwparams["layers"])
    else:
        L = 1

    if "hfactor" in kwparams:
        hf = float(kwparams["hfactor"])
    else:
        hf = 1

    if "inv_reg" in kwparams:
        gamma = 1.0 / float(kwparams["inv_reg"])
    else:
        gamma = 0.005

    if "threshold" in kwparams:
        threshold = float(kwparams["threshold"])
    else:
        threshold = 1E-5

    if "show" in kwparams:
        show = True if kwparams["show"].lower() == "true" else False
    else:
        show = False

    if "debug" in kwparams:
        debug = True if kwparams["debug"].lower() == "true" else False
    else:
        debug = False

    global gcheck
    global params
    global values
    gcheck = debug
    K = 1

    if L < 1:
        print "Need to have at least one hidden layer"
        sys.exit(1)

    L = L + 1

    # Loading data frame and initalizing dimensions
    df = pd.read_csv(train_filename, delimiter=",", na_values="?")
    M = df.shape[0]
    N = df.shape[1]
    S = int(N * hf) # includes the bias unit on each layer, so the number of units is S-1
    print "Number of data samples          :", M
    print "Number of independent variables :", N-1
    print "Number of hidden layers         :", L-1
    print "Number of units per hidden layer:", S-1
    print "Number of output classes        :", K

    # Number of parameters:
    # * (S - 1) x N for the first weight natrix (N input nodes counting the bias term), into S-1 nodes in
    #   the first hidden layer
    # * (S - 1) x S for all the weight matrices in the hidden layers, which go from S (counting bias term)
    #   into S-1 nodes in the next layer. Since L counts the number of hidden layers plus the output layer,
    #   and the first transition was accounted by the first term, then we only need L-2
    # * K x S, for the last transition into the output layer with K nodes
    R = (S - 1) * N + (L - 2) * (S - 1) * S + K * S

    y = df.values[:,0]
    # Building the (normalized) design matrix
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

    theta0 = 1 - 2 * np.random.rand(R)
    params = (X, y, N, L, S, K, gamma)

    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    print "Training Neural Network..."
    values = np.array([])
    theta = fmin_bfgs(cost, theta0, fprime=gradient, args=params, gtol=threshold, callback=add_value)
    print "Done!"

    if show:
        plt.plot(np.arange(values.shape[0]), values)
        plt.xlabel("Step number")
        plt.ylabel("Cost function") 
        plt.show()

    print ""
    print "***************************************"
    print "Best predictor:"
    print_theta(theta, N, L, S, K)
    save_theta(param_filename, theta, N, L, S, K)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./models/test/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./models/test/nnet-params"],
                        help="Output file to save the parameters of the neural net")
    parser.add_argument("-l", "--layers", nargs=1, type=int, default=[1],
                        help="Number of hidden layers")
    parser.add_argument("-f", "--hfactor", nargs=1, type=int, default=[1],
                        help="Hidden units factor")
    parser.add_argument("-r", "--inv_reg", nargs=1, type=float, default=[200],
                        help="Inverse of regularization coefficient, larger values represent lower penalty")
    parser.add_argument("-c", "--convergence", nargs=1, type=float, default=[1E-5],
                        help="Convergence threshold for the BFGS minimizer")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Shows minimization plot")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debugs gradient calculation")
    args = parser.parse_args()
    train(args.train[0], args.param[0],
          layers=str(args.layers[0]),
          hfactor=str(args.hfactor[0]),
          inv_reg=str(args.inv_reg[0]),
          threshold=str(args.convergence[0]),
          show=str(args.show),
          debug=str(args.debug))
