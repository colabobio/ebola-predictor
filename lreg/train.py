"""
Trains a Logistic Regression Classifier with binary output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

def prefix():
    return "lreg"

def title():
    return "Logistic Regression"

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def cost(theta, X, y, gamma):
    M = X.shape[0]

    h = sigmoid(np.dot(X, theta))
    terms =  -y * np.log(h) - (1-y) * np.log(1-h)

    prod = theta * theta
    prod[0] = 0
    penalty = (gamma / (2 * M)) * np.sum(prod)

    return terms.mean() + penalty

def gradient(theta, X, y, gamma):
    M = X.shape[0]
    N = X.shape[1]

    # Note the vectorized operations using numpy:
    # X is a MxN array, and theta a Nx1 array,
    # so np.dot(X, theta) gives a Mx1 array, which
    # in turn is used by the sigmoid function to 
    # perform the calculation component-wise and
    # return another Mx1 array
    h = sigmoid(np.dot(X, theta))
    err = h - y
    # err is a Mx1 array, so that its dot product
    # with the MxN array X gives a Nx1 array, which
    # in this case it is exactly the gradient!
    costGrad = np.dot(err, X) / M

    regCost = (gamma / M) * np.copy(theta)
    regCost[0] = 0

    grad = costGrad + regCost

    global gcheck
    if gcheck:
        ok = True
        epsilon = 1E-5
        maxerr = 0.01
        grad1 = np.zeros(N);
        for i in range(0, N):
            theta0 = np.copy(theta)
            theta1 = np.copy(theta)

            theta0[i] = theta0[i] - epsilon
            theta1[i] = theta1[i] + epsilon

            c0 = cost(theta0, X, y, gamma)
            c1 = cost(theta1, X, y, gamma)
            grad1[i] = (c1 - c0) / (2 * epsilon)
            diff = abs(grad1[i] - grad[i])
            if maxerr < diff: 
                print "Numerical and analytical gradients differ by",diff,"at argument",i,"/",N
                ok = False
        if ok:
            print "Numerical and analytical gradients coincide within the given precision of",maxerr

    return grad

def add_value(theta):
    global params
    global gamma 
    global values
    (X, y, gamma) = params
    value = cost(theta, X, y, gamma);
    values = np.append(values, [value])

def optim(params, threshold):
    global values
    (X, y, gamma) = params
    M = X.shape[0]
    N = X.shape[1]

    print ""
    print "Running BFGS minimization..."
    theta0 = np.random.rand(N)
 
    thetaOpt = fmin_l_bfgs_b(cost, theta0, fprime=gradient, args=(X, y, gamma), pgtol=threshold, callback=add_value)[0]
    return [True, thetaOpt]

def print_theta(theta, N, names):
    print "{:10s} {:3.5f}".format("Intercept", theta[0])
    for i in range(1, N):
        print "{:10s} {:3.5f}".format(names[i-1], theta[i])

def save_theta(filename, theta, N, names):
    with open(filename, "wb") as pfile:
        pfile.write("Intercept " + str(theta[0]) + "\n")
        for i in range(1, N):
            pfile.write(names[i-1] + " " + str(theta[i]) + "\n")

"""
Trains the logistic regression classifier given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting logistic regression parameters
: param kwparams: custom arguments for logistic regression: nv_reg (inverse of regularization 
                  coefficient), threshold (default convergence threshold), show (show 
                  minimization plot), debug (gradient check)
"""
def train(train_filename, param_filename, **kwparams):
    if "inv_reg" in kwparams:
        gamma = 1.0 / float(kwparams["inv_reg"])
    else:
        gamma = 0.08

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

    print "***************************************"

    # Loading data frame and initalizing dimensions
    df = pd.read_csv(train_filename, delimiter=',', na_values="?")
    M = df.shape[0]
    N = df.shape[1]
    vars = df.columns.values[1: N]
    print "Number of independent variables:", N-1
    print "Number of data samples         :", M

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

    values = np.array([])
    params = (X, y, gamma)
    [conv, theta] = optim(params, threshold)

    if conv:
        print "Convergence!"
    else:
        print "Error: cost function increased..."
        print "Try adjusting the learning or the regularization coefficients"

    if show:
        plt.plot(np.arange(values.shape[0]), values)
        plt.xlabel("Step number")
        plt.ylabel("Cost function")
        plt.show()

    print ""
    print "Logistic Regresion parameters:"
    print_theta(theta, N, vars)
    save_theta(param_filename, theta, N, vars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./models/test/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./models/test/lreg-params"],
                        help="Output file to save the parameters of the neural net")
    parser.add_argument("-r", "--inv_reg", nargs=1, type=float, default=[12.5],
                        help="Inverse of regularization coefficient, larger values represent lower penalty")
    parser.add_argument("-c", "--convergence", nargs=1, type=float, default=[1E-5],
                        help="Convergence threshold for the BFGS minimizer")
    parser.add_argument("-s", "--show", action="store_true",
                        help="Shows minimization plot")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Debugs gradient calculation")

    args = parser.parse_args()
    train(args.train[0], args.param[0],
          inv_reg=str(args.inv_reg[0]),
          threshold=str(args.convergence[0]),
          show=str(args.show),
          debug=str(args.debug))
