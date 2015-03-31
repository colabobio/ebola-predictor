"""
Trains a Logistic Regression Classifier with binary output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys
import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

def prefix():
    return "lreg0"

def title():
    return "Logistic Regression Zero"

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
    global gcheck

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

def debug(theta):
    global params
    global gamma 
    global values
    (X, y, gamma) = params
    value = cost(theta, X, y, gamma);
    values = np.append(values, [value])

def optim(params):
    global values
    (X, y, gamma) = params
    M = X.shape[0]
    N = X.shape[1]

    print ""
    print "Running BFGS minimization..."
    theta0 = np.random.rand(N)
 
    thetaOpt = fmin_l_bfgs_b(cost, theta0, fprime=gradient, args=(X, y, gamma), callback=debug)[0]
    return [True, thetaOpt]

def evaluate(X, y, itrain, theta):
    # Calculating the prediction rate by applying the trained model on the remaining
    # 30% of the data (the test set), and comparing with random selection
    ntot = 0
    nhit = 0
    nnul = 0
    for i in range(0, M):
        if not i in itrain:
            ntot = ntot + 1
            p = sigmoid(np.dot(X[i,:], theta))
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
: param param_filename: name of file to store resulting neural network parameters
: param kwparams: custom arguments for neural network: L (number of hidden layers), hf 
                  (factor to calculate number of hidden units given the number of variables),
                  gamma (regularization coefficient), threshold (default convergence threshold),
                  show (show minimization plot), debug (gradient check)
"""
def train(train_filename, param_filename, **kwparams):
    iter      = 1     # Number of training runs 
    trainf    = 1     # Fraction of rows from training data to use in each training iteration.
                  # The rest of rows are used to calculate the accuracy of the parameters,
                  # so that the final parameters are chosen to be those that maximize
                  # the accuracy.
    gamma     = 0.08  # Regularization coefficient
    showp     = False # Show minimization plot
    gcheck    = False # Gradient check

    global gcheck
    global params
    global values

    # Loading data frame and initalizing dimensions
    df = pd.read_csv(train_filename, delimiter=',', na_values="?")
    M = df.shape[0]
    N = df.shape[1]
    vars = df.columns.values[1: N]
    print "Number of independent variables:", N-1
    print "Number of data samples         :", M

#     theta = np.random.rand(N)
#     best_rate = 0

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
    [conv, theta] = optim(params)

    if conv:
        print "Convergence!"
        print_theta(theta, N, vars)
    else:
        print "Error: cost function increased..."
        print "Try adjusting the learning or the regularization coefficients"

    plt.plot(np.arange(values.shape[0]), values)
    plt.xlabel("Step number")
    plt.ylabel("Cost function")
    
    if showp:
        plt.show()

    print ""
    print "***************************************"
    print "Best predictor:"
    print_theta(theta, N, vars)
    save_theta(param_filename, theta, N, vars)
    
'''
    for n in range(0, iter):
        print "-------> Training iteration",n

        # Create training set by randomly choosing a fraction of rows from each output
        # category
        i0 = np.where(y == 0)
        i1 = np.where(y == 1)
        ri0 = np.random.choice(i0[0], size=trainf*i0[0].shape[0], replace=False)
        ri1 = np.random.choice(i1[0], size=trainf*i1[0].shape[0], replace=False)
        itrain = np.concatenate((ri1, ri0))
        itrain.sort()

    Xtrain = X[itrain,:]
    ytrain = y[itrain]

    values = np.array([])
    params = (Xtrain, ytrain, gamma)
    [conv, theta] = train(params)

    if conv:
        print "Convergence!"
        print_theta(theta, vars)
    else:
        print "Error: cost function increased..."
        print "Try adjusting the learning or the regularization coefficients"

    if trainf < 1: 
        [rate, rrate] = evaluate(X, y, itrain, theta)
        if best_rate < rate:
            best_rate = rate
            best_theta = theta
    else:
        best_theta = theta
    
    plt.plot(np.arange(values.shape[0]), values)
    plt.xlabel("Step number")
    plt.ylabel("Cost function")
    
    if showp:
        plt.show()

    print ""
    print "***************************************"
    print "Best predictor:"
    print_theta(best_theta, vars)
    save_theta(param_filename, best_theta)
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./data/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./data/lreg-params"],
                        help="Output file to save the parameters of the neural net")
    parser.add_argument("-l", "--layers", nargs=1, type=int, default=[1],
                        help="Number of hidden layers")
    parser.add_argument("-f", "--hfactor", nargs=1, type=int, default=[1],
                        help="Hidden units factor")
    parser.add_argument("-g", "--gamma", nargs=1, type=float, default=[0.002],
                        help="Regularization coefficient")
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
          gamma=str(args.gamma[0]),
          threshold=str(args.convergence[0]),
          show=str(args.show),
          debug=str(args.debug))
          
'''
iter      = 1     # Number of training runs 
trainf    = 1     # Fraction of rows from training data to use in each training iteration.
                  # The rest of rows are used to calculate the accuracy of the parameters,
                  # so that the final parameters are chosen to be those that maximize
                  # the accuracy.
gamma     = 0.08  # Regularization coefficient
showp     = False # Show minimization plot
gcheck    = False # Gradient check
'''
