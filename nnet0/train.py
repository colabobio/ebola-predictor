# This code implements a logistic regression predictor. The dependent variable
# must be binary and the first column in the data frame, and all the independent
# categorical variables must be binary as well.
# Requires:
# pandas http://pandas.pydata.org/
# numpy http://www.numpy.org/

import sys
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def cost(theta, X, y, gamma):
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

def debug(theta):
    global params 
    global values
    (X, y, gamma) = params
    value = cost(theta, X, y, gamma);
    values = np.append(values, [value])
    # print value

def train(params, alpha, threshold, useBfgs):
    global values
    (X, y, gamma) = params
    M = X.shape[0]
    N = X.shape[1]

    if useBfgs:
        print ''
        print 'Running BFGS minimization...'        
        theta0 = np.ones(N)
        thetaOpt = fmin_bfgs(cost, theta0, fprime=gradient, args=params, gtol=threshold, callback=debug)
        return [True, thetaOpt]
    else:
        print ''
        print 'Running simple gradient descent...'
        theta = np.ones(N)
        count = 0 
        cost0 = cost(theta, X, y, gamma)
        conv = False
        while not conv:
            grad = gradient(theta, X, y, gamma)
            theta = theta - alpha * grad

            cost1 = cost(theta, X, y, gamma)
            diff = cost1 - cost0
            cost0 = cost1
            count = count + 1
            #print count, cost1, diff 
            values = np.append(values, [cost1])
            if 3 < diff: break
            conv = np.linalg.norm(grad) < threshold
        return [conv, theta]

# Main -----------------------------------------------------------------------------

# Loading data frame and initalizing dimensions
df = pd.read_csv(filename, delimiter=',', na_values="\\N")
M = df.shape[0]
N = df.shape[1]
print 'Number of independent variables:', N-1 
print 'Number of data samples         :', M

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
    X[:, j] = (values - minv) / (maxv - minv)

rates = np.array([])
for iter in range(0, testCount):
    if testMode: print "-------> Iteration test",iter

    # Create training set by randomly choosing 70% of rows from each output
    # category
    i0 = np.where(y == 0)
    i1 = np.where(y == 1)
    ri0 = np.random.choice(i0[0], size=0.7*i0[0].shape[0], replace=False)
    ri1 = np.random.choice(i1[0], size=0.7*i1[0].shape[0], replace=False)
    itrain = np.concatenate((ri1, ri0))
    itrain.sort()

    Xtrain = X[itrain,:]
    ytrain = y[itrain]

    values = np.array([])
    params = (Xtrain, ytrain, gamma)
    [conv, theta] = train(params, alpha, threshold, useBfgs)

    if conv:
        print 'Convergence!'
        print '{:10s} {:2.4f}'.format('Intercept', theta[0])
        names = df.columns.values[1: N]
        for i in range(1, N):
            print '{:10s} {:3.5f}'.format(names[i-1], theta[i])
    else:
        print 'Error: cost function increased...'
        print 'Try adjusting the learning or the regularization coefficients'

    # Calculating the prediction rate by applying the trained model on the remaining
    # 30% of the data (the test set), and comparing with random selection
    ntot = 0
    nhit = 0
    nran = 0
    for i in range(0, M):
        if not i in itrain:
            ntot = ntot + 1
            p = sigmoid(np.dot(X[i,:], theta))
            if (y[i] == 1) == (0.5 < p):
                nhit = nhit + 1
            r = np.random.rand()
            if (y[i] == 1) == (0.5 < r):
                nran = nran + 1

    rate = float(nhit) / float(ntot)
    rrate = float(nran) / float(ntot)
    rates = np.append(rates, [rate]) 

    print ''
    print '---------------------------------------'
    print 'Predictor success rate on test set:', round(100 * rate, 2), '%'
    print 'Random success rate on test set   :', round(100 * rrate, 2), '%'

    if showp and not testMode:
        plt.plot(np.arange(values.shape[0]), values)
        plt.xlabel('Step number')
        plt.ylabel('Cost function')
        plt.show()

if testMode:
    print ''
    print '***************************************'
    print 'Average success rate:', round(100 * np.average(rates), 2), '%'
    print 'Standard deviation  :', round(100 * np.std(rates, ddof=1), 2), '%'