'''
Trains the Neural Network predictor.
'''

import sys
import math
import pandas as pd
import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt

training_filename = "./data/training-data-imputed.csv"

iter      = 1     # Number of training runs 
trainf    = 1     # Fraction of rows from training data to use in each training iteration.
                  # The rest of rows are used to calculate the accuracy of the parameters,
                  # so that the final parameters are chosen to be those that maximize
                  # the accuracy.
L = 1             # Number of hidden layers
hf = 1            # Factor to calculate number of hidden units given the number of variables         
gamma = 0.002     # Regularization coefficient
threshold = 1E-5  # Default convergence threshold
showp = False     # Show minimization plot
gcheck = False    # Gradient check

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

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def forwardProp(x, thetam, L):
    a = [None] * (L + 1)
    a[0] = x
    for l in range(0, L):            
        z = np.dot(thetam[l], a[l])
        res = sigmoid(z)
        a[l + 1] = np.insert(res, 0, 1) if l < L - 1 else res
    return a

def backwardProp(y, a, thetam, L, N):
    err = [None] * (L + 1)
    err[L] = a[L] - y
    for l in range(L - 1, 0, -1):  
        backp = np.dot(np.transpose(thetam[l]), err[l + 1])
        deriv = np.multiply(a[l], 1 - a[l])
        err[l] = np.delete(np.multiply(backp, deriv), 0)
    err[0] = np.zeros(N);
    return err

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

def predict(x, theta, N, L, S, K):
    thetam = thetaMatrix(theta, N, L, S, K)
    a = forwardProp(x, thetam, L) 
    h = a[L]
    return h;

def debug(theta):
    global params 
    global values
    (X, y, N, L, S, K, gamma) = params
    value = cost(theta, X, y, N, L, S, K, gamma);
    values = np.append(values, [value])

def evaluate(X, y, itrain, theta):
    # Calculating the prediction rate by applying the trained model on the remaining
    # fraction of the data (the test set), and comparing with random selection
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

##########################################################################################    
#
# Main

K = 1

if L < 1:
    print "Need to have at least one hidden layer"
    sys.exit(1)

L = L + 1

# Loading data frame and initalizing dimensions
df = pd.read_csv(training_filename, delimiter=',', na_values="?")
M = df.shape[0]
N = df.shape[1]
S = int((N - 1) * hf) # includes the bias unit on each layer, so the number of units is S-1
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

best_theta = np.ones(R)
best_rate = 0

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

for n in range(0, iter):
    print "-------> Training iteration",n

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

    theta0 = np.random.rand(R)
    params = (Xtrain, ytrain, N, L, S, K, gamma)

    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_bfgs.html
    print "Training Neural Network..."
    values = np.array([])
    theta = fmin_bfgs(cost, theta0, fprime=gradient, args=params, gtol=threshold, callback=debug)
    print "Done!"

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
print_theta(best_theta, N, L, S, K)
save_theta("./data/predictor.txt", best_theta, N, L, S, K)
