'''
Evaluates the predictor on the test set.
'''

import sys
import pandas as pd
import numpy as np

testing_filename = "testing-data.csv"

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

def predict(x, theta, N, L, S, K):
    thetam = thetaMatrix(theta, N, L, S, K)
    a = forwardProp(x, thetam, L) 
    h = a[L]
    return h;

def print_theta(theta):
    for x in theta:
        print "{:3.5f}".format(x)

def evaluate(X, y, theta):
    n_total = 0
    
    n_out1 = 0
    n_out0 = 0
        
    n_hit = 0          # Hit or True Positive (TP)
    n_correct_rej = 0  # Correct rejection or True Negative (TN)
    n_miss = 0         # Miss or False Negative (FN)    
    n_false_alarm = 0  # False alarm, or False Positive (FP)
    
    failed_preds = []
    
    for i in range(0, M):
        n_total = n_total + 1
        p = predict(X[i,:], theta, N, L, S, K)
        pred = 0.5 < p
        
        if pred == 1:
            if y[i] == 1: 
                n_hit = n_hit + 1
            else: 
                n_false_alarm = n_false_alarm + 1
                failed_preds.append(i)
        else:
            if y[i] == 1: 
                n_miss = n_miss + 1
                failed_preds.append(i)
            else: 
                n_correct_rej = n_correct_rej + 1

        if y[i] == 1: 
            n_out1 = n_out1 + 1
        else:    
            n_out0 = n_out0 + 1
    
    hit_rate = float(n_hit) / n_out1
    correct_rej_rate = float(n_correct_rej) / n_out0    
    miss_rate = float(n_miss) / n_out1
    false_alarm_rate = float(n_false_alarm) / n_out0    
        
    accuracy = float(n_hit + n_correct_rej) / n_total
    sensitivity = float(n_hit) / n_out1
    precision = float(n_hit) / (n_hit + n_false_alarm)
        
    print ""
    print "---------------------------------------"
    print "Confusion matrix"
    print "{:10s} {:5s} {:5s}".format("", "Out 1", "Out 0")
    print "{:10s} {:2.0f}    {:2.0f}".format("Pred 1", n_hit, n_false_alarm)
    print "{:10s} {:2.0f}    {:2.0f}".format("Pred 0", n_miss, n_correct_rej)    
    
    print ""    
    print "Predictor accuracy   :", round(100 * accuracy, 2), "%"
    print "Predictor sensitivity:", round(100 * sensitivity, 2), "%"
    print "Predictor precision  :", round(100 * precision, 2), "%"
    print ""
    print "Null accuracy        :", round(100 * float(n_out1) / n_total, 2), "%"
    print "Null sensitivity     : 100%"
    print "Null precision       :", round(100 * float(n_out1) / n_total, 2), "%"
        
    return failed_preds  

##########################################################################################

# Main
        
df = pd.read_csv(testing_filename, delimiter=",", na_values="?")
M = df.shape[0]
N = df.shape[1]

with open("predictor.txt", "rb") as pfile:
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

fails = evaluate(X, y, theta)

print ""
print "---------------------------------------"
print "Failed predictions:"

if fails:
    for i in fails:
        print "****** Patient", i, "******"
        print df.ix[i]
else:
    print "None!"    
    