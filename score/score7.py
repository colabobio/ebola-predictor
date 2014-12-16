import sys
import pandas as pd
import numpy as np

testing_filename = "profile-data.tsv"

def sigmoid(v):
    return 1 / (1 + np.exp(-v))

def print_theta(theta, names):
    print "---------------------------------------"
    print "Predictor parameters:"
    print "{:10s} {:2.4f}".format("Intercept", theta[0])
    for i in range(1, N):
        print "{:10s} {:3.5f}".format(names[i-1], theta[i])

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
        p = sigmoid(np.dot(X[i,:], theta))
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
vars = df.columns.values[1: N] 

theta = np.ones(N)
with open("predictor.txt", "rb") as pfile:
    i = 0
    for line in pfile.readlines():
        theta[i] = float(line.strip().split(' ')[1])
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

print_theta(theta, vars)
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