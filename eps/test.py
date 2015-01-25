'''
Evaluates the predictor on the test set.
'''

import sys
import pandas as pd
import numpy as np
from eps_utils import design_matrix, eps_dataframe, eps_pred

testing_filename = "./data/testing-data.csv"

def evaluate(probs, y):
    n_total = 0
    
    n_out1 = 0
    n_out0 = 0
        
    n_hit = 0          # Hit or True Positive (TP)
    n_correct_rej = 0  # Correct rejection or True Negative (TN)
    n_miss = 0         # Miss or False Negative (FN)    
    n_false_alarm = 0  # False alarm, or False Positive (FP)
    
    failed_preds = []
    
    for i, p in enumerate(probs):
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
        
        n_total = n_total + 1
    
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

probs, y_test = eps_pred()

print probs

fails = evaluate(probs, y_test)

print ""
print "---------------------------------------"
print "Failed predictions:"

if fails:
# 	df = pd.read_csv(testing_filename, delimiter=",", na_values="?")
	df = eps_dataframe(testing_filename)
	for i in fails:
		print "****** Patient", i, "******"
		print df.ix[i]
else:
    print "None!"    
    