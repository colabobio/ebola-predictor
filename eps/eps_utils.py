'''
Utility functions for the decision tree classifier.
'''

import csv
import numpy as np
import pandas as pd
import pickle

use_naive_classifier = False

def design_matrix(df):
	M = df.shape[0]
	N = df.shape[1]
	
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
		
	return X, y
	
def eps_dataframe(data_filename, threshold_filename="./eps/thresholds.txt"):
    thresh_variables = []
    model_variables = []
    with open(threshold_filename, "rb") as tfile:
        for line in tfile.readlines():
            parts = line.strip().split()        
            thresh_variables.append({"name":parts[0], "type":parts[1], "threshold":parts[2]})
            model_variables.append(parts[0])    
    data = []
    index = []
    columns = ["OUT", "SCORE"]
    with open(data_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        model_idx = [model_variables.index(var) for var in titles[1:]]
        count = 0
        for row in reader:
            out = row[0]
            score = 0
            for i in range(1, len(row)):
                var = thresh_variables[i - 1]
                if var["type"] == "category": 
                    score = score + (1 if row[i] == var["threshold"] else 0) 
                else:
                    str = var["threshold"]
                    if str[0] == "<":
                        str = str[1:]
                        score = score + (1 if float(row[i]) <= float(str) else 0) 
                    else:
                        score = score + (1 if float(row[i]) >= float(str) else 0)
            index.append(count)
            data.append([float(out), float(score)])
            count = count + 1
    df = pd.DataFrame(data, index=index, columns=columns)
    return df

def eps_pred(testing_filename = "./data/testing-data.csv", model_filename = "./data/eps-model.p"):
	# Runs predictors on test sets, returns scores (probabilities).
	
	# Load the test data
# 	df = pd.read_csv(testing_filename, delimiter=",", na_values="?")
	df = eps_dataframe(testing_filename)
	X, y = design_matrix(df)
	
	if use_naive_classifier:
		probs = [0 if s < 2 else 1 for s in df["SCORE"]]
	else:
		# Load the classifier
		clf = pickle.load(open( model_filename, "rb" ) )
		# Make predictions
		scores = clf.predict_proba(X)	
		probs = [x[1] for x in scores]
	
	return probs, y
	
def eps_pred_model(model_filename = "./data/eps-model.p"):

	# Load the decision tree
	clf = pickle.load(open( model_filename, "rb" ) )

	def pred_model(X):
		scores = clf.predict_proba(X)
		probs = [x[1] for x in scores]
		return probs

	return pred_model