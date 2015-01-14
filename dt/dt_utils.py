'''
Utility functions for the decision tree classifier.
'''

import numpy as np
import pandas as pd
import pickle

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
	
def dt_pred(testing_filename = "./data/testing-data.csv", model_filename = "./data/dt-model.p"):
	# Runs predictors on test sets, returns scores (probabilities).
	
	# Load the test data
	df = pd.read_csv(testing_filename, delimiter=",", na_values="?")
	X, y = design_matrix(df)
	
	# Load the decision tree
	clf = pickle.load(open( model_filename, "rb" ) )

	# Make predictions
	scores = clf.predict_proba(X)
	probs = [x[1] for x in scores]
	
	return probs, y
	
def dt_pred_model(model_filename = "./data/dt-model.p"):

	# Load the decision tree
	clf = pickle.load(open( model_filename, "rb" ) )

	def pred_model(X):
		scores = clf.predict_proba(X)
		probs = [x[1] for x in scores]
		return probs

	return pred_model