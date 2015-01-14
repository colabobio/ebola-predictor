'''
Runs predictors on test sets, returns scores (probabilities).
'''

import sys
import pandas as pd
import numpy as np

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
    return h

def nnet_pred(testing_filename = "./data/testing-data.csv", predict_filename = "./data/predictor.txt"):
	# Return probabilities, test binary values
	
	df = pd.read_csv(testing_filename, delimiter=",", na_values="?")
	M = df.shape[0]
	N = df.shape[1]
	
	with open(predict_filename, "rb") as pfile:
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

	scores = []

	for i in range(0, M):
		scores.extend(predict(X[i,:], theta, N, L, S, K))
	
	return scores, y
	
def nnet_pred_model(predict_filename = "./data/predictor.txt"):
	# Return a function that gives a prediction from a design matrix row
	with open(predict_filename, "rb") as pfile:
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
	
	def pred_model(X):
		scores = []

		for i in range(0, len(X)):
			scores.extend(predict(X[i,:], theta, N, L, S, K))
	
		return scores
			
	return pred_model
	