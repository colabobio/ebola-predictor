'''
Build a predictive model that combines multiple nnets and DTs.
'''

import argparse
import numpy as np
import pandas as pd
import cloud

from dt.dt_utils import dt_pred_model, design_matrix
from dt.dt_train import train as dttrain

from nnet.makepredictions import nnet_pred_model
from nnet.train import train as nnettrain

from utils.evaluate import eval
from utils.makesets import makesets
from utils.impute import impute

def prepare_data():
	makesets()
	impute()

def build_nnet():
	prepare_data()
	nnettrain()
	return nnet_pred_model()
	
def build_dt():
	prepare_data()
	dttrain()
	return dt_pred_model()
	
def final_prediction(predictive_models):
	
	ys = []
	
	def pred_model(X):
	
		for f in predictive_models:
			y = f(X)						
			ys.append(y)
			
		return np.mean(np.array(ys), axis=0)
		
	return pred_model
	
def save_model(final_pred, outfile):
	
	f = open(outfile, 'wb')
	cloud.serialization.cloudpickle.dump(final_pred, f)
	
def eval_pred(pred_model, methods):

	makesets(100)
	
	df = pd.read_csv('data/testing-data.csv', delimiter=",", na_values="?")
	X, y_test = design_matrix(df)
	probs = pred_model(X)
		
	for method in methods:
		if method > 4 or method < 1:
			raise Exception("Invalid method given")
	
		eval(probs, y_test, method)

parser = argparse.ArgumentParser()

# Number of nnets
parser.add_argument('-n', type=int, nargs=1, default=[0])

# Number of decision trees
parser.add_argument('-d', type=int,  nargs=1, default=[0])

# File to save final model in
parser.add_argument('-s', type=str,  nargs=1, default=['model.p'])

# If flagged, evaluate the model with given method(s)
parser.add_argument('-e', type=int,  nargs='+')

args = parser.parse_args()

nnets = args.n[0]
dts = args.d[0]

models = []

for _ in range(nnets):
	models.append(build_nnet())

for _ in range(dts):
	models.append(build_dt())

final_model = final_prediction(models)

if args.s:
	outfile = args.s[0]
	save_model(final_model, outfile)

if args.e:
	eval_pred(final_model, args.e)