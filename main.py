'''
Build a predictive model that combines multiple nnets and DTs.
'''

import argparse
import numpy as np
import pandas as pd
import cloud


from matplotlib import pyplot as plt

from dt.dt_utils import dt_pred_model, design_matrix
from dt.dt_train import train as dttrain

from nnet.makepredictions import nnet_pred_model
from nnet.train import train as nnettrain

from utils.evaluate import eval
from utils.makesets import makesets
from utils.impute import impute
from utils.roc import roc

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
	
def aggregate_models(predictive_models):
	
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

def avg_cal_dis(pred_model_dict, X, y_test):
	# Method 1
	output = ''
	
	for model_type in pred_model_dict:
	
		total_cal = 0
		total_dis = 0
		N = len(pred_model_dict[model_type])
	
		for model in pred_model_dict[model_type]:
			probs = model(X)
			cal, dis = eval(probs, y_test, 1)
			total_cal += cal
			total_dis += dis
			
		if N > 0:
			avg_cal = total_cal/N
			avg_dis = total_dis/N
			
			output += '\n'
			output += "Summary for "+model_type+'\n'
			output += "Average calibration: "+str(avg_cal)+'\n'
			output += "Average discrimination: "+str(avg_dis)+'\n'
			output += '\n'
			
	print output
	return output
		
	
def calplots(pred_model_dict, X, y_test):
	# Method 2
	raise Exception("Not implemented")
	
def avg_reports(pred_model_dict, X, y_test):
	# Method 3
	output = ''
	
	for model_type in pred_model_dict:
	
		total_prec = 0
		total_rec = 0
		total_f1 = 0
		N = len(pred_model_dict[model_type])
	
		for model in pred_model_dict[model_type]:
			probs = model(X)
			p, r, f, _ = eval(probs, y_test, 3)
			total_prec += p
			total_rec += r
			total_f1 += f 
	
		if N > 0:
			avg_prec = total_prec/N
			avg_rec = total_rec/N
			avg_f1 = total_f1/N
		
			output += '\n'
			output += "Summary for "+model_type+'\n'
			output += "Average precision: "+str(avg_prec)+'\n'
			output += "Average recall: "+str(avg_rec)+'\n'
			output += "Average F1: "+str(avg_f1)+'\n'
			output += '\n'
			
	print output
	return output
	
def rocplots(pred_model_dict, X, y_test):
	# Method 4
	
	markers = {'nnet' : 'o', 'dt' : 's'}
	plt.clf()
	
	for model_type in pred_model_dict:
		for model in pred_model_dict[model_type]:
			probs = model(X)
			roc(probs, y_test, mark=markers[model_type], label=model_type, pltshow = False)
	
	plt.show()
	
def eval_pred(pred_model_dict, methods):

	makesets(100)
	
	df = pd.read_csv('data/testing-data.csv', delimiter=",", na_values="?")
	X, y_test = design_matrix(df)
	
	for method in methods:
		# Average calibrations and discriminations
		if method == 1:
			avg_cal_dis(pred_model_dict, X, y_test)
	
		# Plot each method on same calibration plot
		elif method == 2:
			calplots(pred_model_dict, X, y_test)
	
		# Average precision, recall, and F1 scores
		elif method == 3:
			avg_reports(pred_model_dict, X, y_test)
	
		# Plot each method on same ROC plot
		elif method == 4:
			rocplots(pred_model_dict, X, y_test)
	
		# Method not defined:
		else:
			raise Exception("Invalid method given")
	

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

nnet_count = args.n[0]
dt_count = args.d[0]

models = {'nnet' : [], 'dt' : []}

# Build all the models
for _ in range(nnet_count):
	models['nnet'].append(build_nnet())

for _ in range(dt_count):
	models['dt'].append(build_dt())
	
if args.s:
	outfile = args.s[0]
	all_models = sum(models.values(), [])
	final_model = aggregate_models(all_models)
	save_model(final_model, outfile)

if args.e:
	eval_pred(models, args.e)