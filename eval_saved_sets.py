"""
Run evaluation metrics on saved data sets prepared by prepare_data.
"""

import argparse
import glob
import pandas as pd
import rpy2.robjects as robjects

from matplotlib import pyplot as plt

from utils.calplot import calplot
from utils.evaluate import eval
from utils.roc import roc

from dt.dt_utils import dt_pred_model, design_matrix
from dt.dt_train import train as dttrain

from eps.eps_utils import eps_pred_model, eps_dataframe
from eps.train import train as epstrain

from nnet.makepredictions import nnet_pred_model
from nnet.train import train as nnettrain


def find_saved_sets():

	file_ids = []
	
	files = glob.glob('data/testing-data-*.csv')
	
	for file in files:
		start_idx = file.find('testing-data-')+len('testing-data-')
		stop_idx = file.find('.csv')
		file_id = file[start_idx:stop_idx]
		file_ids.append(file_id)

	return file_ids

def build_eps(file_name):
	print "Training EPS from "+file_name
	epstrain(training_filename=file_name)
	return eps_pred_model()

def build_nnet(file_name):
	print "Training nnet from "+file_name
	nnettrain(training_filename=file_name)
	return nnet_pred_model()

def build_dt(file_name):
	print "Training DT from "+file_name
	dttrain(training_filename=file_name)
	return dt_pred_model()

def avg_cal_dis(file_ids):
    # Method 1
	
	output = ''
	
	total_cal ={'eps': 0, 'dt': 0, 'nnet': 0}
	total_dis ={'eps': 0, 'dt': 0, 'nnet': 0}

	for id in file_ids:

		train_file = "data/training-data-imputed-"+str(id)+".csv"
		test_file = "data/testing-data-"+str(id)+".csv"

		pred_model_dict = {'eps': build_eps(train_file), 'dt': build_dt(train_file), 'nnet': build_nnet(train_file)}
		
		df = pd.read_csv(test_file, delimiter=",", na_values="?")
		X, y_test = design_matrix(df)
		
		X_eps = eps_dataframe(test_file)

		for model_type, model in pred_model_dict.items():
		
			print "Evaluating "+model_type
		
			if model_type == 'eps':
				probs = model(X_eps)
			else:
				probs = model(X)
			cal, dis = eval(probs, y_test, 1)
			total_cal[model_type] += cal
			total_dis[model_type] += dis

	N = len(file_ids)
	
	if N > 0:
		for model_type in pred_model_dict:
		
			avg_cal = total_cal[model_type]/N
			avg_dis = total_dis[model_type]/N

			output += '\n'
			output += "Summary for "+model_type+'\n'
			output += "Average calibration: "+str(avg_cal)+'\n'
			output += "Average discrimination: "+str(avg_dis)+'\n'
			output += '\n'
		
	print output
	return output

def calplots(file_ids):
	# Method 2
	colors = {'eps' : 'green', 'nnet' : 'red', 'dt' : 'blue'}

	robjects.r.X11()

	for id in file_ids:

		train_file = "data/training-data-imputed-"+str(id)+".csv"
		test_file = "data/testing-data-"+str(id)+".csv"

		pred_model_dict = {'eps': build_eps(train_file), 'dt': build_dt(train_file), 'nnet': build_nnet(train_file)}

		df = pd.read_csv(test_file, delimiter=",", na_values="?")
		X, y_test = design_matrix(df)

		X_eps = eps_dataframe(test_file)

		for model_type, model in pred_model_dict.items():
			if model_type == 'eps':
				probs = model(X_eps)
			else:
				probs = model(X)
			calplot(probs, test_file, "data/calibration-"+model_type+str(id)+".txt", color=colors[model_type])

	# Plot the calibrations
	print "Press Ctrl+C to quit"
	while True:
		continue

def avg_reports(file_ids):
	# Method 3
	output = ''

	total_prec ={'eps': 0, 'dt': 0, 'nnet': 0}
	total_rec ={'eps': 0, 'dt': 0, 'nnet': 0}
	total_f1 ={'eps': 0, 'dt': 0, 'nnet': 0}

	for id in file_ids:

		train_file = "data/training-data-imputed-"+str(id)+".csv"
		test_file = "data/testing-data-"+str(id)+".csv"

		pred_model_dict = {'eps': build_eps(train_file), 'dt': build_dt(train_file), 'nnet': build_nnet(train_file)}
	
		df = pd.read_csv(test_file, delimiter=",", na_values="?")
		X, y_test = design_matrix(df)
	
		X_eps = eps_dataframe(test_file)

		for model_type, model in pred_model_dict.items():
	
			print "Evaluating "+model_type
	
			if model_type == 'eps':
				probs = model(X_eps)
			else:
				probs = model(X)
			p, r, f, _ = eval(probs, y_test, 3)
			total_prec[model_type] += p
			total_rec[model_type] += r
			total_f1[model_type] += f

	N = len(file_ids)

	if N > 0:
		for model_type in pred_model_dict:
	
			avg_prec = total_prec[model_type]/N
			avg_rec = total_rec[model_type]/N
			avg_f1 = total_f1[model_type]/N

			output += '\n'
			output += "Summary for "+model_type+'\n'
			output += "Average precision: "+str(avg_prec)+'\n'
			output += "Average recall: "+str(avg_rec)+'\n'
			output += "Average F1: "+str(avg_f1)+'\n'
			output += '\n'
	
	print output
	return output


def rocplots(file_ids):
    # Method 4

    colors = {'eps' : 'green', 'nnet' : 'red', 'dt' : 'blue'}
    plt.clf()
    
    for id in file_ids:

		train_file = "data/training-data-imputed-"+str(id)+".csv"
		test_file = "data/testing-data-"+str(id)+".csv"

		pred_model_dict = {'eps': build_eps(train_file), 'dt': build_dt(train_file), 'nnet': build_nnet(train_file)}

		df = pd.read_csv(test_file, delimiter=",", na_values="?")
		X, y_test = design_matrix(df)

		X_eps = eps_dataframe(test_file)

		for model_type, model in pred_model_dict.items():
			if model_type == 'eps':
				probs = model(X_eps)
			else:
				probs = model(X)
			roc(probs, y_test, color=colors[model_type], label=model_type, pltshow = False)

    plt.show()

def eval_pred(file_ids, methods):

	print "Evaluating file ids: ", file_ids

	for method in methods:
		# Average calibrations and discriminations
		if method == 1:
			avg_cal_dis(file_ids)

		# Plot each method on same calibration plot
		elif method == 2:
			calplots(file_ids)

		# Average precision, recall, and F1 scores
		elif method == 3:
			avg_reports(file_ids)

		# Plot each method on same ROC plot
		elif method == 4:
			rocplots(file_ids)

		# Method not defined:
		else:
			raise Exception("Invalid method given")

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Evaluate the model with given method(s)
	parser.add_argument('-e', type=int,  nargs='+')

	args = parser.parse_args()

	file_ids = find_saved_sets()

	eval_pred(file_ids, args.e)