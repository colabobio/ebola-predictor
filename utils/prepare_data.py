"""
Prepare training and test sets with optional imputation.
"""

import argparse
import os
import glob

from del_missing import remove
from makesets import makesets
from impute import impute

def prepare_data_round(test_percentage, num_imputed, id):
	print "Preparing data..."
		
	test_file_name = "./data/testing-data-"+str(id)+".csv"
	train_file_name = "./data/training-data-"+str(id)+".csv"
	aggregated_file_name = "./data/training-data-imputed-"+str(id)+".csv"

	print "Making sets..."
	makesets(test_percentage, testing_file=test_file_name, training_file=train_file_name)
	
	if num_imputed > 0:
		print "Imputing..."
		impute(num_imputed, training_file=train_file_name, aggregated_file=aggregated_file_name)
		
	else:
		remove(training_file=train_file_name, out_file=aggregated_file_name)
	
	print "Done."

def prepare_data(iter_count, test_percentage, num_imputed, id_start):
	if id_start == 0:
		# remove old data
		test_files = glob.glob('data/testing-data*.csv')
		train_files = glob.glob('data/training-data*.csv')
		for file in test_files:
			os.remove(file)
		for file in train_files:
			os.remove(file)
		
	for i in range(iter_count):
		prepare_data_round(test_percentage, num_imputed, i+id_start)

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Number of iterations
	parser.add_argument('-n', type=int, nargs=1, default=[0])

	# Test percentage
	parser.add_argument('-t', type=int, nargs=1, default=[70])

	# If flagged, impute the data with number given
	parser.add_argument('-i', type=int,  nargs=1, default=[0])
	
	# If flagged, start the ids at this value
	parser.add_argument('-s', type=int,  nargs=1, default=[0])

	args = parser.parse_args()

	iter_count = args.n[0]
	test_percentage = args.t[0]
	num_imputed = args.i[0]
	id_start = args.s[0]
	
	prepare_data(iter_count, test_percentage, num_imputed, id_start)
	