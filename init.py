"""
Create training and test sets with optional imputation.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import os
import glob

from utils.listdel import listdel
from utils.makesets import makesets
from utils.impute import impute

"""Creates a round of training/test sets

:param test_percentage: percentage of complete rows to include in the test set
:param num_imputed: number of intermediate imputed sets if imputation is selected
:param id: round id
"""
def create_sets_round(test_percentage, num_imputed, id):
    test_filename = "./data/testing-data-"+str(id)+".csv"
    train_filename = "./data/training-data-"+str(id)+".csv"
    completed_filename = "./data/training-data-completed-"+str(id)+".csv"
    makesets(test_percentage, test_filename, train_filename)
    if num_imputed > 0:
        impute(num_imputed, train_filename, completed_filename)
    else:
        listdel(in_file=train_filename, out_file=completed_filename)

"""Creates training/test sets using the provided parameters

:param iter_count: number of training/test sets to create
:param test_percentage: percentage of complete rows to include in the test set
:param num_imputed: number of intermediate imputed sets if imputation is selected
:param id_start: id of first group of sets
"""
def create_sets(iter_count, test_percentage, num_imputed, id_start):
    if id_start == 0:
        # remove old data
        test_files = glob.glob("./data/testing-data*.csv")
        train_files = glob.glob("./data/training-data*.csv")
        if test_files or train_files:
            print "Removing old sets..."
        for file in test_files:
            os.remove(file)
        for file in train_files:
            os.remove(file)
        if test_files or train_files:
            print "Done."

    for i in range(iter_count):
        print "Creating training/test sets #" + str(i) + "..."
        create_sets_round(test_percentage, num_imputed, i+id_start)
        print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Number of iterations
    parser.add_argument('-n', type=int, nargs=1, default=[0])

    # Test percentage
    parser.add_argument('-t', type=int, nargs=1, default=[70])

    # If flagged, impute the data with number given
    parser.add_argument('-i', type=int, nargs=1, default=[0])

    # If flagged, start the ids at this value
    parser.add_argument('-s', type=int, nargs=1, default=[0])

    args = parser.parse_args()

    iter_count = args.n[0]
    test_percentage = args.t[0]
    num_imputed = args.i[0]
    id_start = args.s[0]

    create_sets(iter_count, test_percentage, num_imputed, id_start)
