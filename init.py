"""
Create training and test sets with optional imputation.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, sys, argparse, glob

from utils.makesets import makesets
from importlib import import_module
#from utils.listdel import listdel
# from utils.impute import impute

"""Creates training/test sets using the provided parameters

:param iter_count: number of training/test sets to create
:param test_percentage: percentage of complete rows to include in the test set
:param num_imputed: number of intermediate imputed sets if imputation is selected
:param id_start: id of first group of sets
:param impute_method: name of script in utils folder containing the imputation algorithm
:param kwparams: custom arguments that the imputation method can receive
"""
def create_sets(iter_count, id_start, test_percentage, impute_method, **kwparams):
    if id_start == 0:
        # remove old data
        test_files = glob.glob("./data/testing-data*.csv")
        train_files = glob.glob("./data/training-data*.csv")
        idx_files = glob.glob("./data/*-idx*")
        if test_files or train_files or idx_files:
            print "Removing old sets..."
        for file in test_files: os.remove(file)
        for file in train_files: os.remove(file)
        for file in idx_files: os.remove(file)
        if test_files or train_files or idx_files:
            print "Done."

    module_path = os.path.abspath("./utils")
    sys.path.insert(0, module_path)
    module = import_module(impute_method)

    for i in range(iter_count):
        id = i + id_start
        print "Creating training/test sets #" + str(id) + "..."
        test_filename = "./data/testing-data-"+str(id)+".csv"
        train_filename = "./data/training-data-"+str(id)+".csv"
        idx_filename = "./data/idx-" + str(id)
        completed_filename = "./data/training-data-completed-"+str(id)+".csv"
        makesets(test_percentage, test_filename, train_filename, idx_filename)
        module.process(in_filename=train_filename, out_filename=completed_filename, **kwparams)
        print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number', type=int, nargs=1, default=[10],
                        help="Number of training/test sets")
    parser.add_argument('-s', '--start', type=int, nargs=1, default=[0],
                        help="ID of starting training/test set")
    parser.add_argument('-t', '--test', type=int, nargs=1, default=[50],
                        help="Percentage of complete rows used to build the test set")
    parser.add_argument('-m', '--method', nargs=1, default=["amelia"],
                        help="Name of script implementing imputation method")
    parser.add_argument('vars', nargs='*',
                        help="Custom arguments for imputation algorithm")
    args = parser.parse_args()

    iter_count = args.number[0]
    id_start = args.start[0]
    test_percentage = args.test[0]
    impute_method = args.method[0]
    kwargs = {}
    for var in args.vars:
        [k, v] = var.split("=")
        kwargs[k] = v
    create_sets(iter_count, id_start, test_percentage, impute_method, **kwargs)
