"""
Cleans data folder

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob

test_files = glob.glob("./data/testing-data*.csv")
train_files = glob.glob("./data/training-data*.csv")
param_files = glob.glob("./data/*-params*")
idx_files = glob.glob("./data/*-idx*")
out_files = glob.glob("./out/*")

if test_files or train_files or param_files or idx_files or out_files:
    print "Cleaning folders..."
    for file in test_files: os.remove(file)
    for file in train_files: os.remove(file)
    for file in param_files: os.remove(file)
    for file in idx_files: os.remove(file)
    for file in out_files: os.remove(file)
    print "Done."
else:
    print "Nothing to do."