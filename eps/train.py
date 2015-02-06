"""
Dummy train module for EPS

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse
import pandas as pd
import pickle
from sklearn import tree
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix

def prefix():
    return "eps"

def title():
    return "Ebola Prognosis Score"

def train(train_filename, param_filename):
    print "Nothing to train in EPS."

if __name__ == "__main__":
    train("", "")