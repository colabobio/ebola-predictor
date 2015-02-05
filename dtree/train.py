"""
Trains the Decision Tree predictor from scikit-learn:
http://scikit-learn.org/stable/modules/tree.html

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse
import pandas as pd
import pickle
from sklearn import tree
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix

def prefix():
    return "dtree"

def title():
    return "Decision Tree"

def train(train_filename, param_filename):
    # Loading data frame
#     df = pd.read_csv(train_filename, delimiter=',', na_values="?")

    # Separating target from inputs
    X, y = design_matrix(train_filename=train_filename)

    # Initializing DT classifier
    clf = tree.DecisionTreeClassifier()

    # Fitting DT classifier
    clf.fit(X, y)

    # Pickle and save
    f = open(param_filename, 'wb')
    pickle.dump(clf, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./data/training-data-completed.csv"], help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./data/dtree-params"], help="Output file to save the parameters of the decision tree")
    args = parser.parse_args() 
    train(args.train[0], args.param[0])