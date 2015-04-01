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
    return "scikit_dtree"

def title():
    return "Decision Tree from scikit-learn"

"""
Trains the decision tree given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting decision tree parameters
: param kwparams: custom arguments for decision tree. Same as listed in
                  http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
                  with the exception of random_state (not supported)
"""
def train(train_filename, param_filename, **kwparams):
    if "criterion" in kwparams:
        criterion = kwparams["criterion"]
    else:
        criterion = "gini"

    if "splitter" in kwparams:
        splitter = kwparams["splitter"]
    else:
        splitter = "best"

    max_features = None
    if "max_features" in kwparams:
        temp = kwparams["max_features"]
        if temp in ["auto", "sqrt", "log2"]:
            max_features = temp
        elif temp:
            try:
                max_features = int(temp)
            except ValueError:
                try:
                    max_features = float(temp)
                except ValueError:
                    pass

    max_depth = None
    if "max_depth" in kwparams:
        temp = kwparams["max_depth"]
        if temp: max_depth = int(temp)

    if "min_samples_split" in kwparams:
        min_samples_split = int(kwparams["min_samples_split"])
    else:
        min_samples_split = 2

    if "min_samples_leaf" in kwparams:
        min_samples_leaf = int(kwparams["min_samples_leaf"])
    else:
        min_samples_leaf = 1

    max_leaf_nodes = None
    if "max_leaf_nodes" in kwparams:
        temp = kwparams["max_leaf_nodes"]
        if temp: max_leaf_nodes = int(temp)

    # Separating target from inputs
    X, y = design_matrix(train_filename=train_filename)

    print "Training Decision Tree..."

    # Initializing DT classifier
    clf = tree.DecisionTreeClassifier(criterion=criterion, splitter=splitter,
                                      max_features=max_features,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      max_leaf_nodes=max_leaf_nodes)

    # Fitting DT classifier
    clf.fit(X, y)

    # Pickle and save
    f = open(param_filename, 'wb')
    pickle.dump(clf, f)

    print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./data/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./data/scikit_dtree-params"], 
                        help="Output file to save the parameters of the decision tree")
    parser.add_argument("-c", "--criterion", nargs=1, default=["gini"],
                        help="The function to measure the quality of a split")
    parser.add_argument("-s", "--splitter", nargs=1, default=["best"],
                        help="The strategy used to choose the split at each node")
    parser.add_argument("-maxf", "--max_features", nargs=1, default=[None],
                        help="The number of features to consider when looking for the best split")
    parser.add_argument("-maxd", "--max_depth", nargs=1, type=int, default=[None],
                        help="The maximum depth of the tree")
    parser.add_argument("-mins", "--min_samples_split", nargs=1, type=int, default=[2],
                        help="The minimum number of samples required to split an internal node")
    parser.add_argument("-minl", "--min_samples_leaf", nargs=1, type=int, default=[1],
                        help="The minimum number of samples required to be at a leaf node")
    parser.add_argument("-maxl", "--max_leaf_nodes", nargs=1, type=int, default=[None],
                        help="Grow a tree with max_leaf_nodes in best-first fashion")
    args = parser.parse_args()
    train(args.train[0], args.param[0],
          criterion=args.criterion[0],
          splitter=args.splitter[0],
          max_features=args.max_features[0],
          max_depth=args.max_depth[0],
          min_samples_split=args.min_samples_split[0],
          min_samples_leaf=args.min_samples_leaf[0],
          max_leaf_nodes=args.max_leaf_nodes[0])
