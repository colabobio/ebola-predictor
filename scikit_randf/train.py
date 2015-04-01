"""
Trains the Random Forest classifier from scikit-learn:
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse
import pandas as pd
import pickle
from sklearn import ensemble
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix

def prefix():
    return "scikit_randf"

def title():
    return "Random Forest from scikit-learn"

"""
Trains the Random Forest classifier given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting parameters
: param kwparams: custom arguments for random forest. Same as listed in
                  http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
def train(train_filename, param_filename, **kwparams):
    if "n_estimators" in kwparams:
        n_estimators = int(kwparams["n_estimators"])
    else:
        n_estimators = 10
        
    if "criterion" in kwparams:
        criterion = kwparams["criterion"]
    else:
        criterion = "gini"

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

    if "min_weight_fraction_leaf" in kwparams:
        min_weight_fraction_leaf = float(kwparams["min_weight_fraction_leaf"])
    else:
        min_weight_fraction_leaf = 0

    max_leaf_nodes = None
    if "max_leaf_nodes" in kwparams:
        temp = kwparams["max_leaf_nodes"]
        if temp: max_leaf_nodes = int(temp)

    if "bootstrap" in kwparams:
        bootstrap =  kwparams["bootstrap"].upper() in ['true', '1', 't', 'y']
    else:
        bootstrap = True

    if "oob_score" in kwparams:
        oob_score =  kwparams["oob_score"].upper() in ['true', '1', 't', 'y']
    else:
        oob_score = False

    if "n_jobs" in kwparams:
        n_jobs = int(kwparams["n_jobs"])
    else:
        n_jobs = 1

    if "random_state" in kwparams:
        random_state = int(kwparams["random_state"])
    else:
        random_state = None

    if "class_weight" in kwparams:
        class_weight = kwparams["class_weight"]
    else:
        class_weight = None

    # Separating target from inputs
    X, y = design_matrix(train_filename=train_filename)

    print "Training Random Forest Classifier..."
    clf = ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                          max_features=max_features, max_depth=max_depth,
                                          min_samples_split=min_samples_split,
                                          min_samples_leaf=min_samples_leaf,
#                                           min_weight_fraction_leaf=min_weight_fraction_leaf,
                                          max_leaf_nodes=max_leaf_nodes, bootstrap=bootstrap,
                                          oob_score=oob_score, n_jobs=n_jobs,
                                          random_state=random_state)
#                                           class_weight=class_weight)

    # Fitting LR classifier
    clf.fit(X, y)

    # Pickle and save
    f = open(param_filename, 'wb')
    pickle.dump(clf, f)

    print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", nargs=1, default=["./data/training-data-completed.csv"],
                        help="File containing training set")
    parser.add_argument("-p", "--param", nargs=1, default=["./data/scikit_randf-params"],
                        help="Output file to save the parameters of the SVM classifier")
    parser.add_argument("-n", "--n_estimators", nargs=1, type=int, default=[10],
                        help="The number of trees in the forest")
    parser.add_argument("-c", "--criterion", nargs=1, default=["gini"],
                        help="The function to measure the quality of a split")
    parser.add_argument("-maxf", "--max_features", nargs=1, default=[None],
                        help="The number of features to consider when looking for the best split")
    parser.add_argument("-maxd", "--max_depth", nargs=1, type=int, default=[None],
                        help="The maximum depth of the tree")
    parser.add_argument("-mins", "--min_samples_split", nargs=1, type=int, default=[2],
                        help="The minimum number of samples required to split an internal node")
    parser.add_argument("-minl", "--min_samples_leaf", nargs=1, type=int, default=[1],
                        help="The minimum number of samples required to be at a leaf node")
    parser.add_argument("-minw", "--min_weight_fraction_leaf", nargs=1, type=float, default=[0],
                        help="The minimum weighted fraction of the input samples required to be at a leaf node")
    parser.add_argument("-maxl", "--max_leaf_nodes", nargs=1, type=int, default=[None],
                        help="Grow a tree with max_leaf_nodes in best-first fashion")
    parser.add_argument("-b", "--bootstrap", nargs=1, default=["True"],
                        help="Whether bootstrap samples are used when building trees")
    parser.add_argument("-oob", "--oob_score", nargs=1, default=["False"],
                        help="Whether to use out-of-bag samples to estimate the generalization error")
    parser.add_argument("-j", "--n_jobs", nargs=1, type=int, default=[1],
                        help="The number of jobs to run in parallel for both fit and predict")
    parser.add_argument("-r", "--random_state", nargs=1, type=int, default=[None],
                        help="The seed of the pseudo random number generator to use when shuffling the data for probability estimation")
    parser.add_argument("-w", "--class_weight", nargs=1, default=[None],
                        help="Weights associated with classes in the form {class_label: weight}")

    args = parser.parse_args()
    train(args.train[0], args.param[0],
          n_estimators=args.n_estimators[0],
          criterion=args.criterion[0],
          max_features=args.max_features[0],
          max_depth=args.max_depth[0],
          min_samples_split=args.min_samples_split[0],
          min_samples_leaf=args.min_samples_leaf[0],
          min_weight_fraction_leaf=args.min_weight_fraction_leaf[0],
          max_leaf_nodes=args.max_leaf_nodes[0],
          bootstrap=args.bootstrap[0],
          oob_score=args.oob_score[0],
          n_jobs=args.n_jobs[0],
          random_state=args.random_state[0],
          class_weight=args.class_weight[0])
