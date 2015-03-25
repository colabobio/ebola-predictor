"""
Trains the Logistic Regression classifier from scikit-learn:
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse
import pandas as pd
import pickle
from sklearn import linear_model
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix

def prefix():
    return "lreg"

def title():
    return "Logistic Regression"

"""
Trains the logistic regression classifier given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting parameters
: param kwparams: custom arguments for decision tree. Same as listed in
                  http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
                  with the exception of random_state (not supported)
"""
def train(train_filename, param_filename, **kwparams):
    if "penalty" in kwparams:
        penalty = kwparams["penalty"]
    else:
        penalty = "l2"

    if "dual" in kwparams:
        dual =  kwparams["dual"].upper() in ['true', '1', 't', 'y']
    else:
        dual = False

    if "C" in kwparams:
        C = float(kwparams["C"])
    else:
        C = 1.0

    if "fit_intercept" in kwparams:
        fit_intercept =  kwparams["fit_intercept"].upper() in ['true', '1', 't', 'y']
    else:
        fit_intercept = True

    if "intercept_scaling" in kwparams:
        intercept_scaling = float(kwparams["intercept_scaling"])
    else:
        intercept_scaling = 1.0

    if "class_weight" in kwparams:
        class_weight = kwparams["class_weight"]
    else:
        class_weight = None

    if "random_state" in kwparams:
        random_state = int(kwparams["random_state"])
    else:
        random_state = None

    if "tol" in kwparams:
        tol = float(kwparams["tol"])
    else:
        tol = 0.0001

    # Separating target from inputs
    X, y = design_matrix(train_filename=train_filename)

    print "Training Logistic Regression Classifier..."

    # Initializing LR classifier
    clf = linear_model.LogisticRegression(penalty=penalty, dual=dual, C=C,
                                          fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
                                          class_weight=class_weight, random_state=random_state,
                                          tol=tol)

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
    parser.add_argument("-p", "--param", nargs=1, default=["./data/lreg-params"], 
                        help="Output file to save the parameters of the logistic regression classifier")
    parser.add_argument("-y", "--penalty", nargs=1, default=["l2"],
                        help="Used to specify the norm used in the penalization")
    parser.add_argument("-d", "--dual", nargs=1, default=["False"],
                        help="Dual or primal formulation")
    parser.add_argument("-c", "--inv_reg", nargs=1, type=float, default=[1.0],
                        help="Inverse of regularization strength; must be a positive floa")
    parser.add_argument("-f", "--fit_intercept", nargs=1, default=["True"],
                        help="Specifies if a constant should be added the decision function")
    parser.add_argument("-s", "--intercept_scaling", nargs=1, type=float, default=[1.0],
                        help="when fit_intercept is True, instance vector x becomes [x, self.intercept_scaling]")
    parser.add_argument("-w", "--class_weight", nargs=1, default=[None],
                        help="Over-/undersamples the samples of each class according to the given weights")
    parser.add_argument("-r", "--random_state", nargs=1, type=int, default=[None],
                        help="The seed of the pseudo random number generator to use when shuffling the data")
    parser.add_argument("-l", "--tol", nargs=1, type=float, default=[0.0001],
                        help="The seed of the pseudo random number generator to use when shuffling the data")

    args = parser.parse_args()
    train(args.train[0], args.param[0],
          penalty=args.penalty[0],
          dual=args.dual[0],
          C=args.inv_reg[0],
          fit_intercept=args.fit_intercept[0],
          intercept_scaling=args.intercept_scaling[0],
          class_weight=args.class_weight[0],
          random_state=args.random_state[0],
          tol=args.tol[0])
