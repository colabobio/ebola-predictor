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
    return "randf"

def title():
    return "Random Forest"

"""
Trains the Random Forest classifier given the specified parameters

: param train_filename: name of file containing training set
: param param_filename: name of file to store resulting parameters
: param kwparams: custom arguments for random forest. Same as listed in
                  http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""
def train(train_filename, param_filename, **kwparams):
    if "error" in kwparams:
        C = float(kwparams["error"])
    else:
        C = 1.0
        
    if "kernel" in kwparams:
        kernel = kwparams["kernel"]
    else:
        kernel = "rbf"

    if "degree" in kwparams:
        degree = int(kwparams["degree"])
    else:
        degree = 3

    if "gamma" in kwparams:
        gamma = float(kwparams["gamma"])
    else:
        gamma = 0.0

    if "coef0" in kwparams:
        coef0 = float(kwparams["coef0"])
    else:
        coef0 = 0.0

    if "shrinking" in kwparams:
        shrinking =  kwparams["shrinking"].upper() in ['true', '1', 't', 'y']
    else:
        shrinking = True
        
    if "tol" in kwparams:
        tol = float(kwparams["tol"])
    else:
        tol = 0.0001

    if "cache_size" in kwparams:
        cache_size = float(kwparams["cache_size"])
    else:
        cache_size = 200

    if "class_weight" in kwparams:
        class_weight = kwparams["class_weight"]
    else:
        class_weight = None

    if "max_iter" in kwparams:
        max_iter = int(kwparams["max_iter"])
    else:
        max_iter = -1

    if "random_state" in kwparams:
        random_state = int(kwparams["random_state"])
    else:
        random_state = None

    # Separating target from inputs
    X, y = design_matrix(train_filename=train_filename)

    print "Training Random Forest Classifier..."

    # Initializing SVM classifier
#     clf = svm.SVC(probability=True,
#                   C=C, kernel=kernel, degree=degree, gamma=gamma,
#                   coef0=coef0, shrinking=shrinking, tol=tol, cache_size=cache_size,
#                   class_weight=class_weight, max_iter=max_iter, random_state=random_state)




    clf = ensemble.RandomForestClassifier(criterion='entropy',min_samples_leaf=10, max_depth=5)

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
    parser.add_argument("-p", "--param", nargs=1, default=["./data/randf-params"], 
                        help="Output file to save the parameters of the SVM classifier")
    parser.add_argument("-c", "--error", nargs=1, type=float, default=[1.0],
                        help="Penalty parameter C of the error term")
    parser.add_argument("-k", "--kernel", nargs=1, default=["rbf"],
                        help="Specifies the kernel type to be used in the algorithm")
    parser.add_argument("-d", "--degree", nargs=1, type=int, default=[3],
                        help="Degree of the polynomial kernel function")                        
    parser.add_argument("-g", "--gamma", nargs=1, type=float, default=[0.0],
                        help="Kernel coefficient for rbf, poly and sigmoid")
    parser.add_argument("-0", "--coef0", nargs=1, type=float, default=[0.0],
                        help="Independent term in kernel function")
    parser.add_argument("-s", "--shrinking", nargs=1, default=["True"],
                        help="Dual or primal formulation")
    parser.add_argument("-l", "--tol", nargs=1, type=float, default=[0.0001],
                        help="Tolerance for stopping criterion")
    parser.add_argument("-h", "--cache_size", nargs=1, type=float, default=[200],
                        help="Specify the size of the kernel cache (in MB)")
    parser.add_argument("-w", "--class_weight", nargs=1, default=[None],
                        help="Set the parameter C of class i to class_weight[i]*C for SVC")
    parser.add_argument("-x", "--max_iter", nargs=1, type=int, default=[-1],
                        help="Hard limit on iterations within solver, or -1 for no limit") 
    parser.add_argument("-r", "--random_state", nargs=1, type=int, default=[None],
                        help="The seed of the pseudo random number generator to use when shuffling the data for probability estimation")

    args = parser.parse_args()
    train(args.train[0], args.param[0],
          C=args.error[0],
          kernel=args.kernel[0],
          degree=args.degree[0],
          gamma=args.gamma[0],
          coef0=args.coef0[0],
          shrinking=args.shrinking[0],
          tol=args.tol[0],
          cache_size=args.cache_size[0],
          class_weight=args.class_weight[0],
          max_iter=args.max_iter[0],
          random_state=args.random_state[0])
