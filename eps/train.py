"""
Trains the Ebola Prognostic Score (EPS) predictor.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv
import pandas as pd
import pickle
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from eps_utils import design_matrix, eps_dataframe

training_filename = "./data/training-data-imputed.csv"
model_file = "./data/eps-model.p"

def train(df):
    # Separating target from inputs
    X, y = design_matrix(df)

    # Initializing DT classifier
#     clf = tree.DecisionTreeClassifier()
    clf = linear_model.LogisticRegression()
#     clf = svm.SVC(probability=True)

    # Fitting DT classifier
    clf.fit(X, y)

    # Pickle and save
    f = open(model_file, 'wb')
    pickle.dump(clf, f)

if __name__ == "__main__":
    df = eps_dataframe(training_filename)
    train(df)
