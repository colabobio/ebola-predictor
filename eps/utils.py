"""
Utility functions for the Ebola Prognosis Score.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv
import numpy as np
import pandas as pd

"""
Loads a datafile a constructs a dataframe for EPS where there are only two columns: output 
and score.
"""
def load_dataframe(data_filename, threshold_filename="./eps/thresholds.txt"):
    thresh_variables = []
    model_variables = []
    with open(threshold_filename, "rb") as tfile:
        for line in tfile.readlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            thresh_variables.append({"name":parts[0], "type":parts[1], "threshold":parts[2]})
            model_variables.append(parts[0])
    data = []
    index = []
    columns = ["OUT", "SCORE"]
    with open(data_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        model_idx = [model_variables.index(var) for var in titles[1:]]
        count = 0
        for row in reader:
            out = row[0]
            score = 0
            for i in range(1, len(row)):
                var = thresh_variables[i - 1]
                if var["type"] == "category": 
                    score = score + (1 if row[i] == var["threshold"] else 0) 
                else:
                    str = var["threshold"]
                    if str[0] == "<":
                        str = str[1:]
                        score = score + (1 if float(row[i]) <= float(str) else 0) 
                    else:
                        score = score + (1 if float(row[i]) >= float(str) else 0)
            index.append(count)
            data.append([float(out), float(score)])
            count = count + 1
    df = pd.DataFrame(data, index=index, columns=columns)
    return df

def design_matrix(df):
    M = df.shape[0]
    N = df.shape[1]
    y = df.values[:,0]
    X = np.ones((M, N))
    for j in range(1, N):
        X[:, j] = df.values[:, j]
    return X, y

def gen_predictor(cutoff=0):
    def predictor(X):
        scores = []
        for i in range(0, len(X)):
            scores.extend([0 if x <= cutoff else 1 for x in X[i,1:]])
        return scores
    return predictor
