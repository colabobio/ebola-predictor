"""
Utility functions for the Ebola Prognosis Score.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv
import numpy as np
import pandas as pd

thr_file = "./eps/thresholds.txt"
threshold_names = []
threshold_infos = []
with open(thr_file, "rb") as tfile:
    for line in tfile.readlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        threshold_names.append(parts[0])
        threshold_infos.append({"type":parts[1], "threshold":parts[2]})

def var_score(name, value):
    if not name in threshold_names: return 0
    idx = threshold_names.index(name)
    info = threshold_infos[idx]
    str = info["threshold"]
    if info["type"] == "category":
        return 1 if int(float(value)) == int(str) else 0
    else:
        if str[0] == "<":
            str = str[1:]
            return 1 if float(value) <= float(str) else 0
        else:
            return 1 if float(value) >= float(str) else 0

"""
Loads a datafile a constructs a dataframe for EPS where there are only two columns: output 
and score.
"""
def load_dataframe(data_filename):
    data = []
    index = []
    columns = ["OUT", "SCORE"]
    with open(data_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        count = 0
        for row in reader:
            out = row[0]
            score = 0
            for i in range(1, len(row)):
                score = score + var_score(titles[i], row[i])
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
