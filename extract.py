"""
Extracts models for the app.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os
import pandas as pd
import numpy as np
import shutil

ranking_files = ["./temp/pcr/models/ranking.txt",
                 "./temp/nopcr/models/ranking.txt"]
models_dir = "./evdcare/models"

count = 0
for fn in ranking_files:
    with open(fn, "r") as rfile:
        base_dir = os.path.split(fn)[0]
        lines = rfile.readlines()
        for line in lines:
            line = line.strip()
            parts = line.split(" ")

            pred = parts[2]
            mdl_str = parts[1]

            f1_mean = float(parts[4])
            f1_std = float(parts[5])

            if 0.9 <= f1_mean and 0.05 <= f1_std and pred == "nnet":
                id = os.path.split(mdl_str)[1]
                var_file = os.path.join(base_dir,id, "variables.txt")
                par_file = os.path.join(base_dir,id, "nnet-params-0")
                train_file = os.path.join(base_dir,id, "training-data-completed-0.csv")
                if os.path.exists(var_file) and os.path.exists(par_file) and os.path.exists(train_file):
                     print "Extracting model from",base_dir
                     count += 1
                     dir = os.path.join(models_dir, str(count))
                     if not os.path.exists(dir):
                         os.mkdir(dir)
                     shutil.copy(var_file, os.path.join(dir, "variables.txt"))
                     
                     shutil.copy(par_file, os.path.join(dir, "nnet-params"))
                     df = pd.read_csv(train_file, delimiter=",", na_values="?")
                     M = df.shape[0]
                     N = df.shape[1]
                     names = list(df.columns)
                     y = df.values[:,0]
                     with open(os.path.join(dir, "bounds.txt"), "w") as bfile:
                         for j in range(1, N):
                             values = df.values[:, j]
                             minv = values.min()
                             maxv = values.max()
                             bfile.write(names[j] + " " + str(minv) + " " + str(maxv) + "\n")
