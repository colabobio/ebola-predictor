"""
Transforms a training/testing pair applying the PCA transformation as derived from all
complete data, using the specified number of PCA components.

Careful: original files are overwritten!!

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys, csv, os, random
import numpy as np
from sklearn.decomposition import PCA

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"
range_file = "./data/ranges.txt"

def do_pca(test_filename, train_filename, num_comp):
    input_file = ""
    with open(src_file, "rb") as sfile:
        for line in sfile.readlines():
            input_file = os.path.abspath(line.strip())

    model_variables = []
    with open(var_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            model_variables.append(line.split()[0])

    range_variables = [] 
    with open(range_file, "rb") as rfile:
        for line in rfile.readlines():
            line = line.strip()
            if not line: continue
            parts = line.strip().split()
            if 2 < len(parts):
                range_variables.append({"name":parts[0], "type":parts[1], "range":parts[2].split(",")})

    ids = []
    all_data = []
    idx_info = []
    complete_rows = []
    with open(input_file, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        model_idx = [titles.index(var) for var in model_variables]
        r0 = 0
        r = 0
        for row in reader:
            r0 += 1 # Starts at 1, because of titles
            all_missing = True
            some_missing = False
            missing_dvar = row[model_idx[0]] == "\\N"
            for i in range(1, len(model_variables)):
                var_idx = model_idx[i]
                if row[var_idx] == "\\N":
                    some_missing = True
                else:
                    all_missing = False

            inside_range = True
            for var in range_variables:
                idx = titles.index(var["name"])
                val = row[idx]
                if val == "\\N": continue
                vtype = var["type"]
                vrang = var["range"]
                test = True
                if vtype == "category":
                    test = val in vrang
                else:
                    test = float(vrang[0]) <= float(val) and float(val) < float(vrang[1])
                inside_range = inside_range and test

            if not all_missing and not missing_dvar and inside_range:
                ids.append(row[0])
                idx_info.append([r0, row[0], row[model_idx[0]]])
                all_data.append([row[idx].replace("\\N", "?") for idx in model_idx])
                if not some_missing: complete_rows.append(r)
                r += 1

    ######################################################################################
    # PCA TEST
    dat = []
    for r in range(0, len(all_data)):
        if r in complete_rows:
            dat.append([float(x) for x in all_data[r][1:]])
    X = np.array(dat)
    pca = PCA(n_components=num_comp)
#     print X[0]
    pca.fit(X)
#     print(pca.explained_variance_ratio_)
#     print pca.transform(X)

    title = [model_variables[0]]
    for i in range(0, num_comp):
        title.append("PC" + str(i + 1))
    ttest_data = [title]
    with open(test_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        reader.next()
        for row in reader:
            x0 = [float(v) for v in row[1:]]
            tx = pca.transform(x0)[0].tolist()
            x1 = [row[0]]
            x1.extend(tx)
            ttest_data.append(x1)
#     print ttest_data


    ttrain_data = [title]
    with open(train_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        reader.next()
        for row in reader:
            x0 = [float(v) for v in row[1:]]
            tx = pca.transform(x0)[0].tolist()
            x1 = [row[0]]
            x1.extend(tx)
            ttrain_data.append(x1)
#     print ttrain_data

    print "PCA transform",test_filename
    with open(test_filename, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in ttest_data:
            writer.writerow(row)

    print "PCA transform",train_filename
    with open(train_filename, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in ttrain_data:
            writer.writerow(row)

    ######################################################################################

if __name__ == "__main__":
    do_pca("data/testing-data-0.csv", "data/training-data-completed-0.csv", 4)
