"""
This script creates the training and test sets.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys, csv, os, random
import numpy as np

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"
range_file = "./data/ranges.txt"

"""Returns a set of indices for the test set, making sure that both training and test 
set will have same fraction of outcomes

:param all_data: all rows in the dataset
:param complete_rows: indices of rows w/out missing values in all_data
:param test_percentage: percentage of complete rows that will be used in the test set
"""
def test_set(all_data, complete_rows, test_percentage):
    outlist = []
    for i in complete_rows:
        row = all_data[i]
        outlist.append(int(row[0]))
    out = np.array(outlist)
    i0 = np.where(out == 0)
    i1 = np.where(out == 1)
    f = test_percentage / 100.0
    ri0 = np.random.choice(i0[0], size=f*i0[0].shape[0], replace=False)
    ri1 = np.random.choice(i1[0], size=f*i1[0].shape[0], replace=False)
    itest = np.concatenate((ri1, ri0))
    itest.sort()

    # itest contains the indices for the complete_rows list, which in turn is the list of
    # indices in the original data, so we:
    return np.array(complete_rows)[itest]

"""Creates a training/test sets and saves them to the specified files. The test set won't
have any missing values, and will include the given percentage of complete rows from the
source data

:param test_percentage: percentage of complete rows that will be used in the test set
:param test_filename: name of file to store test set
:param train_filename: name of file to store training set
"""
def makesets(test_percentage, test_filename, train_filename, index_filename):
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

    test_idx = test_set(all_data, complete_rows, test_percentage)
    training_data = []
    testing_data = []
    for r in range(0, len(all_data)):
        row = all_data[r]
        if r in test_idx:
            testing_data.append(row)
        else:
            training_data.append(row)

    # Saving index information
    with open(test_filename.replace("-data", "-index"), "wb") as idxfile:
        writer = csv.writer(idxfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for r in range(0, len(all_data)):
            if r in test_idx: writer.writerow(idx_info[r])
    with open(train_filename.replace("-data", "-index"), "wb") as idxfile:
        writer = csv.writer(idxfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for r in range(0, len(all_data)):
            if not r in test_idx: writer.writerow(idx_info[r])
        
    with open(train_filename, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in training_data:
            writer.writerow(row)
    print "Wrote", len(training_data), "rows to training set in", train_filename

    with open(test_filename, "wb") as tefile:
        writer = csv.writer(tefile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in testing_data:
            writer.writerow(row)
    print "Wrote", len(testing_data), "rows to training set in", test_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', nargs=1, default=["./data/training-data.csv"],
                        help="Filename for training set")
    parser.add_argument('-T', '--test', nargs=1, default=["./data/testing-data.csv"],
                        help="Filename for test set")
    parser.add_argument('-i', '--index', nargs=1, default=["./data/idx"],
                        help="Filename to store original indices")
    parser.add_argument('-p', '--percentage', type=int, nargs=1, default=[50],
                        help="Percentage of complete data to use in test set")
    args = parser.parse_args()
    makesets(args.percentage[0], args.test[0], args.train[0], args.index[0])