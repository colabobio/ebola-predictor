"""
This script creates the training and test sets.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, csv, random
import numpy as np

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"
range_file = "./data/ranges.txt"

# Returns a set of indices for the test set, making sure that both training and test 
# set will have same fraction of outcomes
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

def makesets(test_percentage, testing_file, training_file):
    input_file = ""
    with open(src_file, "rb") as sfile:
        for line in sfile.readlines():
            input_file = line.strip()

    model_variables = []
    with open(var_file, "rb") as vfile:
        for line in vfile.readlines():
            model_variables.append(line.split()[0])

    range_variables = [] 
    with open(range_file, "rb") as rfile:
        for line in rfile.readlines():
            parts = line.strip().split()
            range_variables.append({"name":parts[0], "type":parts[1], "range":parts[2].split(",")})

    ids = []
    all_data = []
    complete_rows = []
    with open(input_file, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        model_idx = [titles.index(var) for var in model_variables]
        r = 0 
        for row in reader:
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
                all_data.append([row[idx].replace("\\N", "?") for idx in model_idx])
                if not some_missing: complete_rows.append(r)
                r = r + 1

    test_idx = test_set(all_data, complete_rows, test_percentage)
    training_data = []
    testing_data = []
    for r in range(0, len(all_data)):
        row = all_data[r]
        if r in test_idx:
            testing_data.append(row)
        else:
            training_data.append(row)

    with open(training_file, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in training_data:
            writer.writerow(row)
    print "Wrote", len(training_data), "rows to training set in", training_file 

    with open(testing_file, "wb") as tefile:
        writer = csv.writer(tefile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in testing_data:
            writer.writerow(row)
    print "Wrote", len(testing_data), "rows to training set in", testing_file

if __name__ == "__main__":
    test_percentage = 50
    if 1 < len(sys.argv):
        test_percentage = float(sys.argv[1])
    training_file = "./data/training-data.csv"
    testing_file = "./data/testing-data.csv"
    makesets(test_percentage, training_file, testing_file)