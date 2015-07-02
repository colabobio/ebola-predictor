"""
This script rebuilds the indices for training/testing sets, so it is possible to show the 
rows in the original data that were used in the sets.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, csv
import pandas as pd

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"

def load_data(fn, vars):
    vidx = []
    data = []
    with open(fn, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        vidx = [titles.index(var) for var in vars]
        for row in reader:
            data.append(row)
    return vidx, data

def save_indices(fn, indices):
    with open(fn, "wb") as ofile:
        writer = csv.writer(ofile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in indices:
            writer.writerow(row)

def rebuild_indices(data_files):
    for dfile in data_files:
        if "completed" in dfile: continue
        print "Rebuilding indices for",dfile
        idx, data = load_data(dfile, var_names)
        indices = []
        for row in data:
            pos = 0
            prow = None
            for row0 in data0:
                found = True
                for i in range(0, len(var_names)):
                    val = row[idx[i]].replace('?', '\\N')
                    val0 = row0[idx0[i]]
                    if not val == val0:
                        found = False
                        break
                if found:
                    prow = row0
                    break
                pos += 1
            if pos == len(data0):
                print "  Warning: row",row,"not found in source data!"
                indices.append(["?", "?", "?"])
            else:
                indices.append([pos, prow[0], prow[idx0[0]]])
        [dir, name] = os.path.split(dfile)
        name = name.replace("data", "index", 1)
        ifile = os.path.join(dir, name)
        save_indices(ifile, indices)

test_files = glob.glob("./data/testing-data*.csv")
train_files = glob.glob("./data/training-data*.csv")

data_file = ""
with open(src_file, "rb") as sfile:
    for line in sfile.readlines():
        data_file = os.path.abspath(line.strip())

var_names = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        line = line.strip()
        if not line: continue
        var_names.append(line.split()[0])

idx0, data0 = load_data(data_file, var_names)

rebuild_indices(test_files)
rebuild_indices(train_files)