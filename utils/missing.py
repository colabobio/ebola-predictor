"""
This script gives information on the missingness in the input file.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, csv, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--show_complete", action="store_true",
                    help="show complete data")
args = parser.parse_args()

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"
range_file = "./data/ranges.txt"

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

total_count = 0
miss_dep_var = 0
complete_count = 0
missing_count = [0] * len(model_variables)

complete_data = []
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
            total_count = total_count + 1
            if not some_missing:
                complete_row = []
                for i in range(0, len(model_variables)):
                    var_idx = model_idx[i]
                    complete_row.append(row[var_idx])
                complete_data.append(complete_row)
                complete_count = complete_count + 1
            for i in range(0, len(model_variables)):
                var_idx = model_idx[i]
                if row[var_idx] == "\\N":
                    missing_count[i] = missing_count[i] + 1

print "Total number of data rows   :",total_count
print "Number of complete data rows:",complete_count

print "Missing counts and precentages for each independent variable:"
for i in range(1, len(model_variables)):
    miss_frac = 100.0 * float(missing_count[i])/ total_count
    print "{:7s} {:2.0f}/{:2.0f} {:2.2f}%".format(model_variables[i], missing_count[i], total_count, miss_frac)

if args.show_complete:
    print "Complete entries (" + str(len(complete_data)) + ")"
    print model_variables
    for row in complete_data:
        print row