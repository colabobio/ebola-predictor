"""
Uses MINE (Maximal Information-based Nonparametric Exploration) to rank variables:

http://www.exploredata.net/

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys, csv, os

src_file = "./data/sources.txt"
var_file = "./data/variables-master.txt"
range_file = "./data/ranges.txt"
ignore_file = "./data/ignore.txt"

def run_test(pvalue_threshold):
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

    ignore_records = []
    with open(ignore_file, "rb") as rfile:
        for line in rfile.readlines():
            line = line.strip()
            if not line: continue
            ignore_records.append(line)

    idx_info = []
    all_data = []
    with open(input_file, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        model_idx = [titles.index(var) for var in model_variables]
        r0 = 0
        r = 0
        for row in reader:
            if row[0] in ignore_records: continue

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
                idx_info.append([r0, row[0], row[model_idx[0]]])
                all_data.append([row[idx].replace("\\N", "") for idx in model_idx])
                r += 1

    test_filename = "./mine_test.csv"

    dvar = model_variables[0]
    with open(test_filename, "w") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in all_data:
            writer.writerow(row)

    os.system("java -jar utils/MINE.jar " + test_filename + " 0")
    os.remove(test_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pvalue", type=float, nargs=1, default=[0.05],
                        help="P-value for the chi-squared statistic")
    args = parser.parse_args()
    run_test(args.pvalue[0])
