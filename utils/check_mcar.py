"""
Tests for Missing Completely At Random (MCAR) using LittleMCAR function in BaylorEdPsych for R:

http://cran.r-project.org/web/packages/BaylorEdPsych/

based on the multivariate test described in

Little, R (1988). A test of missing completely at random for multivariate data with missing
values. Journal of the American Statistical Association, 83 (404), 1198-1202

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, csv, os
import rpy2.robjects as robjects

src_file = "./data/sources.txt"
var_file = "./data/variables-master.txt"
range_file = "./data/ranges.txt"

def doit():
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

    idx_info = []
    all_data = []
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
                idx_info.append([r0, row[0], row[model_idx[0]]])
                all_data.append([row[idx].replace("\\N", "?") for idx in model_idx])
                r += 1

    dvar = model_variables[0]
    for i in range(1, len(model_variables)):
        ivar = model_variables[i]

        robjects.r('library(BaylorEdPsych)')
        with open("test.csv", "w") as trfile:
            writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([dvar, ivar])
            for row in all_data:
                writer.writerow([row[0], row[i]])
        robjects.r('trdat <- read.table("test.csv", sep=",", header=TRUE, na.strings="?")')
        print "Test for",dvar,ivar
        robjects.r('res <- LittleMCAR(trdat)')
        print ""
        res = robjects.r['res']
        print "Value of chi-squared statistic:", res[0][0]
        print "P-value:", res[2][0]

    print "Multivariate test ************************************************************"
    with open("test.csv", "w") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in all_data:
            writer.writerow(row)
    robjects.r('trdat <- read.table("test.csv", sep=",", header=TRUE, na.strings="?")')
    robjects.r('res <- LittleMCAR(trdat)')
    print ""
    res = robjects.r['res']
    print "Value of chi-squared statistic:", res[0][0]
    print "P-value:", res[2][0]
    for patrn in res[5]:
        print patrn
#         print patrn

#     print idx_info

if __name__ == "__main__":
    doit()

# in_filename = "./data/data.csv"
# 


