"""
Tests for Missing Completely At Random (MCAR) using LittleMCAR function in BaylorEdPsych for R:

http://cran.r-project.org/web/packages/BaylorEdPsych/

or the Test Homoscedasticity, Multivariate Normality, and Missing Completely at Random from
Jamshidian and Jalal.

http://cran.r-project.org/web/packages/MissMech/

References:

Little, R (1988). A test of missing completely at random for multivariate data with missing
values. Journal of the American Statistical Association, 83 (404), 1198-1202

Jamshidian, M. Jalal, S., and Jansen, C. (2014). "MissMech: An R Package for Testing 
Homoscedasticity, Multivariate Normality, and Missing Completely at Random (MCAR)," Journal 
of Statistical Software, 56(6), 1-31

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys, csv, os
import rpy2.robjects as robjects

src_file = "./data/sources.txt"
range_file = "./data/ranges.txt"
ignore_file = "./data/ignore.txt"

def run_test(var_file, mcar_test, pvalue_threshold):
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
                all_data.append([row[idx].replace("\\N", "?") for idx in model_idx])
                r += 1

    test_filename = "./mcar_test.csv"
    with open(test_filename, "w") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(model_variables)
        for row in all_data:
            writer.writerow(row)
    robjects.r('dat <- read.table("' + test_filename + '", sep=",", header=TRUE, na.strings="?")')

    if mcar_test == "little":
        robjects.r('library(BaylorEdPsych)')
        robjects.r('res <- LittleMCAR(dat)')
        print ""
        res = robjects.r['res']
        chisq = res[0][0]
        pvalue = res[2][0]
        print "Value of Little'schi-squared statistic:", chisq
        print "P-value of Little's chi-squared test  :", pvalue        
#        print res
#        print ""
#        print "Missingness patterns *****************************************************"
#        print ""
#        i = 0 
#        for patrn in res[5]:
#            print "Pattern",i,"----------------------------------------------------------"
#            i += 1
#            print patrn
#            print "IDs"
#            for rn in patrn.rownames:
#                print idx_info[int(rn) - 1][1]
#            print ""        
    elif mcar_test == "hawkins":
        robjects.r('library(MissMech)')
        robjects.r('res <- TestMCARNormality(dat, alpha = ' + str(pvalue_threshold) + ')')
        res = robjects.r['res']
        print res

    os.remove(test_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var_file", nargs=1, default=["./data/variables-master.txt"],
                        help="File containing list of variables to consider in the test")
    parser.add_argument("-p", "--pvalue", type=float, nargs=1, default=[0.05],
                        help="P-value for the chi-squared statistic")
    parser.add_argument("-t", "--test", nargs=1, default=["little"],
                        help="Test type: little for Little's MCAR chi-square test, hawkins for Hawkins test of normality and homoscedasticity")
    args = parser.parse_args()
    
    run_test(args.var_file[0], args.test[0], args.pvalue[0])
