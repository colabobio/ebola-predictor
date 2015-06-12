"""
Basic Statistics of numerical variables:

Fisher exact test
http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html

T-test for the mean of two samples 
http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.ttest_ind.html

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse, csv
import numpy as np
import pandas as pd
from pandas.tools.pivot import pivot_table
from scipy.stats import ttest_ind
from scipy.stats import fisher_exact

src_file = "./data/sources.txt"
var_file = "./data/variables.txt"
range_file = "./data/ranges.txt"
ignore_file = "./data/ignore.txt"

def runtests(var_file):
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

    all_data = {}
    for var in model_variables:
        all_data[var] = []
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
#                 ids.append(row[0])
#                 idx_info.append([r0, row[0], row[model_idx[0]]])
#                 all_data.append([row[idx] for idx in model_idx])
                for i in range(0, len(model_variables)):
                    var = model_variables[i]
                    idx = model_idx[i]
                    try:
                        val = float(row[idx])
                    except ValueError:
                        val = np.NaN    
                    all_data[var].append(val) 
                
                
#                 if not some_missing: complete_rows.append(r)
                r += 1

  
    data = pd.DataFrame(all_data)
#     data = pd.read_csv(input_file, delimiter=",", na_values="\\N")

    
#     print pd.Series(all_data)
#     print all_data["WEAK"]
#     print pd.DataFrame(all_data)
#     exit(1)
#     print data["WEAK"]
#     exit(1)

    cat_vars = []
    num_vars = []
    count = 0
    with open(var_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue   
            parts = line.split()  
            if count == 0: 
                dvar = parts[0]
                count = 1
                continue
            count += 1
            if parts[1] == "category":
               cat_vars.append(parts[0])
            else:
                num_vars.append(parts[0])

    for var in cat_vars:
        print "***********************"      
        print var
        dat = data.loc[:,(var, dvar)]
        dat["VALUES"] = pd.Series(np.ones(len(dat[var])), index=dat.index)
        # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html
        counts = pivot_table(dat, values="VALUES", index=[var], columns=[dvar], aggfunc=np.sum, fill_value=0)
        fisher_ratio, fisher_pvalue = fisher_exact(counts)
        print counts
        print fisher_ratio, fisher_pvalue
     
    for var in num_vars:
        values0 = data[data[dvar] == 0][var]
        values1 = data[data[dvar] == 1][var]
    
        values0 = values0[~np.isnan(values0)]
        values1 = values1[~np.isnan(values1)]
        
        print "***********************" 
        print var
        print "mean/std for",dvar,"0:",np.mean(values0), "/", np.std(values0)
        print "mean/std for",dvar,"1:",np.mean(values1), "/", np.std(values1)
        ttest_stat, ttest_pvalue = ttest_ind(values0, values1, equal_var=False)
        print "Means are different at p-value",ttest_pvalue




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--var_file", nargs=1, default=["./data/variables.txt"],
                        help="File with list of variables")
    args = parser.parse_args()
    runtests(args.var_file[0])
