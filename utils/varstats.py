"""
Basic Statistics of numerical variables.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

data = pd.read_csv("data/data.csv", delimiter=",", na_values="\\N")

var_file = "./data/variables.txt"
vars = []
count = 0
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        line = line.strip()
        if not line: continue   
        parts = line.split()  
        if count == 0: dvar = parts[0]
        count += 1        
        if parts[1] == "category": continue        
        vars.append(parts[0])

for var in vars:
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

# 500 >
# 180 >
# < 18

