"""
Rank all available models

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import os, csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

'''
src_file = "./data/sources.txt"
range_file = "./data/ranges.txt"
ignore_file = "./data/ignore.txt"

def load_data(var_file):
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

    dvar = model_variables[0]
    ivar = model_variables[1:]

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
                
    print all_data
'''

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--var_file", nargs=1, default=["./data/variables-master.txt"],
                    help="File containing list of variables to consider in the test")
args = parser.parse_args()

var_file = args.var_file[0]
# load_data(var_file)

depvar = None
numvar = []
catvar = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        if not depvar: depvar = parts[0]
        else:
            if parts[1] == "category": catvar.append(parts[0])
            else: numvar.append(parts[0])
#         
#         model_variables.append([0])

# dvar = model_variables[0]
# ivar = model_variables[1:]
# print dvar
# print ivar

src_file = "./data/sources.txt"
input_file = ""
with open(src_file, "rb") as sfile:
    for line in sfile.readlines():
        input_file = os.path.abspath(line.strip())
df = pd.read_csv(input_file, delimiter=",", na_values="\\N")

plt.clf()
fig = plt.figure()

# print df[dvar]
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

range_file = "./data/ranges.txt"
range_variables = [] 
with open(range_file, "rb") as rfile:
    for line in rfile.readlines():
       line = line.strip()
       if not line: continue
       parts = line.strip().split()
       if 2 < len(parts):
           range_variables.append({"name":parts[0], "type":parts[1], "range":parts[2].split(",")})
ranges = (df["DIAG"] == 1) & (10 <= df["AGE"]) & (df["AGE"] <= 50) & ((df["OUT"] == 1) | (df["OUT"] == 0))

# sns.set(style="whitegrid")

for var in numvar:
    #df[ranges].boxplot(var, by=depvar)
    g = sns.factorplot("OUT", var, data=df[ranges], kind="box", ci=0, palette="coolwarm")
    g.despine(offset=10, trim=True)
    g.set_xticklabels(['No', 'Yes'])
    plt.savefig("./out/" + var + ".pdf")
    plt.clf()

for var in catvar:
	with mpl.rc_context({"lines.linewidth": 0.5}):
# 		g = sns.barplot(df[ranges][var], df[ranges]["OUT"], palette="coolwarm")
		g = sns.FacetGrid(df[ranges], aspect=1)
		g.map(sns.barplot, "OUT", var, palette="coolwarm");
		g.set_xticklabels(['No', 'Yes'])
		g.set(ylim=(0.0, 1.0))
#     df[ranges].hist(var, by=depvar)
#     df[ranges].plot(kind='area', var, by=depvar)
# 
    	plt.savefig("./out/" + var + ".pdf")
    	plt.clf()


# df[ranges & (df["OUT"] == 0)]["ALT_1"].plot(kind='box', color=color, sym='r+')
# df[ranges & (df["OUT"] == 1)]["ALT_1"].plot(kind='box', color=color, sym='r+')

# 

# df[criterion]["ALT_1"].plot(kind='box', color=color, sym='r+')


# sns.factorplot("ALT_1", df, kind="bar");