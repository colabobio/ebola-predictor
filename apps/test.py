"""
Runs the app on all the data instances

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, operator
import pandas as pd


####################################################################################
# Read the variables

units = {}
variables = []
var_label = {}
var_kind = {}
var_def_unit = {}
var_alt_unit = {}
var_unit_conv = {}
with open("./ebolacare/variables.csv", "r") as vfile:
    lines = vfile.readlines()
    for line in lines[1:]:
        line = line.strip()
        row = line.split(",")
        name = row[0]
        label = row[1]
        kind = row[2] 
        def_unit = row[3]
        alt_unit = row[4]
        unit_conv = row[5]
 
        variables.append(name)
        var_label[name] = label
        var_kind[name] = kind
        var_def_unit[name] = def_unit
        var_alt_unit[name] = alt_unit
        if unit_conv:
            c, f = unit_conv.split("*")
            unit_cf = [float(c), float(f)]
        else:
            unit_cf = [0, 1]
        var_unit_conv[name] = unit_cf

        units[name] = def_unit
        print name, label, kind, def_unit, alt_unit, unit_cf

####################################################################################
# Read the predictive models

ranking = {}
dirs = glob.glob("./ebolacare/models/*")
for d in dirs:
    with open(os.path.join(d, "ranking.txt")) as rfile:
        line = rfile.readlines()[0].strip()
        f1score = float(line.split()[0])
        ranking[d] = f1score 

sorted_ranking = reversed(sorted(ranking.items(), key=operator.itemgetter(1)))
models_info = []
for pair in sorted_ranking:
    d = pair[0]
    v = []
    with open(os.path.join(d, "variables.txt")) as vfile:
        lines = vfile.readlines()
        for line in lines[1:]:
            line = line.strip()
            parts = line.split()
            v.append(parts[0])
    info = [d, v] 
    #print info
    models_info.append(info)

####################################################################################
# Read the data

data = pd.read_csv("../data/data.csv", delimiter=",", na_values="\\N")

for idx, row in data.iterrows():
    age = row['AGE']
    if age < 10 or 50 < age: continue
    print row['GID'],
    for var in variables:
        val = row[var]
        print var + ":" + str(val),
    print    

    


# parser = argparse.ArgumentParser()
# parser.add_argument("-d", "--dir", nargs=1, default=["test"],
#                     help="App directory")
# args = parser.parse_args()
# 
# curr_dir = os.getcwd()
# app_dir = os.path.join(curr_dir, args.dir[0])
# 
# os.chdir(app_dir)
# cmd_str = 'python main.py'
# os.system(cmd_str)
# os.chdir(curr_dir)
