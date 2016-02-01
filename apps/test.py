"""
Runs the app on all the data instances

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, glob, operator
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath('./ebolacare'))
from utils import gen_predictor

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

count = 0
nmiss = 0
nmatch = 0
for idx, row in data.iterrows():
    age = row['AGE']
    out = row['OUT']    
    if age < 10 or 50 < age or np.isnan(out): continue
    vv = set([])
    values = {}
    
    for var in variables:
        val = row[var]
        if not np.isnan(val): 
            values[var] = val
            vv.add(var)

    if vv:
        model_dir = None
        model_vars = None
        for info in models_info: 
            v = set(info[1])
            res = v.issubset(vv)
            #print res, info[1]
            if res:
                model_dir = info[0]
                model_vars = info[1]
                break
                
        if not model_dir or not models_info:
            continue
            
        count += 1
        
        predictor = gen_predictor(os.path.join(model_dir, 'nnet-params'))
        N = len(model_vars)

        model_min = []
        model_max = []
        with open(os.path.join(model_dir, 'bounds.txt')) as bfile:
            lines = bfile.readlines()
            for line in lines:
                line = line.strip()
                parts = line.split()
                model_min.append(float(parts[1]))  
                model_max.append(float(parts[2]))

#         print model_min
#         print model_max                 
        
        v = [None] * (N + 1)
        v[0] = 1
        for i in range(N):
            var = model_vars[i]
            if var in values:
                try:                  
                    v[i + 1] = float(values[var])
                    if var in units and units[var] != var_def_unit[var]:
                        # Need to convert units
                        c, f = var_unit_conv[var]
                        print "convert",var,v[i + 1],"->",f*(v[i + 1] + c)
                        v[i + 1] = f * (v[i + 1] + c)
                except ValueError:
                    pass
                except TypeError:
                    pass
                    
        if None in v:
            print 'INSUFFICIENT DATA'
            continue

        for i in range(N):
            f = (v[i + 1] - model_min[i]) / (model_max[i] - model_min[i]) 
            if f < 0: v[i + 1] = 0
            elif 1 < f: v[i + 1] = 1
            else: v[i + 1] = f

#         print values
#         print v
        
        X = np.array([v])
        probs = predictor(X)
        pred = probs[0]
#         print "------------->",pred, out
        
        if pred < 0.5:
            if out == 1: 
                nmiss += 1
                print "Mismatch:",out,pred
            else: nmatch += 1
        else:          
            if out == 1: nmatch += 1
            else: 
                nmiss += 1
                print "Mismatch:",out,pred                
        
print "total:",count
print "misses:",nmiss
print "matches:",nmatch


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
