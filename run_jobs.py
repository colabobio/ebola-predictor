#!/usr/bin/env python

import sys

total_sets = 10
test_prec = 60
predictors = ["lreg", "scikit_lreg"]
pred_options = {"lreg":"", 
                "scikit_lreg":""}
# predictors = ["lreg", "nnet", "scikit_lreg", "scikit_dtree", "scikit_svm", "scikit_randf"]
# pred_options = {"lreg":"", 
#                 "nnet":"", 
#                 "scikit_lreg":"", 
#                 "scikit_dtree":"min_samples_leaf=10 max_depth=5 criterion=entropy",
#                 "scikit_svm":"error=10 kernel=rbf",
#                 "scikit_randf":"min_samples_leaf=10 max_depth=5 criterion=entropy"}
impute_method = "hmisc"
impute_fallback = "mice"
impute_options = {"hmisc":"",
                  "mice":""}
# var_file = "./data/variables.txt"
max_restarts = 10

if len(sys.argv) < 2:
    print "Need list of variables, quitting!"
    exit(0)

var_filename = sys.argv[1]
with 


print "hello", 











import os, threading
import time, glob, time
import itertools

total_sets = 10
test_prec = 60
model_sizes = [2]
predictors = ["lreg", "scikit_lreg"]
pred_options = {"lreg":"", 
                "scikit_lreg":""}
# predictors = ["lreg", "nnet", "scikit_lreg", "scikit_dtree", "scikit_svm", "scikit_randf"]
# pred_options = {"lreg":"", 
#                 "nnet":"", 
#                 "scikit_lreg":"", 
#                 "scikit_dtree":"min_samples_leaf=10 max_depth=5 criterion=entropy",
#                 "scikit_svm":"error=10 kernel=rbf",
#                 "scikit_randf":"min_samples_leaf=10 max_depth=5 criterion=entropy"}
impute_method = "hmisc"
impute_fallback = "mice"
impute_options = {"hmisc":"",
                  "mice":""}
master_file = "./data/variables-master.txt"
# var_file = "./data/variables.txt"
max_restarts = 10

def get_last(name):
    train_files = glob.glob("./models/" + name + "/training-data-completed*.csv")
    idx = [int(fn[fn.rfind("-") + 1: fn.rfind(".")]) for fn in train_files]
    idx.sort()
    if idx: return idx[-1]
    else: return -1

def worker(name, count, first):
    print "Start"
#     os.system("python clean.py")
    os.system("python init.py -N " + name + " -n " + str(count) + " -s " + str(first) + " -t " + str(test_prec) + " -m " + impute_method + " " + impute_options[impute_method])
    return

def create_var_file(mdl_count, fix_var, mdl_vars):
    dir = "./models/" + str(mdl_count)
    if not os.path.exists(dir):
        os.makedirs(dir)
    vfn = dir + "/variables.txt"
    with open(vfn, "w") as vfile:
        vfile.write(out_var + " " + var_dict[out_var] + "\n")
        if fix_var: vfile.write(fix_var + " " + var_dict[fix_var] + "\n")
        for v in mdl_vars:
            vfile.write(v + " " + var_dict[v] + "\n")

def run_model(fix_var, mdl_vars, mdl_count):
    vdict = [] 
    if fix_var: vdict = [fix_var]
    vdict.extend(mdl_vars)
    name = str(mdl_count)
    print "running model", name, vdict

    count = total_sets
    start = 0
    nrest = 0
    while True:
        thread = threading.Thread(target=worker, args=(name, count, start))
        thread.start()
        while True:
            time.sleep(0.1)
            if not thread.isAlive():
                break

        n = get_last(name)
        if n < total_sets - 1:
            if max_restarts < nrest:
                print "Model cannot be succesfully imputed, skipping!"
                return 
            start = n + 1
            count = total_sets - start
            nrest += 1
        else:
            print "Done! Number of restarts:", nrest
            break
            
    for pred_name in predictors:
        print "PREDICTOR",pred_name,"---------------"
        pred_opt = pred_options[pred_name]
        os.system("python train.py -N " + name + " " + pred_name + " "+ pred_opt)
        repfn = "./models/" + name + "/report-" + pred_name + ".out"
        os.system("python eval.py -N " + name + " -p " + pred_name + " -m report > " + repfn)

##########################################################################################

all_vars = []
var_dict = {}
fix_var = None
with open(master_file, "rb") as mfile:
    for line in mfile.readlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        name = parts[0]
        type = parts[1]
        if len(parts) == 3 and parts[2] == "*": fix_var = name
        all_vars.append(name)
        var_dict[name] = type
out_var = all_vars[0]
mdl_vars = all_vars[1:]
if fix_var: mdl_vars.remove(fix_var)

model_count = 0
for size in model_sizes:
    fsize = size
    if fix_var: fsize -= 1
    var_comb = itertools.combinations(mdl_vars, fsize)
    for mvars in var_comb:
        create_var_file(model_count, fix_var, mvars)
        run_model(fix_var, mvars, model_count)
        model_count += 1
        
        
        
'''
    ##a very useful function for writing Python scripts to give Unix/LSF commands

    import subprocess

    def givecommand(commandstr):
        commands = commandstr.split(' ')
        subprocess.check_output(commands)

     ##sample usage for a script, runReps.py, that takes the number of replicates to run as an argument
    reps = [10, 20, 30]
    for nrep in reps:
        outfile = "out_" + str(nrep) + "reps_run" + str(i)
        runreps_command = "bsub -P sabeti_CMS -q week -o " + outfile
        runreps_command += " python runReps.py " + str(nrep)
        #print samplepoint_command
        givecommand(samplepoint_command)
'''

use R-3.1
use .anaconda-2.1.0