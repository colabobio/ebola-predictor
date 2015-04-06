#!/usr/bin/env python

import sys, os, threading
import time, glob, time
import itertools

master_file = "./data/variables-master.txt"

def get_last(name):
    train_files = glob.glob("./models/" + name + "/training-data-completed*.csv")
    idx = [int(fn[fn.rfind("-") + 1: fn.rfind(".")]) for fn in train_files]
    idx.sort()
    if idx: return idx[-1]
    else: return -1

def worker(name, count, first, imeth):
    print "Start"
    os.system("python init.py -N " + name + " -n " + str(count) + " -s " + str(first) + " -t " + str(test_prec) + " -m " + imeth + " " + impute_options[imeth])
    return

def create_var_file(mdl_id, mdl_vars):
    dir = "./models/" + str(mdl_id)
    if not os.path.exists(dir):
        os.makedirs(dir)
    vfn = dir + "/variables.txt"
    with open(vfn, "w") as vfile:
        vfile.write(out_var + " " + var_dict[out_var] + "\n")
        for v in mdl_vars:
            vfile.write(v + " " + var_dict[v] + "\n")

def run_model(mdl_id, mdl_vars):
    print "running model", mdl_id, mdl_vars
    create_var_file(mdl_id, mdl_vars)
    
    count = total_sets
    start = 0
    nrest = 0
    imeth = impute_method
    while True:
        thread = threading.Thread(target=worker, args=(mdl_id, count, start, imeth))
        thread.start()
        while True:
            time.sleep(0.1)
            if not thread.isAlive():
                break
        n = get_last(mdl_id)
        if n < total_sets - 1:
            if max_restarts < nrest:
                if not imeth == impute_fallback and impute_fallback:
                    print "Primary imputation, will try fallback imputation."
                    imeth == impute_fallback
                    start = 0
                    nrest = 0
                else:
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
        os.system("python train.py -N " + mdl_id + " " + pred_name + " " + pred_opt)
        repfn = "./models/" + mdl_id + "/report-" + pred_name + ".out"
        os.system("python eval.py -N " + mdl_id + " -p " + pred_name + " -m report > " + repfn)

##########################################################################################

if len(sys.argv) < 2:
    print "Need job file, quitting!"
    exit(0)

job_filename = sys.argv[1]

total_sets = 100
test_prec = 60
max_restarts = 5
predictors = ["lreg", "scikit_lreg"]
pred_options = {"lreg":"", "scikit_lreg":""}
impute_method = "hmisc"
impute_fallback = "mice"
impute_options = {"hmisc":"", "mice":""}
with open("job.cfg", "r") as cfg:
    lines = cfg.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        key,value = line.split("=", 1)
        if key == "total_sets": total_sets = int(value)
        elif key == "test_prec": test_prec = int(value)
        elif key == "max_restarts": max_restarts = int(value)
        elif key == "predictors": predictors = value.split(",")
        elif "pred_options" in key:
            pred = key.split(".")[1]
            pred_options[pred] = value
        elif key == "impute_method": impute_method = value
        elif key == "impute_fallback": impute_fallback = value
        elif "impute_options" in key:
            imp = key.split(".")[1]
            impute_options[imp] = value

all_vars = []
var_dict = {}
with open(master_file, "rb") as mfile:
    for line in mfile.readlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        name = parts[0]
        type = parts[1]
        all_vars.append(name)
        var_dict[name] = type
out_var = all_vars[0]

mdl_ids = []
mdl_vars = []
with open(job_filename, "r") as cfg:
    lines = cfg.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue
        id, vars = line.split(" ")
        mdl_ids.append(id)
        mdl_vars.append(vars.split(","))

for i in range(0, len(mdl_ids)):
    id = mdl_ids[i]
    vars = mdl_vars[i]
    run_model(id, vars)
