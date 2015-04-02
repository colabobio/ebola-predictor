"""
Exhaustive generation of models

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, threading
import time, glob, time
import itertools
import operator

total_sets = 10
test_prec = 50
model_sizes = [2]
predictors = ["lreg", "nnet", "scikit_lreg", "scikit_dtree", "scikit_svm", "scikit_randf"]
pred_options = {"lreg":"", 
                "nnet":"", 
                "scikit_lreg":"", 
                "scikit_dtree":"min_samples_leaf=10 max_depth=5 criterion=entropy",
                "scikit_svm":"error=10 kernel=rbf",
                "scikit_randf":"min_samples_leaf=10 max_depth=5 criterion=entropy"}
impute_meth = "amelia"
master_file = "./data/variables-master.txt"
var_file = "./data/variables.txt"
max_restarts = 10

# import ctypes

# def terminate_thread(thread):
#     """Terminates a python thread from another thread.
# 
#     :param thread: a threading.Thread instance
#     """
#     if not thread.isAlive():
#         return
# 
#     exc = ctypes.py_object(SystemExit)
#     res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
#         ctypes.c_long(thread.ident), exc)
#     if res == 0:
#         raise ValueError("nonexistent thread id")
#     elif res > 1:
#         # """if it returns a number greater than one, you're in trouble,
#         # and you should call it again with exc=NULL to revert the effect"""
#         ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
#         raise SystemError("PyThreadState_SetAsyncExc failed")

def get_last():
    train_files = glob.glob("./data/training-data-completed*.csv")
    idx = [int(fn[fn.rfind("-") + 1: fn.rfind(".")]) for fn in train_files]
    idx.sort()
    if idx: return idx[-1]
    else: return -1

def worker(count, first):
    print "Start"
    os.system("python clean.py")
    os.system("python init.py -n " + str(count) + " -s " + str(first) + " -t " + str(test_prec) + " -m " + impute_meth)
    return

def create_var_file(fix_var, mdl_vars):
    with open(var_file, "w") as vfile:
        vfile.write(out_var + " " + var_dict[out_var] + "\n")
        if fix_var: vfile.write(fix_var + " " + var_dict[fix_var] + "\n")
        for v in mdl_vars:
            vfile.write(v + " " + var_dict[v] + "\n")

def run_model(fix_var, mdl_vars, mdl_count):
    vdict = [] 
    if fix_var: vdict = [fix_var]
    vdict.extend(mdl_vars)
    print "running model", mdl_count, vdict

    count = total_sets
    start = 0
    nrest = 0
    while True:
#         n0 = 0
#         t0 = time.time() 
        thread = threading.Thread(target=worker, args=(count, start))
        thread.start()
        while True:
            time.sleep(0.1)
#             n = get_last()
#             t = time.time() 
#             if n == n0:
#                 if 30 < t - t0:
#                     print "amelia seems unable to converge, will try to terminate..."
#                     terminate_thread(thread)
#                     break
#             else:
#                 t0 = time.time()
#             n0 = n
            if not thread.isAlive():
                break

        n = get_last()
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
        mdl_name = "model-" +str(mdl_count) + "-" + pred_name
        model_vars[mdl_name] = vdict
        pred_opt = pred_options[pred_name]
        os.system("python train.py " + pred_name + " "+ pred_opt)
        os.system("python eval.py -p " + pred_name + " -m report > report.out")
        with open("report.out", "r") as report:
            last = report.readlines()[-1]
            parts = last.split(",")
            print "------------------------>",float(parts[3]), float(parts[6])
            model_f1_scores[mdl_name] = float(parts[3])
            model_f1_dev[mdl_name] = float(parts[6])
    os.system("python pack.py model-" +str(mdl_count))

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

model_f1_scores = {}
model_f1_dev = {}
model_vars = {}

model_count = 0
for size in model_sizes:
    fsize = size
    if fix_var: fsize -= 1
    var_comb = itertools.combinations(mdl_vars, fsize)
    for mvars in var_comb:
        create_var_file(fix_var, mvars)
        run_model(fix_var, mvars, model_count)
        model_count += 1

sorted_preds = reversed(sorted(model_f1_scores.items(), key=operator.itemgetter(1)))

with open("store/ranking.txt", "w") as rfile:
    for pair in sorted_preds:
        pred_name = pair[0]
        pred_f1_score = pair[1]
        pred_f1_std = model_f1_dev[pred_name]
        mvars = model_vars[pred_name]
        rfile.write(pred_name + " " + ",".join(mvars) + " " + str(pred_f1_score) + " " + str(pred_f1_std) + "\n")
