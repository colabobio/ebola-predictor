import os, threading
import time, glob, time
import itertools

total_sets = 10
test_prec = 50
model_size = 5
impute_meth = "amelia"
master_file = "./data/variables-master.txt"
var_file = "./data/variables.txt"

import ctypes

def terminate_thread(thread):
    """Terminates a python thread from another thread.

    :param thread: a threading.Thread instance
    """
    if not thread.isAlive():
        return

    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(thread.ident), exc)
    if res == 0:
        raise ValueError("nonexistent thread id")
    elif res > 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def get_last():
    train_files = glob.glob("./data/training-data-completed*.csv")
    idx = [int(fn[fn.rfind("-") + 1: fn.rfind(".")]) for fn in train_files]
    idx.sort()
    if idx: return idx[-1]
    else: return -1

def worker(count, first):
    print "Start"
    os.system("python clean.py")
    os.system("python init.py -n " + str(count) + " -s " + str(first) + " -t " + str(test_prec) + " -m " + impute_meth + " > super.out")
    return

def run_model(mdl_count):
    count = total_sets
    start = 0
    nrest = 0
    while True:
        n0 = 0
        t0 = time.time() 
        thread = threading.Thread(target=worker, args=(count, start))
        thread.start()
        while True:
            time.sleep(0.1)
            n = get_last()
            t = time.time() 
            if n == n0:
                if 30 < t - t0:
                    print "amelia seems unable to converge, will try to terminate..."
                    terminate_thread(thread)
                    break
            else:
                t0 = time.time()
            n0 = n
            if not thread.isAlive():
                break

        n = get_last()
        if n < total_sets - 1:
            start = n + 1
            count = total_sets - start
            nrest += 1
        else:
            print "Done! Number of restarts:", nrest
            break
    os.system("python train.py lreg")
    os.system("python eval.py -p lreg -m report")
    os.system("python pack.py model-" + str(mdl_count))

##########################################################################################

all_vars = []
var_dict = {}
with open(master_file, "rb") as mfile:
    for line in mfile.readlines():
        line = line.strip()
        if not line: continue
        nam, typ = line.split()
        all_vars.append(nam)
        var_dict[nam] = typ

out_var = all_vars[0]
mdl_vars = all_vars[1:]
# print out_var
# print mdl_vars


model_count = 0
var_comb = itertools.combinations(mdl_vars, model_size)
for vars in var_comb:
    with open(var_file, "w") as vfile:
        vfile.write(out_var + " " + var_dict[out_var] + "\n")
        for v in vars:
            vfile.write(v + " " + var_dict[v] + "\n")
    print "running model",vars,"**************************************" 
    run_model(model_count)
    model_count += 1



