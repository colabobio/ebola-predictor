#!/usr/bin/env python

"""
Exhaustive generation of models

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse, shutil, itertools

def save_clump(dir, count, ids, vars):
    print "Create file for job", count
    with open(dir + "/job-" + str(count), "w") as jfile:
        for k in range(0, len(vars)):
            jfile.write(str(ids[k]) + " " + ",".join(vars[k]) + "\n")

parser = argparse.ArgumentParser()
parser.add_argument("-V", "--var_file", nargs=1, default=["./data/variables-master.txt"],
                    help="File with list of variables")
parser.add_argument("-i", "--incomplete", nargs=1, default=[""],
                    help="list of incomplete models")
parser.add_argument("-s", "--model_sizes", nargs=1, default=["2,3,4"],
                    help="list of model sizes")
parser.add_argument("-c", "--clump_size", type=int, nargs=1, default=[5],
                    help="size of models inside a single job")
parser.add_argument("-d", "--jobs_dir", nargs=1, default=["./jobs"],
                    help="Directory where to save job files")

args = parser.parse_args()

dir = args.jobs_dir[0]
if not os.path.exists(dir): os.makedirs(dir)
else:
    files = glob.glob(dir + "/*")
    if files:
        print "Jobs folder not empty, deleting all its contents..."
        for file in files:
            if os.path.isdir(file):
                shutil.rmtree(file)
            else:
                os.remove(file)
        print "Done."

inc_file = args.incomplete[0]
if inc_file:
    # Generate jobs from list of incomplete models
    inc_models = []
    with open(inc_file, "rb") as ifile:
        for line in ifile.readlines():
            inc_models.append(line.strip())
    count = 0
    for mdl in inc_models:
        _, name = os.path.split(mdl)
        vars = []
        with open(os.path.join(mdl, "variables.txt"), "rb") as vfile:
            lines = vfile.readlines()[1:]
            for line in lines:
                line = line.strip()
                if line: vars.append(line.split()[0])
        save_clump(dir, count, [name], [vars])
        count += 1
    exit(0)

sizes = args.model_sizes[0]
def_clump_size = args.clump_size[0]
clump_sizes = []
parts = sizes.split(',')
model_sizes = []
for s in parts:
    if ":" in s:
        s, sc = s.split(":")
        c = int(sc)
    else:
        c = def_clump_size
    if '-' in s:
        t = s.split('-')
        sizes = range(int(t[0]), int(t[1]) + 1)
        model_sizes.extend(sizes)
        clump_sizes.extend([c] * len(sizes))
    else:
        model_sizes.append(int(s))
        clump_sizes.append(c)

all_vars = []
var_dict = {}
fix_var = None
var_file = args.var_file[0]
with open(var_file, "rb") as mfile:
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
clump_count = 0
clump_idxs = []
clump_vars = []
for i in range(0, len(model_sizes)):
    size = model_sizes[i]
    fsize = size
    if fix_var: fsize -= 1
    var_comb = itertools.combinations(mdl_vars, fsize)
    for mvars in var_comb:
        if len(clump_vars) == clump_sizes[i]:
            save_clump(dir, clump_count, clump_idxs, clump_vars)
            clump_count += 1
            clump_idxs = []
            clump_vars = []
        lvars = []
        if fix_var: lvars.append(fix_var)
        lvars.extend(mvars)
        clump_idxs.append(model_count)
        clump_vars.append(lvars)
        model_count += 1
    if clump_vars:
        save_clump(dir, clump_count, clump_idxs, clump_vars)
        clump_count += 1
        clump_idxs = []
        clump_vars = []