#!/usr/bin/env python

"""
Rank all available models

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse
import operator

var_file = "./data/variables.txt"
def load_vars(fn):
    res = []
    if not os.path.exists(fn):
        fn = var_file
    with open(fn, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            res.append(parts[0])
    return res[1:]

def reading_model(train_files, report_files, mdl_num):
    count = 0
    for rfn in report_files:
        with open(rfn, "r") as report:
            pred = os.path.splitext(rfn.split("-")[-1])[0]
            lines = report.readlines()
            if not lines:
                print "  Cannot find scores in",rfn,", skipping!"
                continue
            for pred in predictors:
                if pred in rfn:
                    count += 1
                    break
            last = lines[-1]
            parts = last.split(",")
            mdl_name = mdl_num + "-" + pred
            if 3 < len(parts):
                model_f1_scores[mdl_name] = float(parts[3])
                model_f1_dev[mdl_name] = float(parts[6])
                model_vars[mdl_name] = mdl_vars
            else:
                print "  Cannot find scores in",rfn,", skipping!"
    return count

model_f1_scores = {}
model_f1_dev = {}
model_vars = {}
incomplete_models = []

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument('-p', '--pred_list', nargs=1, default=[""],
                    help="Predictors to search results for")

args = parser.parse_args()
base_dir = args.base_dir[0]

predictors = []
if args.pred_list[0]:
    predictors = args.pred_list[0].split(",")

mdl_count = 0
for dir_name, subdir_list, file_list in os.walk(base_dir):
    if file_list:
        train_files = glob.glob(dir_name + "/training-data-completed-*.csv")
        if train_files:
            mdl_count += 1
            report_files = glob.glob(dir_name + "/report-*.out")
            mdl_vars = load_vars(dir_name + "/variables.txt")
            mdl_num = dir_name
            print "Reading model",mdl_num,"with variables", ",".join(mdl_vars)
            count = reading_model(train_files, report_files, mdl_num)
            if count < len(predictors):
                incomplete_models.append(dir_name)

print "Number of models:",mdl_count
print "Number of incomplete models:",len(incomplete_models)

sorted_preds = reversed(sorted(model_f1_scores.items(), key=operator.itemgetter(1)))
with open(base_dir + "/ranking.txt", "w") as rfile:
    for pair in sorted_preds:
        full_name = pair[0]
        idx = full_name.rfind('-')
        mdl_name = full_name[0:idx]
        pred_name = full_name[idx + 1:]
        pred_f1_score = pair[1]
        pred_f1_std = model_f1_dev[full_name]
        mvars = model_vars[full_name]
        line = mdl_name + " " + pred_name + " " + ",".join(mvars) + " " + str(pred_f1_score) + " " + str(pred_f1_std)
        rfile.write(line + "\n")
print "Saved ranking to",base_dir + "/ranking.txt"

if incomplete_models:
    with open(base_dir + "/incomplete.txt", "w") as rfile:
        for mdl in incomplete_models:
            rfile.write(mdl + "\n")
    print "Saved incomplete models to",base_dir + "/incomplete.txt"