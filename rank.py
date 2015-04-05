import os, glob
import operator

def load_vars(fn):
    res = [] 
    with open(fn, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            parts = line.split()
            res.append(parts[0])
    return res[1:]

model_f1_scores = {}
model_f1_dev = {}
model_vars = {}

model_dirs = glob.glob("./models/*")
for dir in model_dirs:
    if not os.path.isdir(dir): continue
    mdl_num = dir.split("/")[2]
    mdl_vars = load_vars(dir + "/variables.txt")
    report_files = glob.glob(dir + "/report-*.out")
    print "Reading model",mdl_num,"with variables", ",".join(mdl_vars)
    for rfn in report_files:
        with open(rfn, "r") as report:
            pred = os.path.splitext(rfn.split("-", 1)[1])[0]
            last = report.readlines()[-1]
            parts = last.split(",")
            mdl_name = mdl_num + "-" + pred
            model_f1_scores[mdl_name] = float(parts[3])
            model_f1_dev[mdl_name] = float(parts[6])
            model_vars[mdl_name] = mdl_vars

sorted_preds = reversed(sorted(model_f1_scores.items(), key=operator.itemgetter(1)))
with open("./models/ranking.txt", "w") as rfile:
    for pair in sorted_preds:
        full_name = pair[0]
        mdl_count, pred_name = full_name.split("-")
        pred_f1_score = pair[1]
        pred_f1_std = model_f1_dev[full_name]
        mvars = model_vars[full_name]
        line = mdl_count + " " + pred_name + " " + ",".join(mvars) + " " + str(pred_f1_score) + " " + str(pred_f1_std)
        rfile.write(line + "\n")
