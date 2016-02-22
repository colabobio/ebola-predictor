import os, argparse
import numpy as np
 
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefix", nargs=1, default=["pcr"],
                    help="Prefix of input files")
args = parser.parse_args()
prefix = args.prefix[0]
 
olines = []

with open(prefix + "-res.txt", "r") as ifile:
    nfr = 0
    lines = ifile.readlines()
    for line in lines:
        line = line.strip()
        if not line: continue

        if "---" in line:
            if 0 < nfr:
                oline = "%i,%f,%f,%f\n" % (nfr, np.mean(auc), np.mean(bias), np.std(bias))
                olines.append(oline)

            nfr += 1
            auc = []
            bias = []
            
        elif "Apparent AUC" in line:
            parts = line.split()
            auc.append(float(parts[-1].replace('"', '')))
            
        elif "Mean AUC bias" in line:
            parts = line.split()
            bias.append(float(parts[-1].replace('"', '')))

    oline = "%i,%f,%f,%f\n" % (nfr, np.mean(auc), np.mean(bias), np.std(bias))
    olines.append(oline)

with open(prefix + "-res.csv", "w") as ofile:
    title = "Imputed frames #,Mean AUC,Mean Bias,STD bias\n"
    ofile.write(title)
    for oline in olines:
        ofile.write(oline)