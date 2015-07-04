"""
Generates plots for all variables

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import os, csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--var_file", nargs=1, default=["./data/variables-master.txt"],
                    help="File containing list of variables to consider in the test")
parser.add_argument("-p", "--pvalue_file", nargs=1, default=["./out/variable-stats.tsv"],
                    help="File containing the p-value for each variable")

args = parser.parse_args()
var_file = args.var_file[0]
pvalue_file = args.pvalue_file[0]

var_labels = {}
with open("./data/alias.txt", "r") as afile:
    lines = afile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ", 1)
        var_labels[parts[0]] = parts[1]

var_units = {}
with open("./data/units.txt", "r") as ufile:
    lines = ufile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(" ", 1)
        if 1 < len(parts):
            var_units[parts[0]] = parts[1]
        else:
            var_units[parts[0]] = ""

outcome_states = []
outcome_labels = []
outcome_colors = []
with open("./data/outcome.txt", "r") as ofile:
    lines = ofile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split(',')
        val = parts[0]
        label = parts[1]
        color = tuple([float(x)/255.0 for x in parts[2].split()])
        outcome_states.append(val)
        outcome_labels.append(label)
        outcome_colors.append(color)

var_pvalues = {}
with open(pvalue_file, "r") as pfile:
    lines = pfile.readlines()
    for line in lines:
        line = line.strip()
        parts = line.split("\t", 1)
        var_pvalues[parts[0]] = float(parts[1])

depvar = None
numvar = []
catvar = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        line = line.strip()
        if not line: continue
        parts = line.split()
        if not depvar: depvar = parts[0]
        else:
            if parts[1] == "category": catvar.append(parts[0])
            else: numvar.append(parts[0])

src_file = "./data/sources.txt"
input_file = ""
with open(src_file, "rb") as sfile:
    for line in sfile.readlines():
        input_file = os.path.abspath(line.strip())
df = pd.read_csv(input_file, delimiter=",", na_values="\\N")

plt.clf()
fig = plt.figure()

range_file = "./data/ranges.txt"
range_variables = []
ranges = ((df["OUT"] == 1) | (df["OUT"] == 0))
with open(range_file, "rb") as rfile:
    for line in rfile.readlines():
       line = line.strip()
       if not line: continue
       parts = line.strip().split()
       if 2 < len(parts):
             name = parts[0]
             type = parts[1]
             values = parts[2].split(",")
             if type == "category":
                 for val in values:
                     ranges = ranges & (df[name] == int(val))
             elif len(values) == 2:
                 vmin = float(values[0])
                 vmax = float(values[1])
                 ranges = ranges & (vmin <= df[name]) & (df[name] <= vmax)

binary_palette = sns.color_palette(outcome_colors)
# binary_palette = sns.color_palette("Set1", 2)
# binary_palette = "coolwarm"
# binary_palette = sns.xkcd_palette(["pale red", "denim blue"])

sns.set_context("paper", rc={"lines.linewidth": 2})
for var in numvar:
    print "Generating factorplot for",var
    with mpl.rc_context({"lines.linewidth": 0.0, 'axes.linewidth': 0.0}):
        g = sns.factorplot("OUT", var, data=df[ranges], kind="box", ci=0, size=3, aspect=1, palette=binary_palette)
        g.despine(offset=10, trim=True)
        g.set_xticklabels(outcome_labels)
        p = var_pvalues[var]
        if p < 0.001:
            pstr = "<0.001"
        elif p < 0.01:
            pstr = "=%.3f" % var_pvalues[var]
        else:
            pstr = "=%.2f" % var_pvalues[var]
        plt.title(var_labels[var] + " - P" + pstr)
        plt.xlabel('Outcome')
        plt.ylabel(var_units[var])
        plt.savefig("./out/" + var + ".pdf")
        plt.clf()

for var in catvar:
    print "Generating barplot for",var
    with mpl.rc_context({"lines.linewidth": 0.5, 'patch.linewidth': 0.0}):
        g = sns.FacetGrid(df[ranges], aspect=1)
        g.map(sns.barplot, "OUT", var, palette=binary_palette)
        g.set_xticklabels(outcome_labels)
        g.set(ylim=(0.0, 1.0))
        if p < 0.001:
            pstr = "<0.001"
        elif p < 0.01:
            pstr = "=%.3f" % var_pvalues[var]
        else:
            pstr = "=%.2f" % var_pvalues[var]
        plt.title(var_labels[var] + " - P" + pstr)
        plt.xlabel('Outcome')
        plt.ylabel("Fraction")
        plt.savefig("./out/" + var + ".pdf")
        plt.clf()
