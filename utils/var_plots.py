"""
Generates plots for all variables

TODO: need to be generic for any type of dataset

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import os, csv
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

var_labels = { "OUT": "Outcome",
               "PCR": "PCR",
               "WEAK": "Weakness",
               "VOMIT": "Vomit",
               "AST_1": "AST",
               "Ca_1": "Ca",
               "AlkPhos_1": "ALK",
               "EDEMA": "Edema",
               "CONF": "Confussion",
               "Cl_1": "Cl",
               "TEMP": "Temperature",
               "RRATE": "Respiratory rate",
               "PBACK": "Back pain",
               "DIZZI": "Dizziness",
               "ALT_1": "ALT",
               "Cr_1": "CRE",
               "TCo2_1": "tCO2",
               "PRETROS": "Retrosternal pain",
               "DIARR": "Diarrhea",
               "HRATE": "Hearth rate",
               "Alb_1": "Alb",
               "BUN_1": "BUN",
               "TP_1": "TP",
               "PDIAST": "Diastolic pressure",
               "PABD": "Abdominal pain" }

var_units = { "OUT": "",
              "PCR": "log(EBOV copies/mL plasma)",
              "WEAK": "",
              "VOMIT": "",
              "AST_1": "U/L",
              "Ca_1": "mmol/L",
              "AlkPhos_1": "U/L",
              "EDEMA": "",
              "CONF": "",
              "Cl_1": "mmol/L",
              "TEMP": "Celsius",
              "RRATE": "BPM",
              "PBACK": "",
              "DIZZI": "",
              "ALT_1": "U/L",
              "Cr_1": "umol/L",
              "TCo2_1": "mmol/L",
              "PRETROS": "",
              "DIARR": "",
              "HRATE": "BPM",
              "Alb_1": "g/L",
              "BUN_1": "mmol urea/L",
              "TP_1": "g/L",
              "PDIAST": "mm Hg",
              "PABD": "" }


var_pvalues = { "OUT": 0,
                "PCR": 0.001,
                "WEAK": 0.06,
                "VOMIT": 0.15,
                "AST_1": 0.0007,
                "Ca_1": 0.48,
                "AlkPhos_1": 0.008,
                "EDEMA": 0.3,
                "CONF": 0.56,
                "Cl_1": 0.24,
                "TEMP": 0.004,
                "RRATE": 0.41,
                "PBACK": 0.55,
                "DIZZI": 0.17,
                "ALT_1": 0.03,
                "Cr_1": 0.04,
                "TCo2_1": 0.009,
                "PRETROS": 0.57,
                "DIARR": 0.024,
                "HRATE": 0.07,
                "Alb_1": 0.16,
                "BUN_1": 0.02,
                "TP_1": 0.14,
                "PDIAST": 0.59,
                "PABD": 0.37 }

parser = argparse.ArgumentParser()
parser.add_argument("-v", "--var_file", nargs=1, default=["./data/variables-master.txt"],
                    help="File containing list of variables to consider in the test")
args = parser.parse_args()

var_file = args.var_file[0]
# load_data(var_file)

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
#         
#         model_variables.append([0])

# dvar = model_variables[0]
# ivar = model_variables[1:]
# print dvar
# print ivar

src_file = "./data/sources.txt"
input_file = ""
with open(src_file, "rb") as sfile:
    for line in sfile.readlines():
        input_file = os.path.abspath(line.strip())
df = pd.read_csv(input_file, delimiter=",", na_values="\\N")

plt.clf()
fig = plt.figure()

# print df[dvar]
# color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')

range_file = "./data/ranges.txt"
range_variables = [] 
with open(range_file, "rb") as rfile:
    for line in rfile.readlines():
       line = line.strip()
       if not line: continue
       parts = line.strip().split()
       if 2 < len(parts):
           range_variables.append({"name":parts[0], "type":parts[1], "range":parts[2].split(",")})
ranges = (df["DIAG"] == 1) & (10 <= df["AGE"]) & (df["AGE"] <= 50) & ((df["OUT"] == 1) | (df["OUT"] == 0))

# sns.set(style="whitegrid")

binary_palette = sns.color_palette([(122/255.0, 192/255.0,119/255.0),(153/255.0, 94/255.0, 78/255.0)])
# binary_palette = sns.color_palette("Set1", 2)
# binary_palette = "coolwarm"
# binary_palette = sns.xkcd_palette(["pale red", "denim blue"])

# sns.set_context(rc = {'lines.linewidth': 0.0})

sns.set_context("paper", rc={"lines.linewidth": 2})

# sns.axes_style({'axes.linewidth': 1})
for var in numvar:
    with mpl.rc_context({"lines.linewidth": 0.0, 'axes.linewidth': 0.0}):
        g = sns.factorplot("OUT", var, data=df[ranges], kind="box", ci=0, size=3, aspect=1, palette=binary_palette)
        g.despine(offset=10, trim=True)
        g.set_xticklabels(['Discharged', 'Died'])
        plt.title(var_labels[var] + " - P=" + str(var_pvalues[var]))
        plt.xlabel('Outcome')
        plt.ylabel(var_units[var])
        plt.savefig("./out/" + var + ".pdf")
        plt.clf()

for var in catvar:
    with mpl.rc_context({"lines.linewidth": 0.5, 'patch.linewidth': 0.0}):
        g = sns.FacetGrid(df[ranges], aspect=1)
        g.map(sns.barplot, "OUT", var, palette=binary_palette)
        g.set_xticklabels(['Discharged', 'Died'])
        g.set(ylim=(0.0, 1.0))
        plt.title(var_labels[var] + " - P=" + str(var_pvalues[var]))		
        plt.xlabel('Outcome')
        plt.ylabel("Fraction")
        plt.savefig("./out/" + var + ".pdf")
        plt.clf()
