"""
Rank all available models

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

for var in numvar:
    #df[ranges].boxplot(var, by=depvar)
    g = sns.factorplot("OUT", var, data=df[ranges], kind="box", ci=0, palette="coolwarm")
    g.despine(offset=10, trim=True)
    g.set_xticklabels(['Discharged', 'Died'])
    plt.xlabel('Outcome')    
    plt.ylabel(var_labels[var])
    plt.savefig("./out/" + var + ".pdf")
    plt.clf()

for var in catvar:
	with mpl.rc_context({"lines.linewidth": 0.5}):
# 		g = sns.barplot(df[ranges][var], df[ranges]["OUT"], palette="coolwarm")
		g = sns.FacetGrid(df[ranges], aspect=1)
		g.map(sns.barplot, "OUT", var, palette="coolwarm");
		g.set_xticklabels(['Discharged', 'Died'])
		g.set(ylim=(0.0, 1.0))
        plt.xlabel('Outcome')    
        plt.ylabel(var_labels[var])		
#     df[ranges].hist(var, by=depvar)
#     df[ranges].plot(kind='area', var, by=depvar)
# 
    	plt.savefig("./out/" + var + ".pdf")
    	plt.clf()
