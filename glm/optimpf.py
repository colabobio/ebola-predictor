"""
Script that generates the curve relating number of imputation frames 
and optimistic bias
"""

import os, argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit

def expfunc(x, a, b, c):
    return a * np.exp(-b * x) + c

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prefix", nargs=1, default=["pcr"],
                    help="Prefix of input files")
args = parser.parse_args()
prefix = args.prefix[0]

x = []
y = []
yerr = []
with open(prefix + "-res.csv", "r") as sfile:
    lines = sfile.readlines()
    for line in lines[1:]:
        line = line.strip()
        parts = line.split(',')
        x.append(int(parts[0]))
        y.append(float(parts[2]))
        yerr.append(float(parts[3]))

x = np.array(x)
y = np.array(y)
yerr = np.array(yerr)


popt, pcov = curve_fit(expfunc, x, y)
xi = np.linspace(x[0] - 0.25, x[-1] + 0.25, 50)

plt.clf()
fig = plt.figure()
plt.plot(xi, expfunc(xi, *popt), 'r-')
plt.errorbar(x, y, yerr=yerr, fmt='o', markersize=6, elinewidth=1.5)
plt.xlabel('Imputed frames #')
plt.ylabel('Mean optimistic bias')
plt.xlim([0.5,5.5])
plt.xticks(x)
fig.savefig("glm-" + prefix + ".pdf")
# plt.show()
