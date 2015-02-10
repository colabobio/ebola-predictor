"""
Display pairwise plots for each combination of predictor variables, labeling the items
according to the value of the dependent variable in the training set.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import sys
import itertools
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt

var_file = "./data/variables.txt"

"""Plots a scatterplot matrix of subplots. Each row of "data" is plotted against other 
rows, resulting in a nrows by nrows grid of subplots with the diagonal subplots labeled 
with "names".  Additional keyword arguments are passed on to matplotlib's "plot" command. 
Returns the matplotlib figure object containg the subplot grid.

Adapted from: 
http://stackoverflow.com/questions/7941207/is-there-a-function-to-make-scatterplot-matrices-in-matplotlib
"""
def scatterplot_matrix(data, names=[], types={}, **kwargs):
    print "Generating scatterplot matrix..."
    numvars, numdata = data.shape
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)

    for ax in axes.flat:
        # Hide all ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        # Set up ticks only on one side for the "edge" subplots...
        if ax.is_first_col():
            ax.yaxis.set_ticks_position('left')
        if ax.is_last_col():
            ax.yaxis.set_ticks_position('right')
        if ax.is_first_row():
            ax.xaxis.set_ticks_position('top')
        if ax.is_last_row():
            ax.xaxis.set_ticks_position('bottom')

    if names and types:
        # Jittering categorical values so they are not overdrawn
        for i in range(0, len(names)):
            if types[names[i]]:
                data[i] = data[i] + np.random.uniform(-0.07, 0.07, numdata)

    # Plot the data.
    for i, j in zip(*np.triu_indices_from(axes, k=1)):
        for x, y in [(i,j), (j,i)]:
            # FIX #1: this needed to be changed from ...(data[x], data[y],...)
            dy = data[y]
            dx = data[x]
            axes[x,y].scatter(dy, dx, **kwargs)

    # Label the diagonal subplots...
    if not names:
        names = ['x'+str(i) for i in range(numvars)]

    for i, label in enumerate(names):
        axes[i,i].annotate(label, (0.5, 0.5), xycoords='axes fraction',
                ha='center', va='center')

    # Turn on the proper x or y axes ticks.
    for i, j in zip(range(numvars), itertools.cycle((-1, 0))):
        axes[j,i].xaxis.set_visible(True)
        axes[i,j].yaxis.set_visible(True)

    # FIX #2: if numvars is odd, the bottom right corner plot doesn't have the
    # correct axes limits, so we pull them from other axes
    if numvars%2:
        xlimits = axes[0,-1].get_xlim()
        ylimits = axes[-1,0].get_ylim()
        axes[-1,-1].set_xlim(xlimits)
        axes[-1,-1].set_ylim(ylimits)

    print "Done."
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", nargs='?', default="./data/training-data-completed.csv",
                        help="data file to plot")
    args = parser.parse_args()

    var_types = {}
    with open(var_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            [name, type] = line.split()[0:2]
            var_types[name] = type == "category"

    df = pd.read_csv(args.data, delimiter=',', na_values="?")
    M = df.shape[0]
    N = df.shape[1]
    names = df.columns.values[1: N].tolist()
    dvar = df.as_matrix(columns=df.columns.values[0: 1])
    data = np.transpose(df.as_matrix(columns=names))

    fig = scatterplot_matrix(data, names, var_types, c=dvar, marker='o', s = 10, edgecolors = 'none', alpha=0.5)
    fig.suptitle('Model variables scatter plot')
    plt.show()
