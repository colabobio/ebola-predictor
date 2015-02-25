"""
Create a graphical representation of the decision tree model.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os
import argparse
import pandas as pd
import pickle
import pydot

from sklearn import tree
from sklearn.externals.six import StringIO 

parser = argparse.ArgumentParser()
parser.add_argument("param", nargs='?', default="./data/dtree-params",
                    help="parameters of decision tree")
args = parser.parse_args()

if not os.path.exists("./out"): os.makedirs("./out")
out_file = './out/dtree.pdf'

# Load the decision tree
clf = pickle.load(open(args.param, "rb" ))

# Get the names of the features
train_filename = "./data/training-data-completed.csv"
df = pd.read_csv(train_filename, delimiter=",", na_values="?")
features = df.columns.values

# Draw and save decision tree
data = StringIO()
tree.export_graphviz(clf, out_file=data, feature_names=features) 
graph = pydot.graph_from_dot_data(data.getvalue())
for node in graph.get_nodes():
	node.add_style('filled')
	node.set('fillcolor', 'lightblue')
graph.write_pdf(out_file)

print "Saved graphical representation of decision tree to",out_file