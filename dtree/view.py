'''
Create a graphical representation of the decision tree model.
'''

import pickle
import pydot

from sklearn import tree
from sklearn.externals.six import StringIO 

model_filename = './data/dt-model.p'
outfile = './data/dt.pdf'

# Load the decision tree
clf = pickle.load(open( model_filename, "rb" ) )

# Draw and save decision tree
data = StringIO()
tree.export_graphviz(clf, out_file=data) 
graph = pydot.graph_from_dot_data(data.getvalue())
graph.write_pdf(outfile)