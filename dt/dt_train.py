'''
Trains the Decision Tree predictor.
'''
import pandas as pd
import pickle

from sklearn import tree
from dt_utils import design_matrix

training_filename = "./data/training-data-imputed.csv"
model_file = "./data/dt-model.p"

# Loading data frame
df = pd.read_csv(training_filename, delimiter=',', na_values="?")

# Separating target from inputs
X, y = design_matrix(df)

# Initializing DT classifier
clf = tree.DecisionTreeClassifier()

# Fitting DT classifier
clf.fit(X,y)

# Pickle and save
f = open(model_file, 'wb')
pickle.dump(clf, f)