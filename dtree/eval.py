'''
Run variety of evaluation metrics on nnet predictive model.
'''

import sys
import os
sys.path.append(os.path.abspath('./utils'))

from evaluate import eval
from dt_utils import dt_pred

probs, y_test = dt_pred()

method = 1

if 1 < len(sys.argv):
    method = int(sys.argv[1])
    
eval(probs, y_test, method)