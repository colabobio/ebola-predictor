'''
Run variety of evaluation metrics on nnet predictive model.
'''

import sys
import os
sys.path.append(os.path.abspath('./utils'))

from calibrationdiscrimination import caldis
from calplot import calplot
from classificationreport import report
from roc import roc

from makepredictions import nnet_pred

probs, y_test = nnet_pred()

method = 1

if 1 < len(sys.argv):
    method = int(sys.argv[1])

if method == 1:
	caldis(probs, y_test)
	
elif method == 2:
	calplot(probs, y_test)
	
elif method == 3:
	report(probs, y_test)
	
elif method == 4:
	roc(probs, y_test)
	
else:
	raise Exception("Invalid method argument given")