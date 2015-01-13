'''
Run variety of evaluation metrics on predictive model.
'''

from calibrationdiscrimination import caldis
from calplot import calplot
from classificationreport import report
from roc import roc

def eval(probs, y_test, method=1):

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