'''
Run variety of evaluation metrics on predictive model.
'''

from calibrationdiscrimination import caldis
from calplot import calplot
from classificationreport import report
from confusion import confusion
from roc import roc

def eval(probs, y_test, method=1):

	if method == 1:
		return caldis(probs, y_test)
	
	elif method == 2:
		return calplot(probs, y_test)
	
	elif method == 3:
		return report(probs, y_test)
	
	elif method == 4:
		return roc(probs, y_test)
		
	elif method == 5:
		return confusion(probs, y_test)
	
	else:
		raise Exception("Invalid method argument given")