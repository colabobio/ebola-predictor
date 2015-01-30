'''
Builds a text report showing precision, recall, F1 score.
'''
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def report(probs, y_test):

	preds = [int(0.5 < p) for p in probs]

	target_names = ['Survived', 'Died']
	report = classification_report(y_test, preds, target_names=target_names)

	print report
	
	return precision_recall_fscore_support(y_test, preds)