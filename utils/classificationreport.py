'''
Builds a text report showing precision, recall, F1 score.
'''
from sklearn.metrics import classification_report

def report(probs, y_test):

	preds = [int(0.5 < p) for p in probs]

	target_names = ['Died', 'Survived']
	report = classification_report(y_test, preds, target_names=target_names)

	print report