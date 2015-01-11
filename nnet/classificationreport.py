'''
Builds a text report showing precision, recall, F1 score.
'''
from makepredictions import nnet_pred
from sklearn.metrics import classification_report

probs, y_test = nnet_pred()
preds = [int(0.5 < p) for p in probs]

target_names = ['Died', 'Survived']
report = classification_report(y_test, preds, target_names=target_names)

print report