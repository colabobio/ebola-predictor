'''
Creates calibration plot and Hosmer-Lemeshow statistics.
Main R package from http://www.genabel.org/PredictABEL/plotCalibration.html
'''
import rpy2.robjects as robjects
import time

from makepredictions import nnet_pred

test_file = "testing-data.csv"
outplot = '"calplot"'

probs, y_test = nnet_pred()
predrisk = 'c('+ ', '.join([str(p) for p in probs])+')'

robjects.r.X11()

robjects.r('library(PredictABEL)')
robjects.r('trdat <- read.table("' + test_file + '", sep = ",", header=TRUE)')
robjects.r('cOutcome <- 1')
robjects.r('predRisk <- '+predrisk)
robjects.r('outplot <-'+outplot)
robjects.r('plotCalibration(data=trdat, cOutcome=cOutcome, predRisk=predRisk)')

print "Ctrl+C to close"

while True:
	time.sleep(0.1)