'''
Creates calibration plot and Hosmer-Lemeshow statistics.
Main R package from http://www.genabel.org/PredictABEL/plotCalibration.html
'''
import rpy2.robjects as robjects
import time

test_file = "./data/testing-data.csv"

def calplot(probs, y_test):
	predrisk = 'c('+ ', '.join([str(p) for p in probs])+')'

	robjects.r.X11()

	robjects.r('library(PredictABEL)')
	robjects.r('trdat <- read.table("' + test_file + '", sep = ",", header=TRUE)')
	robjects.r('cOutcome <- 1')
	robjects.r('predRisk <- '+predrisk)
	robjects.r('plotCalibration(data=trdat, cOutcome=cOutcome, predRisk=predRisk)')

	print "Ctrl+C to close"

	while True:
		time.sleep(0.1)