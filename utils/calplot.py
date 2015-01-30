'''
Creates calibration plot and Hosmer-Lemeshow statistics.
Main R package from http://www.genabel.org/PredictABEL/plotCalibration.html
'''
import rpy2.robjects as robjects

test_file = "./data/testing-data.csv"

def calplot(probs, test_file=test_file, out_file = "./data/calibration.txt", color='red'):
	predrisk = 'c('+ ', '.join([str(p) for p in probs])+')'

	robjects.r('par(new=TRUE)')
	robjects.r('par(col="'+color+'")')

	robjects.r('library(PredictABEL)')
	robjects.r('trdat <- read.table("' + test_file + '", sep = ",", header=TRUE)')
	robjects.r('cOutcome <- 1')
	robjects.r('predRisk <- '+predrisk)
	robjects.r('plotCalibration(data=trdat, cOutcome=cOutcome, predRisk=predRisk, filename="' + out_file + '")')