"""
Creates calibration plot and Hosmer-Lemeshow statistics.
Main R package from http://www.genabel.org/PredictABEL/plotCalibration.html
"""

import rpy2.robjects as robjects

test_file = "./data/testing-data.csv"

def calplot(probs, y_test, **kwparams):
    if "color" in kwparams:
        color = kwparams["color"]
    else:
        color = "black"
        
    if "plot_file" in kwparams:
        plot_file = kwparams["plot_file"]
    else:
        plot_file = "./out/calplot.pdf"

    if "out_file" in kwparams:
        out_file = kwparams["out_file"]
    else:
        out_file = "./out/calstats.txt"

    test_file = kwparams["test_file"]

    robjects.r('pdf("'+ plot_file +'", useDingbats=FALSE)') 
    predrisk = 'c('+ ', '.join([str(p) for p in probs])+')'

    robjects.r('par(new=TRUE)')
    robjects.r('par(col="'+color+'")')

    robjects.r('library(PredictABEL)')
    robjects.r('trdat <- read.table("' + test_file + '", sep = ",", header=TRUE)')
    robjects.r('cOutcome <- 1')
    robjects.r('predRisk <- '+predrisk)
    robjects.r('plotCalibration(data=trdat, cOutcome=cOutcome, predRisk=predRisk, filename="' + out_file + '")')
    robjects.r('dev.off()')

    print "Saved calibration plot to:           ",plot_file
    print "Saved Hosmer-Lemeshow statistics to: ",out_file
