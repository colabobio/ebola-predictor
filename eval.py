import argparse, glob, os
from nnet.eval import eval as nnet_eval

predictors = ["nnet"]

def avg_cal_dis():
    print "1"

def cal_plots():
    print "2"

def avg_report():
    test_files = glob.glob("./data/testing-data-*.csv")
    for pred in predictors:
        print "Calculating average report for " + pred + "..."
        count = 0
        for testfile in test_files:
            start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
            stop_idx = testfile.find('.csv')
            id = testfile[start_idx:stop_idx]
            pfile = "./data/" + pred + "-params-" + str(id)
            trainfile = "./data/training-data-completed-" + str(id) + ".csv"
            if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
                count = count + 1
                print id, testfile, pfile, trainfile
                p, r, f, _ = nnet_eval(testfile, trainfile, pfile, 3)

    print "report"

def roc_plots():
    print "4"

def avg_conf_mat():
    print "5"

def evaluate(methods):
    for method in methods:
        # Average calibrations and discriminations
        if method == "cd":
            avg_cal_dis()
        # Plot each method on same calibration plot
        elif method == "calibration":
            cal_plots()
        # Average precision, recall, and F1 scores
        elif method == "report":
            avg_report()
        # Plot each method on same ROC plot
        elif method == "roc":
            roc_plots()
        # Average confusion matrix
        elif method == "confusion":
            avg_conf_mat()
        # Method not defined:
        else:
            raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-e', nargs='+', default=["confusion"], help="Supported evaluation methods: roc, cd, report, calibration, confusion")
    args = parser.parse_args()
    evaluate(args.e)