import argparse, glob, os, sys
from matplotlib import pyplot as plt
from importlib import import_module

label_file = "./data/labels.txt"
target_names = []
with open(label_file, "rb") as vfile:
    for line in vfile.readlines():
        target_names.append(line.split()[1])

def avg_cal_dis(module):
    print "1"

def cal_plots(module):
    print "2"

def avg_report(module):
    test_files = glob.glob("./data/testing-data-*.csv")
    print "Calculating average report for " + module.title() + "..."
    count = 0
    total_prec = 0
    total_rec = 0
    total_f1 = 0
    for testfile in test_files:
        start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = "./data/" + module.prefix() + "-params-" + str(id)
        trainfile = "./data/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            count = count + 1
            print "Report for test set " + id + " ----------------------------------"
            p, r, f, _ = module.eval(testfile, trainfile, pfile, 3)
            total_prec += p
            total_rec += r
            total_f1 += f

    avg_prec = (total_prec)/(count)
    avg_rec = (total_rec)/(count)
    avg_f1 = (total_f1)/(count)

    print "Average report for " + module.title() + " ********************************************"
    print "{:10s} {:10s} {:10s} {:10s}".format("", "precision", "recall", "f1-score")
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[0], avg_prec[0], avg_rec[0], avg_f1[0])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[1], avg_prec[1], avg_rec[1], avg_f1[1])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format("Total", (avg_prec[0] + avg_prec[1])/2, (avg_rec[0]+avg_rec[1])/2, (avg_f1[0]+avg_f1[1])/2)

def roc_plots(module):
    plt.clf()

    test_files = glob.glob("./data/testing-data-*.csv")
    print "Calculating ROC curves for " + module.title() + "..."
    count = 0
    total_roc_auc = 0
    
    for testfile in test_files:
        start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = "./data/" + module.prefix() + "-params-" + str(id)
        trainfile = "./data/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            print "Report for test set " + id + " ----------------------------------"
            count = count + 1
            total_roc_auc += module.eval(testfile, trainfile, pfile, 4)
    ave_roc_auc = (total_roc_auc)/(count)
    print "Average area under the ROC curve: " + str(ave_roc_auc)
    plt.show()

def avg_conf_mat(module):
    print "5"

def list_misses(module):
    test_files = glob.glob("./data/testing-data-*.csv")
    print "Miss-classifications for predictor " + module.title() + "..."
    count = 0
    for testfile in test_files:
        start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = "./data/" + module.prefix() + "-params-" + str(id)
        trainfile = "./data/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            count = count + 1
            module.miss(testfile, trainfile, pfile)
    print "Total miss-classifications:",count

def evaluate(predictor, methods):
    module_path = os.path.abspath(predictor)
    module_filename = "eval"
    sys.path.insert(0, module_path)
    module = import_module(module_filename)

    for method in methods:
        # Average calibrations and discriminations
        if method == "cd":
            avg_cal_dis(module)
        # Plot each method on same calibration plot
        elif method == "calibration":
            cal_plots(module)
        # Average precision, recall, and F1 scores
        elif method == "report":
            avg_report(module)
        # Plot each method on same ROC plot
        elif method == "roc":
            roc_plots(module)
        # Average confusion matrix
        elif method == "confusion":
            avg_conf_mat(module)
        elif method == "misses":
            list_misses(module)
        # Method not defined:
        else:
            raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-p', nargs=1, default=["nnet"], help="Folder containing predictor to evaluate")
    parser.add_argument('-e', nargs='+', default=["confusion"], help="Supported evaluation methods: roc, cd, report, calibration, confusion, misses")
    args = parser.parse_args()
    evaluate(args.p[0], args.e) 