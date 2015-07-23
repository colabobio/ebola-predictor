import argparse, glob, os, sys, csv
import numpy as np
from matplotlib import pyplot as plt
from importlib import import_module
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score

label_file = "./data/outcome.txt"
target_names = []
with open(label_file, "rb") as vfile:
    for line in vfile.readlines():
        target_names.append(line.split(',')[1])

def avg_cal_dis(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Calculating Calibration/Discrimination for " + module.title() + "..."
#     count = 0
    total_cal = []
    total_dis = []
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
#             count = count + 1
            print "Calibration/Discrimination for test set " + id + " ----------------------------------"
            cal, dis = module.eval(testfile, trainfile, pfile, 1)
#             total_cal += cal
#             total_dis += dis
            total_cal.append(cal)
            total_dis.append(dis)

    avg_cal = np.mean(np.array(total_cal), axis=0)
    avg_dis = np.mean(np.array(total_dis), axis=0)
    std_cal = np.std(np.array(total_cal), axis=0)
    std_dis = np.std(np.array(total_dis), axis=0)

    print module.title() + " Calibration/Discrimination Average ********************************************"
    print "Calibration   : " + str(avg_cal)
    print "Discrimination: " + str(avg_dis)
    print module.title() + " Calibration/Discrimination Error ********************************************"
    print "Calibration   : " + str(std_cal)
    print "Discrimination: " + str(std_dis)

def cal_plots(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Calculating calibration plots for " + module.title() + "..."
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            print "Calibration for test set " + id + " ----------------------------------"
            out_file = "./out/calstats-" + id + ".txt"
            plot_file = "./out/calplot-" + id + ".pdf"
            module.eval(testfile, trainfile, pfile, 2, test_file=testfile, out_file=out_file, plot_file=plot_file)
    print "********************************************"
    print "Saved calibration plot and Hosmer-Lemeshow goodness of fit for " + module.title() + " in out folder."

def avg_report(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Calculating average report for " + module.title() + "..."
    count = 0
    total_prec = []
    total_rec = []
    total_f1 = []
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            count = count + 1
            print "Report for test set " + id + " ----------------------------------"
            p, r, f, _ = module.eval(testfile, trainfile, pfile, 3)
            total_prec.append(p)
            total_rec.append(r)
            total_f1.append(f)

    avg_prec = np.mean(np.array(total_prec), axis=0)
    avg_rec = np.mean(np.array(total_rec), axis=0)
    avg_f1 = np.mean(np.array(total_f1), axis=0)

    std_prec = np.std(np.array(total_prec), axis=0)
    std_rec = np.std(np.array(total_rec), axis=0)
    std_f1 = np.std(np.array(total_f1), axis=0)

    tot_prec_mean = (avg_prec[0] + avg_prec[1])/2
    tot_rec_mean = (avg_rec[0]+avg_rec[1])/2
    tot_f1_mean = (avg_f1[0]+avg_f1[1])/2

    tot_prec_std = (std_prec[0] + std_prec[1])/2 
    tot_rec_std = (std_rec[0]+std_rec[1])/2
    tot_f1_std = (std_f1[0]+std_f1[1])/2

    print "Average report for " + module.title() + " ********************************************"
    print "{:10s} {:10s} {:10s} {:10s}".format("", "precision", "recall", "f1-score")
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[0], avg_prec[0], avg_rec[0], avg_f1[0])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[1], avg_prec[1], avg_rec[1], avg_f1[1])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format("Total", tot_prec_mean, tot_rec_mean, tot_f1_mean)
    print
    print "Standard Deviation report for " + module.title() + " ********************************************"
    print "{:10s} {:10s} {:10s} {:10s}".format("", "precision", "recall", "f1-score")
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[0], std_prec[0], std_rec[0], std_f1[0])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format(target_names[1], std_prec[1], std_rec[1], std_f1[1])
    print "{:10s}      {:2.2f}    {:2.2f}        {:2.2f}".format("Total", tot_prec_std, tot_rec_std, tot_f1_std)
    print
    print "Summary ********************************************"
    print "Total,"+str(tot_prec_mean)+","+str(tot_rec_mean)+","+str(tot_f1_mean)+","+str(tot_prec_std)+","+str(tot_rec_std)+","+str(tot_f1_std)

def roc_plots(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Calculating ROC curves for " + module.title() + "..."
    count = 0
    all_prob = []
    all_y = []
    total_fpr = np.array([])
    total_tpr = np.array([])
    total_auc = 0
    plt.clf()
    fig = plt.figure()
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            print "Report for test set " + id + " ----------------------------------"
            fpr, tpr, auc = module.eval(testfile, trainfile, pfile, 4, pltshow=False)       
            p, y = module.pred(testfile, trainfile, pfile)
            all_prob.extend(p)
            all_y.extend(y)
#             if fpr.size < 3: continue
            if total_fpr.size:
                if total_fpr[0].size != fpr.size: continue
                total_fpr = np.append(total_fpr, np.array([fpr]), axis=0)
                total_tpr = np.append(total_tpr, np.array([tpr]), axis=0)
            else:
                total_fpr = np.array([fpr])
                total_tpr = np.array([tpr])
            total_auc += auc
            count += 1
    ave_auc = (total_auc)/(count)
    print "********************************************"
    ave_fpr = np.mean(total_fpr, axis=0)
    ave_tpr = np.mean(total_tpr, axis=0)
#     std_fpr = np.std(total_fpr, axis=0)
    std_tpr = np.std(total_tpr, axis=0)

    # The AUC of the aggregated curve
    all_auc = roc_auc_score(all_y, all_prob)
    
#    f2 = interp1d(ave_fpr, ave_tpr, kind='cubic')
    plt.plot(ave_fpr, ave_tpr, c="grey")
#     print f2(ave_fpr)
#     plt.plot(ave_fpr, f2(ave_fpr), c="red")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.fill_between(ave_fpr, ave_tpr-std_tpr, ave_tpr+std_tpr, alpha=0.5)
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    print "Average area under the ROC curve for " + module.title() + ": " + str(ave_auc)
    print "Area under the aggregated ROC curve for " + module.title() + ": " + str(all_auc)    
    fig.savefig('./out/roc.pdf')
    print "Saved ROC curve to ./out/roc.pdf"
    
    with open("./out/roc.csv", "wb") as rfile:
        writer = csv.writer(rfile, delimiter=",")
        writer.writerow(["Y", "P"])
        for i in range(0, len(all_prob)):
            writer.writerow([all_y[i], all_prob[i]])
    print "Saved aggregated ROC data to ./out/roc.csv"

def avg_conf_mat(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Calculating average report for " + module.title() + "..."
    count = 0
    total_n_hit = 0
    total_n_false_alarm = 0
    total_n_miss = 0
    total_n_correct_rej = 0
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            count = count + 1
            print "Confusion matrix for test set " + id + " ------------------------------"
            n_hit, n_false_alarm, n_miss, n_correct_rej = module.eval(testfile, trainfile, pfile, 5)
            total_n_hit += n_hit
            total_n_false_alarm += n_false_alarm
            total_n_miss += n_miss
            total_n_correct_rej += n_correct_rej

    avg_n_hit = total_n_hit/(1.0*count)
    avg_n_false_alarm = total_n_false_alarm/(1.0*count)
    avg_n_miss = total_n_miss/(1.0*count)
    avg_n_correct_rej = total_n_correct_rej/(1.0*count)
     
    print "Average confusion matrix for " + module.title() + " ********************************************"
    print "{:25s} {:20s} {:20s}".format("", "Output " + target_names[1], "Output " + target_names[0])
    print "{:25s} {:2.2f}{:17s}{:2.2f}".format("Predicted " + target_names[1], avg_n_hit,"", avg_n_false_alarm)
    print "{:25s} {:2.2f}{:17s}{:2.2f}".format("Predicted " + target_names[0], avg_n_miss,"", avg_n_correct_rej) 

def list_misses(dir, module):
    test_files = glob.glob(dir + "/testing-data-*.csv")
    print "Miss-classifications for predictor " + module.title() + "..."
    count = 0
    for testfile in test_files:
        start_idx = testfile.find(dir + "/testing-data-") + len(dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            idx = module.miss(testfile, trainfile, pfile)
            count += len(idx)
    print "********************************************"
    print "Total miss-classifications for " + module.title() + ":",count

def evaluate(base, name, predictor, method):
    dir =  os.path.join(base, "models", name)

    module_path = os.path.abspath(predictor)
    module_filename = "eval"
    sys.path.insert(0, module_path)
    module = import_module(module_filename)

    if not os.path.exists("./out"): os.makedirs("./out")

    # Average calibrations and discriminations
    if method == "caldis":
        avg_cal_dis(dir, module)
    # Plot each method on same calibration plot
    elif method == "calplot":
        cal_plots(dir, module)
    # Average precision, recall, and F1 scores
    elif method == "report":
        avg_report(dir, module)
    # Plot each method on same ROC plot
    elif method == "roc":
        roc_plots(dir, module)
    # Average confusion matrix
    elif method == "confusion":
       avg_conf_mat(dir, module)
    elif method == "misses":
        list_misses(dir, module)
    # Method not defined:
    else:
        raise Exception("Invalid method given")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                        help="Base directory")
    parser.add_argument('-N', '--name', nargs=1, default=["test"],
                        help="Model name")
    parser.add_argument('-p', '--predictor', nargs=1, default=["nnet"], 
                        help="Folder containing predictor to evaluate")
    parser.add_argument('-m', '--method', nargs=1, default=["report"], 
                        help="Evaluation method: caldis, calplot, report, roc, confusion, misses")
    args = parser.parse_args()
    evaluate(args.base_dir[0], args.name[0], args.predictor[0], args.method[0])