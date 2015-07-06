import argparse, glob, os, sys, csv
from importlib import import_module
import numpy as np

def aggregate_model(mdl_dir, out_file, module):
    test_files = glob.glob(mdl_dir + "/testing-data-*.csv")
    all_prob = []
    all_y = []
    for testfile in test_files:
        start_idx = testfile.find(mdl_dir + "/testing-data-") + len(mdl_dir + "/testing-data-")
        stop_idx = testfile.find('.csv')
        id = testfile[start_idx:stop_idx]
        pfile = mdl_dir + "/" + module.prefix() + "-params-" + str(id)
        trainfile = mdl_dir + "/training-data-completed-" + str(id) + ".csv"
        if os.path.exists(testfile) and os.path.exists(pfile) and os.path.exists(trainfile):
            p, y = module.pred(testfile, trainfile, pfile)
            all_prob.extend(p)
            all_y.extend(y)

    with open(out_file, "wb") as ofile:
        writer = csv.writer(ofile, delimiter=",")
        writer.writerow(["Y", "P"])
        for i in range(0, len(all_prob)):
            writer.writerow([all_y[i], all_prob[i]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                        help="Base directory")
    parser.add_argument('-N', '--name', nargs=1, default=["test"],
                        help="Model name")
    parser.add_argument('-p', '--predictor', nargs=1, default=["nnet"], 
                        help="Folder containing predictor to evaluate")
    parser.add_argument('-out', '--out_file', nargs=1, default=["./out/predictions.csv"],
                        help="File storing aggregated predictions")
    args = parser.parse_args()
    
    module_path = os.path.abspath(args.predictor[0])
    module_filename = "eval"
    sys.path.insert(0, module_path)
    module = import_module(module_filename)

    base_dir = args.base_dir[0]
    name = args.name[0]
    out_file = args.out_file[0]
    mdl_dir = os.path.join(base_dir, "models", name)
    
    aggregate_model(mdl_dir, out_file, module)