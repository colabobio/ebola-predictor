"""
Trains all the predictors and saves the parameters to the data folder for later 
evaluation.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse, glob
from importlib import import_module

def train(predictor):
    module_path = os.path.abspath(predictor)
    module_filename = "train"
    sys.path.insert(0, module_path)
    module = import_module(module_filename)

    train_files = glob.glob("./data/training-data-completed-*.csv")

    # remove old parameters
    param_files = glob.glob("./data/" + module.prefix() + "-params-*")
    if param_files:
        print "Removing old " + module.title() + " parameters..."
        for file in param_files:
            os.remove(file)
        print "Done."

    print "Training " + module.title() + " predictor..."
    for tfile in train_files:
        print "Training set: " + tfile + "..."
        start_idx = tfile.find("training-data-completed-") + len("training-data-completed-")
        stop_idx = tfile.find(".csv")
        id = tfile[start_idx:stop_idx]
        module.train(train_filename=tfile, param_filename="./data/" + module.prefix() + "-params-" + str(id))
    print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('pred', nargs=1, default=["nnet"], help="Folder containing predictor to evaluate")
    args = parser.parse_args()
    train(args.pred[0])
