"""
Trains all the predictors and saves the parameters to the data folder for later 
evaluation.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, argparse, glob
from importlib import import_module

def train(mdl_name, predictor, **kwparams):
    model_dir = "./models/" + mdl_name

    module_path = os.path.abspath(predictor)
    module_filename = "train"
    sys.path.insert(0, module_path)
    module = import_module(module_filename)

    train_files = glob.glob(model_dir + "/training-data-completed-*.csv")

    # remove old parameters
    param_files = glob.glob(model_dir + "/" + module.prefix() + "-params-*")
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
        module.train(train_filename=tfile, param_filename=model_dir + "/" + module.prefix() + "-params-" + str(id), **kwparams)
    print "Done."

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Evaluate the model with given method(s)
    parser.add_argument('-N', '--name', nargs=1, default=["test"],
                        help="Model name")
    parser.add_argument('pred', nargs=1, default=["nnet"],
                        help="Folder containing predictor to evaluate")
    parser.add_argument('vars', nargs='*')
    args = parser.parse_args()
    kwargs = {}
    for var in args.vars:
        [k, v] = var.split("=")
        kwargs[k] = v
    train(args.name[0], args.pred[0], **kwargs)
