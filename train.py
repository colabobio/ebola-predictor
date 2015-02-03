"""
Trains all the predictors and saves the parameters to the data folder for later 
evaluation.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

from nnet.train import train as nnet_train

import os, argparse, glob

if __name__ == "__main__":
    train_files = glob.glob("./data/training-data-completed-*.csv")

    # remove old data
    nnet_param_files = glob.glob("./data/nnet-params-*")
    dtree_param_files = glob.glob("./data/dtree-params-*")
    if nnet_param_files or dtree_param_files:
        print "Removing old parameters..."
        for file in nnet_param_files:
            os.remove(file)
        for file in dtree_param_files:
            os.remove(file)
        print "Done."

    print "Training all predictors..."
    for tfile in train_files:
        print "Training set: " + tfile + "..."
        start_idx = tfile.find("training-data-completed-") + len("training-data-completed-")
        stop_idx = tfile.find(".csv")
        id = tfile[start_idx:stop_idx]
        nnet_train(train_filename=tfile, param_filename="./data/nnet-params-" + str(id))
    print "Done."
