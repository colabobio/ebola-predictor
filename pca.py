"""
Transforms all training/testing pairs applying the PCA transformation as derived from all
complete data, using the specified number of PCA components.

Careful: original files are overwritten!!

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, glob
sys.path.append(os.path.abspath('./utils'))
from pca_transform import do_pca

test_files = glob.glob("./data/testing-data-*.csv")
for testfile in test_files:
    start_idx = testfile.find("./data/testing-data-") + len("./data/testing-data-")
    stop_idx = testfile.find('.csv')
    id = testfile[start_idx:stop_idx]
    trainfile = "./data/training-data-completed-" + str(id) + ".csv"
    if os.path.exists(testfile) and os.path.exists(trainfile):
        do_pca(testfile, trainfile, 6)