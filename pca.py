"""
Transforms all training/testing pairs applying the PCA transformation as derived from all
complete data, using the specified number of PCA components.

Careful: original files are overwritten!!

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, glob
sys.path.append(os.path.abspath('./utils'))
from pca_transform import do_pca

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument('-N', '--name', nargs=1, default=["test"],
                    help="Model name")
args = parser.parse_args()

base_dir = args.base_dir[0]
mdl_name = args.name[0]
mdl_dir =  os.path.join(base_dir, "models", mdl_name)

test_files = glob.glob(mdl_dir + "/testing-data-*.csv")
for testfile in test_files:
    start_idx = testfile.find(mdl_dir + "/testing-data-") + len(mdl_dir + "/testing-data-")
    stop_idx = testfile.find('.csv')
    id = testfile[start_idx:stop_idx]
    trainfile = mdl_dir + "/training-data-completed-" + str(id) + ".csv"
    if os.path.exists(testfile) and os.path.exists(trainfile):
        do_pca(testfile, trainfile, 6)