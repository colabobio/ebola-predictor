"""
Calculates ....

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, sys, os
import numpy as np
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind
from utils import load_dataframe, design_matrix
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix as design_matrix_full

def calculate(test_filename):
    df = load_dataframe(test_filename)
    X, y = design_matrix(df)

    surival_scores = df[df["OUT"] == 0]["SCORE"]
    death_scores = df[df["OUT"] == 1]["SCORE"]
    
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    (tval, pval) = ttest_ind(surival_scores, death_scores, equal_var=False)
        
    print np.mean(surival_scores),np.std(surival_scores)
    print np.mean(death_scores),np.std(death_scores)
    print "P-value:",pval

    
#     predictor = gen_predictor(cutoff)
#     probs = predictor(X)
#     indices = get_misses(probs, y)
#     for i in indices:
#         print "----------------"
#         if meta: print "META:",",".join(lines[i].split(",")).strip()
#         print "SCORE:", df.ix[i][1]
#         print df0.ix[i]
#     return indices

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test", nargs='?', default="./data/testing-data.csv",
                        help="test file")

    args = parser.parse_args()
    calculate(args.test)
