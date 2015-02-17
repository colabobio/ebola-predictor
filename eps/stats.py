"""
Calculates ....

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, sys, os
import numpy as np
import pandas as pd
from pandas.tools.pivot import pivot_table
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind
from utils import load_dataframe, design_matrix
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix as design_matrix_full

def calculate(test_filename, def_thresh=None):
    df = load_dataframe(test_filename)
    X, y = design_matrix(df)

    surival_scores = df[df["OUT"] == 0]["SCORE"]
    death_scores = df[df["OUT"] == 1]["SCORE"]
    
    # Test if the mean values of the scores among the recovered and deceased patients
    # are significantly different:
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
    ttest_stat, ttest_pvalue = ttest_ind(surival_scores, death_scores, equal_var=False)
        
    # Builds the 2x2 contingency table by defining the EPS prediction as:
    # Survive: score <= mean(survival) + std(survival)
    # Die: otherwise
    if def_thresh: surv_thresh = def_thresh
    else: surv_thresh = np.mean(surival_scores) + np.std(surival_scores) 
    df["PRED"] = df["SCORE"].map(lambda x: 0 if x <= surv_thresh else 1)
    print df    
    df["VALUE"] = pd.Series(np.ones(df.shape[0]), index=df.index)
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.tools.pivot.pivot_table.html
    counts = pivot_table(df, values="VALUE", index=["OUT"], columns=["PRED"], aggfunc=np.sum, fill_value=0)
    # ...and performs a Fisher exact test on the 2x2 contingency table 
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    fisher_ratio, fisher_pvalue = fisher_exact(counts)

    print "Mean,standard deviation of survival score:", np.mean(surival_scores),"",np.std(surival_scores)
    print "Mean,standard deviation of death score   :", np.mean(death_scores),"",np.std(death_scores)
    print "P-value of T-test to for means of survival and death scores :",ttest_pvalue    
    print ""
    
    print "Observed outcome/EPS prediction contingency table"
    print counts

    print "P-Value of Fisher exact test on the Outcome/Prediction table:",fisher_pvalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("test", nargs='?', default="./data/testing-data.csv",
                        help="test file")

    args = parser.parse_args()
    calculate(args.test)
