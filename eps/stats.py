"""
Calculates a number of basic statistics for the EPS: T-test to check if the difference
between the mean score between surviving and deceased patients, and Fisher test for the
2x2 contingency table with outcome and EPS prediction.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, sys, os, csv
import numpy as np
import pandas as pd
from pandas.tools.pivot import pivot_table
from scipy.stats import fisher_exact
from scipy.stats import ttest_ind
from utils import load_dataframe, var_score, design_matrix
sys.path.append(os.path.abspath('./utils'))
from evaluate import design_matrix as design_matrix_full

def calculate(test_filename, def_thresh=None, score_name="EPS", case_filename="./out/cases.csv"):
    _, _, df0 = design_matrix_full(test_filename=test_filename, train_filename="", get_df=True)

    fn = test_filename.replace("-data", "-index")
    meta = None
    if os.path.exists(fn):
        with open(fn, "r") as idxfile:
            meta = idxfile.readlines()

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
    if not def_thresh == None: surv_thresh = def_thresh
    else: surv_thresh = np.mean(surival_scores) + np.std(surival_scores)
    df["PRED"] = df["SCORE"].map(lambda x: 0 if x <= surv_thresh else 1)
    print df
    df["VALUE"] = pd.Series(np.ones(df.shape[0]), index=df.index)
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.tools.pivot.pivot_table.html
    counts = pivot_table(df, values="VALUE", index=["OUT"], columns=["PRED"], aggfunc=np.sum, fill_value=0)
    # ...and performs a Fisher exact test on the 2x2 contingency table 
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
    fisher_ratio, fisher_pvalue = fisher_exact(counts)

    print "Survival cutoff:",surv_thresh
    print "Mean,standard deviation of survival score:", np.mean(surival_scores),"",np.std(surival_scores)
    print "Mean,standard deviation of death score   :", np.mean(death_scores),"",np.std(death_scores)
    print "P-value of T-test to for means of survival and death scores :",ttest_pvalue    
    print ""
    
    print "Observed outcome/EPS prediction contingency table"
    print counts
    print "P-Value of Fisher exact test on the Outcome/Prediction table:",fisher_pvalue

    with open(case_filename, "wb") as cfile:
        writer = csv.writer(cfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        titles = ["GID"]
        titles.extend(df0.columns.values[1:])
        titles.extend([score_name, "PRED_" + score_name, "OUT"])
        writer.writerow(titles)
        for i in range(0, df0.shape[0]):
            gid = meta[i].split(",")[1]
            row0 = df0.ix[i]
            row = df.ix[i]
            row1 = [gid]
            for name in row0.index[1:]:
                value = row0[name]
                row1.append(var_score(name, str(value)))
            row1.append(int(row["SCORE"]))
            row1.append(int(row["PRED"]))
            row1.append(int(row0["OUT"]))
            writer.writerow(row1)
    print ""
    print "Wrote cases to",case_filename

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--test', nargs=1, default=["./data/testing-data.csv"],
                        help="Filename for testing set")
    parser.add_argument('-o', '--out', nargs=1, default=["./out/cases.csv"],
                        help="Filename for cases file")
    parser.add_argument('-c', '--cutoff', nargs=1, type=int, default=[None],
                        help="Cutoff for EPS prediction")
    parser.add_argument('-n', '--name', nargs=1, default=["PRED"],
                        help="Predictor name")
    args = parser.parse_args()
    calculate(test_filename=args.test[0], def_thresh=args.cutoff[0], score_name=args.name[0], case_filename=args.out[0])
