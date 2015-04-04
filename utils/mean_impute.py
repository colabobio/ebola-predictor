"""
This script imputes the missing values using means.
It should not be used in practice! Just for dev purposes.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, csv, os
import numpy as np
import pandas as pd

var_file = "./data/variables.txt"

"""Creates a complete output file by imputing the missing values using the mean value for
each column.

:param in_filename: input file with all the data
:param out_filename: output file with only complete rows
"""
def process(in_filename, out_filename):
    print "Imputing missing values in",in_filename
    training = pd.DataFrame.from_csv(in_filename)
    training.reset_index(inplace=True)
    
    dir = os.path.split(in_filename)[0]
    if os.path.exists(dir + "/variables.txt"):
        fn = dir + "/variables.txt"
    else:
        fn = var_file
    categorical_vars = []
    float_vars = []
    with open(fn, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            [name, type] = line.split()[0:2]
            if type == "category":
                categorical_vars.append(unicode(name))
            else:
                float_vars.append(unicode(name))
                
    mean_training = training.replace('?', '0')
    mean_training = mean_training.applymap(lambda x:float(x))
    means = mean_training.mean()
    float_means = means[float_vars]
    cat_means = means[categorical_vars].round()
    combined_means = pd.concat([float_means, cat_means])
        
    training.replace('?', combined_means, inplace=True)
    print "Writing complete data to",out_filename
    training.to_csv(out_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=1, default=["./models/test/training-data.csv"],
                        help="name of input training file")
    parser.add_argument('-o', '--output', nargs=1, default=["./models/test/training-data-completed.csv"],
                        help="name of output training file afer mean imputation")
    args = parser.parse_args()
    process(args.input[0], args.output[0])
