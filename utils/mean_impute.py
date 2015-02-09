"""
This script imputes the missing values using means.
It should not be used in practice! Just for dev purposes.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, csv
import numpy as np
import pandas as pd

"""Creates a complete output file by imputing the missing values using the mean value for
each column.

:param in_filename: input file with all the data
:param out_filename: output file with only complete rows
"""
def process(in_filename, out_filename):
    print "Imputing missing values in",in_filename
    training = pd.DataFrame.from_csv(in_filename)
    training.reset_index(inplace=True)

    mean_training = training.replace('?', '0')
    mean_training = mean_training.applymap(lambda x:float(x))
    means = mean_training.mean()

    training.replace('?', means, inplace=True)
    print "Writing complete data to",out_filename
    training.to_csv(out_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=1, default=["./data/training-data.csv"],
                        help="name of input training file")
    parser.add_argument('-o', '--output', nargs=1, default=["./data/training-data-completed.csv"],
                        help="name of output training file afer mean imputation")
    args = parser.parse_args()
    process(args.input[0], args.output[0])
