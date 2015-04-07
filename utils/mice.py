"""
This script imputes the missing values using the MICE (Multivariate Imputation by
Chained Equations) package in R:

http://www.stefvanbuuren.nl/mi/MICE.htm
http://www.ats.ucla.edu/stat/r/faq/R_pmm_mi.html

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import rpy2.robjects as robjects
from imputation import load_variables, load_bounds, aggregate_files

"""Creates a complete training set by imputing missing values using MICE

:param in_filename: input file with all the data
:param out_filename: output file with only complete rows
:param kwparams: optional arguments for MICE: num_imputed (number of imputed dataframes)
"""
def process(in_filename, out_filename, **kwparams):
    if "num_imputed" in kwparams:
        num_imputed = int(kwparams["num_imputed"])
    else:
        num_imputed = 5

    var_names, var_types = load_variables(in_filename)
    bounds = load_bounds(in_filename, var_names, var_types)

    dir, = os.path.split(in_filename)
    tmp_filename = os.path.join(dir, "temp-data-mice.csv")

    print "Generating " + str(num_imputed) + " imputed datasets with MICE..."
    robjects.r('library(mice)') 
    robjects.r('trdat <- read.table("' + in_filename + '", sep=",", header=TRUE, na.strings="?")')
    robjects.r('imdat <- mice(trdat, m=' + str(num_imputed) + ')')
    robjects.r('codat <- complete(imdat, "long")')
    robjects.r('drops <- c(".imp",".id")')
    robjects.r('codat <- codat[,!(names(codat) %in% drops)]')
    robjects.r('write.csv(codat, file="' + tmp_filename + '", row.names=FALSE)')

    imp_files = [tmp_filename]
    aggregate_files(out_filename, imp_files, var_names, var_types, bounds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs=1, default=["./models/test/training-data.csv"],
                        help="name of input training file")
    parser.add_argument("-o", "--output", nargs=1, default=["./models/test/training-data-completed.csv"],
                        help="name of output training file afer imputation")
    parser.add_argument("-n", "--num_imputed", type=int, nargs=1, default=[5],
                        help="number of imputed datasets")

    args = parser.parse_args()
    process(in_filename=args.input[0], out_filename=args.output[0],
            num_imputed=str(args.num_imputed[0]))