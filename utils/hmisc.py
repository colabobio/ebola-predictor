"""
This script imputes the missing values using the Multiple Imputation method aregImpute from 
Hmisc:

http://www.inside-r.org/packages/cran/Hmisc/docs/aregImpute

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, csv, argparse
import rpy2.robjects as robjects
from imputation import load_variables, load_bounds, aggregate_files

"""Creates a complete training set by imputing missing values using aregImpute

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
    model_str = ""
    list_str = ""
    for name in var_names:
        if model_str: model_str += ' + '
        else: model_str = '~ '
        model_str += name
        if list_str: list_str += ', '
        list_str += '"' + name + '"'
    frame_str = ",".join(["comp$" + x for x in list_str.split(",")])
     
    tmp_prefix = "./temp/training-data-hmisc-"

    print "Generating " + str(num_imputed) + " imputed datasets with Hmisc..."
    robjects.r('library(Hmisc)') 
    robjects.r('trdat <- read.table("' + in_filename + '", sep=",", header=TRUE, na.strings="?")')
    robjects.r('imdat <- aregImpute(' + model_str + ', nk=c(0,3:5), tlinear=FALSE, data=trdat, n.impute=' + str(num_imputed) + ')')

    imp_files = [tmp_prefix + str(i) + ".csv" for i in range(1, num_imputed + 1)]
    for i in range(1, num_imputed + 1):
        robjects.r('comp <- impute.transcan(imdat, imputation=' + str(i)+ ', data=trdat, list.out=TRUE, pr=FALSE, check=FALSE)')
        robjects.r('df = data.frame(' + frame_str + ')')
        robjects.r('colnames(df) <- c(' + list_str + ')')
        robjects.r('write.csv(df, file="' + tmp_prefix + str(i) + '.csv", row.names=FALSE)')
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
