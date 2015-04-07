"""
This script imputes the missing values using Amelia in R, and creates and aggregated 
training dataset using all the imputed datasets from Amelia.

http://gking.harvard.edu/amelia

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse
import rpy2.robjects as robjects
from imputation import load_variables, load_bounds, aggregate_files

"""Creates a complete training set by imputing missing values using Amelia

:param in_filename: input file with all the data
:param out_filename: output file with only complete rows
:param kwparams: optional arguments for Amelia: num_imputed (number of imputed dataframes),
                 num_resamples (number of resamples), in_check (enable/disable input check),
                 gen_plots (enable/disable imputation plots)
"""
def process(in_filename, out_filename, **kwparams):
    if "num_imputed" in kwparams:
        num_imputed = int(kwparams["num_imputed"])
    else:
        num_imputed = 5

    if "num_resamples" in kwparams:
        resamples_opt = int(kwparams["num_resamples"])
    else:
        resamples_opt = 10000

    if "max_iter" in kwparams:
        max_iter = int(kwparams["max_iter"])
    else:
        max_iter = 10000

    if "in_check" in kwparams:
        incheck_opt = True if kwparams["in_check"].lower() == "true" else False
    else:
        incheck_opt = False

    if "gen_plots" in kwparams:
        gen_plots = True if kwparams["gen_plots"].lower() == "true" else False
    else:
        gen_plots = False

    var_names, var_types = load_variables(in_filename)
    bounds = load_bounds(in_filename, var_names, var_types)
    nom_rstr = ''
    for name in var_names:
        if var_types[name] == "category":
            if nom_rstr: nom_rstr = nom_rstr + ', '
            nom_rstr = nom_rstr + '"' + name + '"'

    # Construct bounds for the numerical variables
    # See section 4.6.3 in
    # http://r.iq.harvard.edu/docs/amelia/amelia.pdf
    # Matrix construction in R:
    # http://www.r-tutor.com/r-introduction/matrix/matrix-construction
    idx_str = ""
    min_str = ""
    max_str = ""
    num_vars = 0
    for i in range(0, len(var_names)):
        name = var_names[i]
        if var_types[name] != "category":
            num_vars = num_vars + 1
            if idx_str: 
                idx_str = idx_str + ", "
                min_str = min_str + ", "
                max_str = max_str + ", "
            idx_str = idx_str + str(i + 1)
            min_str = min_str + str(bounds[i][0])
            max_str = max_str + str(bounds[i][1])
    bds_str = idx_str + ", " + min_str + ", " + max_str
    robjects.r('num_bounds <- matrix(c(' + bds_str + '), nrow=' + str(num_vars) +', ncol=3)')

    print "Generating " + str(num_imputed) + " imputed datasets with Amelia..."
    robjects.r('library(Amelia)')
    robjects.r('trdat <- read.table("' + in_filename + '", sep=",", header=TRUE, na.strings="?")')
    robjects.r('nom_vars = c(' + nom_rstr + ')')

    if incheck_opt:
         incheck_str = "TRUE"
    else:
         incheck_str = "FALSE"

    dir, = os.path.split(in_filename)
    tmp_prefix = os.path.join(dir, "temp-data-amelia-")

    robjects.r('imdat <- amelia(trdat, m=' + str(num_imputed) + ', noms=nom_vars, bounds=num_bounds, max.resample = ' + str(resamples_opt) + ', incheck=' + incheck_str + ', emburn = c(5,' + str(max_iter) +'))')
    robjects.r('write.amelia(obj=imdat, file.stem="' + tmp_prefix + '", format="csv", row.names=FALSE)')
    
    if gen_plots:
        if not os.path.exists("./out"): os.makedirs("./out")
        robjects.r('pdf("./out/missingness.pdf", useDingbats=FALSE)')
        robjects.r('missmap(imdat)')
        robjects.r('dev.off()')
        for i in range(0, len(var_names)):
            name = var_names[i]
            # Compare observed density with imputed density
            robjects.r('pdf("./out/obs-vs-imp-' + name+ '.pdf", useDingbats=FALSE)')
            robjects.r('compare.density(imdat, var = "' + name + '")')
            robjects.r('dev.off()')
            if var_types[name] != "category":
                # Numerical variable, we can generate the quality of imputation plot
                robjects.r('pdf("./out/quality-imp-' + name+ '.pdf", useDingbats=FALSE)')
                robjects.r('overimpute(imdat, var = "' + name + '")')
                robjects.r('dev.off()')
        print "Saved Amelia plots to out folder"

    print "Success!"
    imp_files = [tmp_prefix + str(i) + ".csv" for i in range(1, num_imputed + 1)]
    aggregate_files(out_filename, imp_files, var_names, var_types, bounds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", nargs=1, default=["./models/test/training-data.csv"],
                        help="name of input training file")
    parser.add_argument("-o", "--output", nargs=1, default=["./models/test/training-data-completed.csv"],
                        help="name of output training file afer imputation")
    parser.add_argument("-n", "--num_imputed", type=int, nargs=1, default=[5],
                        help="number of imputed datasets")
    parser.add_argument("-r", "--num_resamples", type=int, nargs=1, default=[10000],
                        help="number of resamples")
    parser.add_argument("-x", "--max_iter", type=int, nargs=1, default=[10000],
                        help="maximum number of EM iterations")
    parser.add_argument("-c", "--in_check", action="store_true",
                        help="check input data")
    parser.add_argument("-p", "--gen_plots", action="store_true",
                        help="generate plots")

    args = parser.parse_args()
    process(in_filename=args.input[0], out_filename=args.output[0],
            num_imputed=str(args.num_imputed[0]),
            resamples_opt=str(args.num_resamples[0]),
            max_iter=str(args.max_iter[0]),
            incheck_opt=str(args.in_check),
            gen_plots=str(args.gen_plots))