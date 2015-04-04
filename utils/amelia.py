"""
This script imputes the missing values using Amelia in R, and creates and aggregated 
training dataset using all the imputed datasets from Amelia.

http://gking.harvard.edu/amelia

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, csv, argparse
import rpy2.robjects as robjects

var_file = "./data/variables.txt"

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

    model_variables = []
    var_types = {}
    nom_rstr = ''
    with open(var_file, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            [name, type] = line.split()[0:2]
            model_variables.append(name)
            var_types[name] = type == "category"
            if var_types[name]:
                if nom_rstr: nom_rstr = nom_rstr + ', '
                nom_rstr = nom_rstr + '"' + name + '"'

    # Extract bounds from data
    bounds = [[1000, 0] for x in model_variables]
    with open(in_filename, "rb") as tfile:
        reader = csv.reader(tfile, delimiter=",")
        reader.next()
        for row in reader:
            for i in range(0, len(row)):
                if row[i] == "?": continue
                val = float(row[i])
                bounds[i][0] = min(val, bounds[i][0])
                bounds[i][1] = max(val, bounds[i][1])

    # Expand the bounds a bit...
    tol = 0.0
    for i in range(0, len(model_variables)):
        if var_types[model_variables[i]]: continue
        bounds[i][0] = (1 - tol) * bounds[i][0]
        bounds[i][1] = (1 + tol) * bounds[i][1]

    # Construct bounds for the numerical variables
    # See section 4.6.3 in
    # http://r.iq.harvard.edu/docs/amelia/amelia.pdf
    # Matrix construction in R:
    # http://www.r-tutor.com/r-introduction/matrix/matrix-construction
    idx_str = ""
    min_str = ""
    max_str = ""
    num_vars = 0
    for i in range(0, len(model_variables)):
        name = model_variables[i]
        if not var_types[name]:
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

    robjects.r('imdat <- amelia(trdat, m=' + str(num_imputed) + ', noms=nom_vars, bounds=num_bounds, max.resample = ' + str(resamples_opt) + ', incheck=' + incheck_str + ', emburn = c(5,' + str(max_iter) +'))')
    robjects.r('write.amelia(obj=imdat, file.stem="./temp/training-data-", format="csv", row.names=FALSE)')
    
    if gen_plots:
        if not os.path.exists("./out"): os.makedirs("./out")
        robjects.r('pdf("./out/missingness.pdf", useDingbats=FALSE)')
        robjects.r('missmap(imdat)')
        robjects.r('dev.off()')
        for i in range(0, len(model_variables)):
            name = model_variables[i]
            # Compare observed density with imputed density
            robjects.r('pdf("./out/obs-vs-imp-' + name+ '.pdf", useDingbats=FALSE)')
            robjects.r('compare.density(imdat, var = "' + name + '")')
            robjects.r('dev.off()')
            if not var_types[name]:
                # Numerical variable, we can generate the quality of imputation plot
                robjects.r('pdf("./out/quality-imp-' + name+ '.pdf", useDingbats=FALSE)')
                robjects.r('overimpute(imdat, var = "' + name + '")')
                robjects.r('dev.off()')
        print "Saved Amelia plots to out folder"

    print "Success!"

    print "Aggregating imputed datasets..."
    aggregated_data = []
    for i in range(1, num_imputed + 1):
        filename = "./temp/training-data-" + str(i) + ".csv"
        print "  Reading " + filename
        with open(filename, "rb") as ifile:
            reader = csv.reader(ifile, delimiter=",")
            reader.next()
            for row in reader:
                if row[0] == "NA": 
                    print "    Empty dataset, skipping!"
                    break
                add = True
                for i in range(0, len(row)):
                    name = model_variables[i]
                    if not var_types[name]:
                        val = float(row[i])
                        if val < bounds[i][0] or bounds[i][1] < val:
                            print "    Value " + row[i] + " for variable " + name + " is out of bounds [" + str(bounds[i][0]) + ", " + str(bounds[i][1]) + "], skipping"
                            add = False
                            break
                if add: aggregated_data.append(row)

    if aggregated_data:
        with open(out_filename, "wb") as trfile:
            writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(model_variables)
            for row in aggregated_data:
                writer.writerow(row)
        print "Saved aggregated imputed datasets to", out_filename

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