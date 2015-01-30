"""
This script imputes the missing values using Amelia in R, and creates and aggregated 
training dataset using all the imputed datasets from Amelia.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, csv
import rpy2.robjects as robjects

var_file = "./data/variables.txt"
training_file = "./data/training-data.csv"
aggregated_file = "./data/training-data-imputed.csv"
incheck_opt = "FALSE"

def impute(num_imputed, training_file=training_file, aggregated_file=aggregated_file):
	
	model_variables = []
	var_types = {}
	nom_rstr = ''
	with open(var_file, "rb") as vfile:
		for line in vfile.readlines():
			[name, type] = line.split()[0:2]
			model_variables.append(name)
			var_types[name] = type == "category" 
			if var_types[name]:
				if nom_rstr: nom_rstr = nom_rstr + ', '
				nom_rstr = nom_rstr + '"' + name + '"'

	# Extract bounds from data
	bounds = [[1000, 0] for x in model_variables]
	with open(training_file, "rb") as tfile:
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

	robjects.r('library(Amelia)')
	robjects.r('trdat <- read.table("' + training_file + '", sep=",", header=TRUE, na.strings="?")')
	robjects.r('nom_vars = c(' + nom_rstr + ')')
	
	robjects.r('imdat <- amelia(trdat, m=' + str(num_imputed) + ', noms=nom_vars, bounds=num_bounds, max.resample = 10000, incheck=' + incheck_opt + ')')
	robjects.r('write.amelia(obj=imdat, file.stem="./data/training-data-", format="csv", row.names=FALSE)')        
        
	print "Aggregating imputed datasets..."
	aggregated_data = []
	for i in range(1, num_imputed + 1):
		filename = "./data/training-data-" + str(i) + ".csv"
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
	
	with open(aggregated_file, "wb") as trfile:
		writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
		writer.writerow(model_variables)
		for row in aggregated_data:
			writer.writerow(row)

	print "Done."    

if __name__ == "__main__":
	num_imputed = 10
	if 1 < len(sys.argv):
		num_imputed = int(sys.argv[1])
	impute(num_imputed)