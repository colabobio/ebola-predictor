"""

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv

training_file = "./data/training-data.csv"
out_file = "./data/training-data-imputed.csv"

def remove():
    titles = []
    data = []
    with open(training_file, "rb") as ifile:
        reader = csv.reader(ifile)        
        titles = reader.next()
        for row in reader:
            if "?" in row: continue
            data.append(row)
    
# 	print "Aggregating imputed datasets..."
# 	aggregated_data = []
# 	for i in range(1, num_imputed + 1):
# 		filename = "./data/training-data-" + str(i) + ".csv"
# 		print "  Reading " + filename
# 		with open(filename, "rb") as ifile:
# 			reader = csv.reader(ifile, delimiter=",")
# 			reader.next()
# 			for row in reader:
# 				if row[0] == "NA": 
# 					print "    Empty dataset, skipping!"
# 					break
# 				add = True    
# 				for i in range(0, len(row)):
# 					name = model_variables[i]
# 					if not var_types[name]:
# 						val = float(row[i])
# 						if val < bounds[i][0] or bounds[i][1] < val:
# 							print "    Value " + row[i] + " for variable " + name + " is out of bounds [" + str(bounds[i][0]) + ", " + str(bounds[i][1]) + "], skipping"
# 							add = False
# 							break
# 				if add: aggregated_data.append(row)
	
    with open(out_file, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for row in data:
            writer.writerow(row)

    print "Done."

if __name__ == "__main__":
    remove()