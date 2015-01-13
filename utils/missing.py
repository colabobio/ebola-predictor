'''
This script simply gives information on the missingness in the input file.
'''

import csv

var_file = "./data/variables.txt"
input_file = "./data/profile-data.tsv"

model_variables = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        model_variables.append(line.split()[0])

total_count = 0
miss_dep_var = 0
complete_count = 0
missing_count = [0] * len(model_variables)

with open(input_file, "rb") as ifile:
    reader = csv.reader(ifile, delimiter="\t")
    titles = reader.next()
    model_idx = [titles.index(var) for var in model_variables]
    r = 0 
    for row in reader:
        all_missing = True
        some_missing = False
        missing_dvar = row[model_idx[0]] == "\\N"
        for i in range(1, len(model_variables)):
             var_idx = model_idx[i]
             if row[var_idx] == "\\N":
                 some_missing = True
             else:
                 all_missing = False
        if not all_missing and not missing_dvar:
            total_count = total_count + 1            
            if not some_missing: complete_count = complete_count + 1
            for i in range(0, len(model_variables)):
                var_idx = model_idx[i]
                if row[var_idx] == "\\N":
                    missing_count[i] = missing_count[i] + 1         

print "Total number of data rows   :",total_count        
print "Number of complete data rows:",complete_count

print "Missing counts and precentages for each independent variable:"
for i in range(1, len(model_variables)):
    miss_frac = 100.0 * float(missing_count[i])/ total_count
    print "{:7s} {:2.0f}/{:2.0f} {:2.2f}%".format(model_variables[i], missing_count[i], total_count, miss_frac)

