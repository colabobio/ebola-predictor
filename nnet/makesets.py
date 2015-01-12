'''
This script creates the training and testing sets.
'''

import sys, csv, random

var_file = "variables.txt"
input_file = "profile-data.tsv"
training_file = "training-data.csv"
testing_file = "testing-data.csv"

model_variables = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines():
        model_variables.append(line.split()[0])

test_percentage = 75
if 1 < len(sys.argv):
    test_percentage = float(sys.argv[1])

all_data = []
complete_rows = []

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
            all_data.append([row[idx].replace("\\N", "?") for idx in model_idx])
            if not some_missing: complete_rows.append(r)
            r = r + 1 

n = int(len(complete_rows) * (test_percentage / 100.0))
test_idx = random.sample(complete_rows, n)
training_data = []
testing_data = []
for r in range(0, len(all_data)):
    row = all_data[r]
    if r in test_idx:
        testing_data.append(row)
    else:
        training_data.append(row)    

print "Writing train set..."
with open(training_file, "wb") as trfile:
    writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(model_variables)
    for row in training_data:
        writer.writerow(row)
print "Done, wrote",len(training_data),"rows."

print "Writing test set..."
with open(testing_file, "wb") as tefile:
    writer = csv.writer(tefile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(model_variables)
    for row in testing_data:
        writer.writerow(row)
print "Done, wrote",len(testing_data),"rows."