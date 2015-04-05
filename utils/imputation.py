import sys, os, csv

var_file = "./data/variables.txt"
def load_variables(in_filename):
    dir = os.path.split(in_filename)[0]
    if os.path.exists(dir + "/variables.txt"):
        fn = dir + "/variables.txt"
    else:
        fn = var_file
    var_names = []
    var_types = {}
    with open(fn, "rb") as vfile:
        for line in vfile.readlines():
            line = line.strip()
            if not line: continue
            [name, type] = line.split()[0:2]
            var_names.append(name)
            var_types[name] = type
    return var_names, var_types

def load_bounds(in_filename, var_names, var_types):
    # Extract bounds from data
    bounds = [[1000, 0] for x in var_names]
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
    for i in range(0, len(var_names)):
        if var_types[var_names[i]] == "category": continue
        bounds[i][0] = (1 - tol) * bounds[i][0]
        bounds[i][1] = (1 + tol) * bounds[i][1]
    return bounds
    
def aggregate_files(out_filename, imp_files, var_names, var_types, bounds):
    print "Aggregating imputed datasets..."
    aggregated_data = []
    for fn in imp_files:
        print "  Reading " + fn
        with open(fn, "rb") as ifile:
            reader = csv.reader(ifile, delimiter=",")
            reader.next()
            for row in reader:
                if row[0] == "NA": 
                    print "    Empty dataset, skipping!"
                    break
                add = True
                for i in range(0, len(row)):
                    name = var_names[i]
                    if var_types[name] != "category":
                        val = float(row[i])
                        if val < bounds[i][0] or bounds[i][1] < val:
                            print "    Value " + row[i] + " for variable " + name + " is out of bounds [" + str(bounds[i][0]) + ", " + str(bounds[i][1]) + "], skipping"
                            add = False
                            break
                if add: aggregated_data.append(row)

    if aggregated_data:
        with open(out_filename, "wb") as trfile:
            writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(var_names)
            for row in aggregated_data:
                writer.writerow(row)
        print "Saved aggregated imputed datasets to", out_filename
