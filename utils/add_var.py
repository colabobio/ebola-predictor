"""
Adds a new variable to the data, using the provided script

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import sys, os, csv, argparse
from importlib import import_module

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", nargs=1, default=["./data/data.csv"],
                    help="File containing input data file")
parser.add_argument("-o", "--output", nargs=1, default=["./data/data-new.csv"],
                    help="File containing output data file")
parser.add_argument("-s", "--script", nargs=1, default=["./scripts/var.py"],
                    help="Python script implementing calculation of new variable")
args = parser.parse_args()

source_data = args.input[0]
dest_data = args.output[0]
user_script = args.script[0]

work_path = os.getcwd()
[module_path, module_filename] = os.path.split(user_script);

module_path = os.path.abspath(module_path)
module_filename = module_filename.split('.')[0]
sys.path.insert(0, module_path)

module = import_module(module_filename)

try:
    # Changing to script folder just in case it opens some files during initialization
    os.chdir(module_path)
    module.init()
    os.chdir(work_path);
except AttributeError:
    pass

print "ADDING VARIABLE " + module.get_name() + " TO DATA IN " + source_data + "..."

data = []
with open(source_data, "rb") as ifile:
    for row in csv.reader(ifile, dialect="excel"):
        data.append(row)
titles = data[0]

variables = module.variables()
    
with open(dest_data, "wb") as tsv:
    writer = csv.writer(tsv, dialect="excel")
    titles.append(module.get_name())
    writer.writerow(titles)

    for x in range(1, len(data)):
        values = {}
        for var in variables:
            values[var] = data[x][titles.index(var)]
            
        data[x].append(module.calculate(values))
        writer.writerow(data[x])

print "DONE"