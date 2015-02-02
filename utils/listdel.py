"""
This script performs list-wise deletion on the input file, so all rows with any missing
values are removed from the output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv

def listdel(in_file, out_file):
    titles = []
    data = []
    with open(in_file, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        for row in reader:
            if "?" in row: continue
            data.append(row)
    
    with open(out_file, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for row in data: writer.writerow(row)
    print "Done."

if __name__ == "__main__":
    in_file = "./data/training-data.csv"
    out_file = "./data/training-data-completed.csv"
    listdel(in_file, out_file)