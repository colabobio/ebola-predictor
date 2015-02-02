"""
This script performs list-wise deletion on the input file, so all rows with any missing
values are removed from the output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse
import csv

"""Creates a complete output file by removing any rows in the input file with at least
one missing value

:param in_file: input file with all the data
:param out_file: output file with only complete rows
"""
def listdel(in_file, out_file):
    print "Removing incomplete rows from",in_file
    titles = []
    data = []
    with open(in_file, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        for row in reader:
            if "?" in row: continue
            data.append(row)
            
    print "Writing complete data to",out_file
    with open(out_file, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for row in data: writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', nargs=1, default=["./data/training-data.csv"])
    parser.add_argument('-o', nargs=1, default=["./data/training-data-completed.csv"])
    args = parser.parse_args()
    listdel(args.i[0], args.o[0])