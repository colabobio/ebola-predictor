"""
This script performs list-wise deletion on the input file, so all rows with any missing
values are removed from the output.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import argparse, csv

"""Creates a complete output file by removing any rows in the input file with at least
one missing value

:param in_filename: input file with all the data
:param out_filename: output file with only complete rows
"""
def process(in_filename, out_filename):
    print "Removing incomplete rows from",in_filename
    titles = []
    data = []
    with open(in_filename, "rb") as ifile:
        reader = csv.reader(ifile)
        titles = reader.next()
        for row in reader:
            if "?" in row: continue
            data.append(row)

    print "Writing complete data to",out_filename
    with open(out_filename, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for row in data: writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs=1, default=["./data/training-data.csv"],
                        help="name of input training file")
    parser.add_argument('-o', '--output', nargs=1, default=["./data/training-data-completed.csv"],
                        help="name of output training file afer list-wise deletion")
    args = parser.parse_args()
    process(args.input[0], args.output[0])