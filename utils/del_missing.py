"""

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import csv

training_file = "./data/training-data.csv"
out_file = "./data/training-data-imputed.csv"

def remove(training_file=training_file, out_file=out_file):
    titles = []
    data = []
    with open(training_file, "rb") as ifile:
        reader = csv.reader(ifile)        
        titles = reader.next()
        for row in reader:
            if "?" in row: continue
            data.append(row)
    
    with open(out_file, "wb") as trfile:
        writer = csv.writer(trfile, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(titles)
        for row in data:
            writer.writerow(row)
    print "Done."

if __name__ == "__main__":
    remove()