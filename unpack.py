"""
Unpacks the contents of a zip file in the store folder into the data folder

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse, glob
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("name", nargs=1, default=["no name"],
                    help="Name of zip file to unpack")
args = parser.parse_args()

zip_filename = os.path.join("./store", args.name[0] + ".zip")
if not os.path.exists(zip_filename):
    print "The file",zip_filename,"does not exist!"
    exit(1)

all_files = glob.glob("./data/*")
for file in all_files: os.remove(file)

zf = zipfile.ZipFile(zip_filename, "r")
print "Extracting " + zip_filename + " into data folder..."
try:
    for name in zf.namelist():
        outpath = "./"    
        fn = os.path.join(outpath, name)
        print "  Extracting file", fn
        zf.extract(name, outpath)
finally:
    zf.close()
print "Done."
