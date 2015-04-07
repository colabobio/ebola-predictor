"""
Unpacks the contents of a zip file in the store folder into the data folder

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse, glob
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument('-N', '--name', nargs=1, default=["test"],
                    help="Model to unpack")
args = parser.parse_args()
base = args.base_dir[0]
name = args.name[0]

zip_filename = os.path.join("./store", name + ".zip")
if not os.path.exists(zip_filename):
    print "The file",zip_filename,"does not exist!"
    exit(1)

zf = zipfile.ZipFile(zip_filename, "r")
print "Extracting " + zip_filename + " into data folder..."
try:
    for name in zf.namelist():
        fn = os.path.join(base, name)
        print "  Extracting file", fn
        zf.extract(name, base)
finally:
    zf.close()
print "Done."
