"""
Packs the current data folder into a zip file inside the store folder

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse, glob
import zipfile, zlib

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument('-N', '--name', nargs=1, default=["test"],
                    help="Model name")
args = parser.parse_args()
base = args.base_dir[0]
name = args.name[0]

data_files = glob.glob("./data/*")
model_dir =  os.path.join(base, "models", name)
model_files = glob.glob(model_dir + "/*")

if not os.path.exists("./store"): os.makedirs("./store")

zip_filename = os.path.join("./store", name + ".zip")

if os.path.exists(zip_filename):
    print "The file",zip_filename,"already exists, choose another name!"
    exit(1)

zf = zipfile.ZipFile(zip_filename, mode='w')
print "Compressing data folder into " + zip_filename + "..."
try:
    for file in model_files:
        print "  Adding file", file
        zf.write(file, compress_type=zipfile.ZIP_DEFLATED)
    for file in data_files:
        print "  Adding file", file
        zf.write(file, compress_type=zipfile.ZIP_DEFLATED)
finally:
    zf.close()

print "Done."