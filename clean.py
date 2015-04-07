"""
Cleans folders

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse, shutil

def delete(files):
    for file in files:
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument("-m", "--clean_models", action="store_true",
                    help="clean models folder")
parser.add_argument("-j", "--clean_jobs", action="store_true",
                    help="clean jobs folder")
parser.add_argument("-o", "--clean_out", action="store_true",
                        help="clean out folder")
parser.add_argument("-a", "--clean_all", action="store_true",
                    help="clean all folders")
args = parser.parse_args()

base = args.base_dir[0]
if args.clean_models or args.clean_all:
    print "Cleaning models folder..."
    delete(glob.glob(os.path.join(base, "models") + "/*"))
if args.clean_jobs or args.clean_all:
    print "Cleaning jobs folder..."
    delete(glob.glob(os.path.join(base, "jobs") + "/*"))
if args.clean_out or args.clean_all:
    print "Cleaning out folder..."
    delete(glob.glob(os.path.join(base, "out") + "/*"))

print "Cleaning temp folder..."
delete(glob.glob(os.path.join(base, "temp") + "/*"))

print "Done."