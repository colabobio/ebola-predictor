#!/usr/bin/env python

"""
Runs a Kivy app as a desktop app.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs=1, default=["test"],
                    help="App directory")
args = parser.parse_args()

curr_dir = os.getcwd()
app_dir = os.path.join(curr_dir, args.dir[0])

os.chdir(app_dir)
cmd_str = 'python main.py'
os.system(cmd_str)
os.chdir(curr_dir)
