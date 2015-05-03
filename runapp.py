#!/usr/bin/env python

import os

curr_dir = os.getcwd()
app_dir = os.path.join(curr_dir, 'evdcare')

os.chdir(app_dir)
cmd_str = 'python main.py'
os.system(cmd_str)
os.chdir(curr_dir)
