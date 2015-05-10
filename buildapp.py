#!/usr/bin/env python

import os

#build_options = 'debug installd'
build_options = 'release'
curr_dir = os.getcwd()
app_dir = os.path.join(curr_dir, 'evdcare')
icon_file = os.path.join(app_dir, 'images', 'icon.png')
load_file = os.path.join(app_dir, 'images', 'loading.jpg')
pfora_dir = os.path.abspath('../python-for-android/dist/default')
os.chdir(pfora_dir)

cmd_str = './build.py --package org.sabetilab.ebolacare --name "Ebola CARE" --version 0.5 --dir ' + app_dir + ' --orientation "sensor" --icon ' + icon_file + ' --presplash ' + load_file + ' ' + build_options
os.system(cmd_str)
os.chdir(curr_dir)


