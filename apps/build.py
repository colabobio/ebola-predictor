#!/usr/bin/env python

"""
Build a Kivy app for the selected platform.

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, argparse, shutil, platform

def build_android(app_dir, build_options):
    p4a_dist = ''
    with open("./settings.cfg", "r") as sfile:
        lines = sfile.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            key, val = line.split("=")
            if key == "python-for-android":
                p4a_dist = val

    propfile = os.path.join(app_dir, "properties.txt")
    name="Name"
    version="0.0"
    package="org.company.dept"
    orientation="sensor"
    image_icon=""
    image_load=""
    sdk_target="19"
    sdk_minimum="19"
    with open(propfile, "r") as pfile:
        lines = pfile.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            key, val = line.split("=")
            if key == "name":
                name = val
            elif key == "version":
                version = val 
            elif key == "package":
                package = val
            elif key == "orientation":
                orientation = val
            elif key == "images.icon":
                image_icon = val
            elif key == "images.load":
                image_load = val
            elif key == "sdk.target":
                sdk_target = val
            elif key == "sdk.minimum":
                sdk_minimum = val

    curr_dir = os.getcwd()
    app_dir = os.path.join(curr_dir, app_dir)
    icon_file = os.path.join(app_dir, image_icon)
    load_file = os.path.join(app_dir, image_load)
    p4a_dir = os.path.abspath(p4a_dist)

    cmd_str = './build.py --package ' + package + ' --sdk ' + sdk_target + ' --minsdk ' + sdk_minimum + ' --name "' + name + '" --version ' + version + ' --dir ' + app_dir + ' --orientation "' + orientation + '" --icon ' + icon_file + ' --presplash ' + load_file + ' ' + build_options

    if "release" in build_options:
        build="release-unsigned"
    elif "debug" in build_options:
        build="debug"

    package_name = name.replace(" ", "") + "-" + version + "-" + build + ".apk"

    bin_dir = os.path.join(p4a_dir, "bin")
    src_file = os.path.join(bin_dir, package_name)
    dst_file = os.path.join(app_dir, "bin", package_name)

    if os.path.exists(bin_dir):
        shutil.rmtree(bin_dir)
    if os.path.exists(dst_file):
        os.remove(dst_file)

    os.chdir(p4a_dir)
    os.system(cmd_str)
    os.chdir(curr_dir)

    shutil.copyfile(src_file, dst_file)
    print ""
    print "Copied app package from",src_file,"to",dst_file

def build_ios(app_dir, build_options):
    k4ios_dir = ''
    with open("./settings.cfg", "r") as sfile:
        lines = sfile.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            key, val = line.split("=")
            if key == "kivy-ios":
                k4ios_dir = val

    propfile = os.path.join(app_dir, "properties.txt")
    name="Name"
    with open(propfile, "r") as pfile:
        lines = pfile.readlines()
        for line in lines:
            line = line.strip()
            if not line: continue
            key, val = line.split("=")
            if key == "name":
                name = val.replace(" ", "")

    curr_dir = os.getcwd()
    app_dir = os.path.join(curr_dir, app_dir)
    cmd_str = './toolchain.py create "' + name + '" ' + app_dir

    os.chdir(k4ios_dir)
    os.system(cmd_str)
    os.chdir(curr_dir)
    
    print ""
    prj_path = os.path.abspath(os.path.join(k4ios_dir, name.lower() + "-ios"))
    print "XCode project created in " + prj_path

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", nargs=1, default=["test"],
                    help="App directory")
parser.add_argument("-b", "--build", nargs=1, default=["release"],
                    help="Build options")
args = parser.parse_args()

app_dir = args.dir[0]
build_options = args.build[0]

if platform.system() == 'Linux':
    build_android(app_dir, build_options)
elif platform.system() == 'Darwin':
    build_ios(app_dir, build_options)
else:
    print "Unsupported platform."

