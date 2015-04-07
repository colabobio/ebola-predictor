"""
Runs all the jobs in the jobs folder

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, sys, glob, argparse

parser = argparse.ArgumentParser()
parser.add_argument('-B', '--base_dir', nargs=1, default=["./"],
                    help="Base directory")
parser.add_argument('-P', '--proj_name', nargs=1, default=["Sabeti-ML-pred"],
                    help="LSF project name")
parser.add_argument("-m", "--mode", nargs=1, default=["debug"],
                    help="running mode: debug, local, lsf")
parser.add_argument("-q", "--queue", nargs=1, default=["hour"],
                    help="type of queue in LSF mode: hour or week")
parser.add_argument("-c", "--config", nargs=1, default=["./job.cfg"],
                    help="job config file")
parser.add_argument("-d", "--jobs_dir", nargs=1, default=["./jobs"],
                    help="Directory where to read job files from")
args = parser.parse_args()

base = args.base_dir[0]
mode = args.mode[0].lower()
queue = args.queue[0].lower()
cfgfile = args.config[0]
prjname = args.proj_name[0]
jobs_dir = args.jobs_dir[0]

out_dir = os.path.join(base, "out")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

job_files = glob.glob(jobs_dir + "/*")
for file in job_files:
    out_file = os.path.join(out_dir, os.path.split(file)[1] + ".out")
    if mode == "local":
        cmd_str = "./job.py -j " + file + " -c " + cfgfile + " > " + out_file
        print "Running job",file,"with configuration",cfgfile,"and output to",out_file,"..."
        os.system(cmd_str)
    else:
        cmd_str = "bsub -o " + out_file + " -q " + queue + " -P " + prjname + " ./job.py -j " + file + " -c " + cfgfile
        if mode == "debug":
            print cmd_str
        else:
            print "Submitting job",file,"with configuration",cfgfile,"and output to",out_file,"..."
            os.system(cmd_str)