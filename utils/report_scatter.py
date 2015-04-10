"""
Creates scatter plot from reports

@copyright: The Broad Institute of MIT and Harvard 2015
"""

import os, glob, argparse
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument('pred', nargs=1, default=["nnet"],
                    help="Predictor to generate plot for")
parser.add_argument('-m', '--scatter_mode', nargs=1, default=["ps"],
                    help="Scatter mode: ps (precision/sensitivity), f1 (F1-score)")
args = parser.parse_args()
predictor = args.pred[0]
mode = args.scatter_mode[0]

# Color Brewer tool is useful to generate colors that are easy to differentiate:
# http://colorbrewer2.org/
properties = {
    # Amelia: reds
    'amelia-df1-t50': { "label":"Amelia,#1,50%", "color":(252,174,145), "size":20},
    'amelia-df1-t65': { "label":"Amelia,#1,65%", "color":(251,106,74), "size":20},
    'amelia-df1-t80': { "label":"Amelia,#1,80%", "color":(203,24,29), "size":20},

    'amelia-df5-t50': { "label":"Amelia,#5,50%", "color":(252,174,145), "size":60},
    'amelia-df5-t65': { "label":"Amelia,#5,65%", "color":(251,106,74), "size":60},
    'amelia-df5-t80': { "label":"Amelia,#5,80%", "color":(203,24,29), "size":60},

    'amelia-df10-t50': { "label":"Amelia,#10,50%", "color":(252,174,145), "size":120},
    'amelia-df10-t65': { "label":"Amelia,#10,65%", "color":(251,106,74), "size":120},
    'amelia-df10-t80': { "label":"Amelia,#10,80%", "color":(203,24,29), "size":120},

    # MICE: greens    
    'mice-df1-t50': { "label":"MICE,#1,50%", "color":(186,228,179), "size":20},
    'mice-df1-t65': { "label":"Amelia,#1,65%", "color":(116,196,118), "size":20},
    'mice-df1-t80': { "label":"Amelia,#1,80%", "color":(35,139,69), "size":20},

    'mice-df5-t50': { "label":"MICE,#5,50%", "color":(186,228,179), "size":60},
    'mice-df5-t65': { "label":"MICE,#5,65%", "color":(116,196,118), "size":60},
    'mice-df5-t80': { "label":"MICE,#5,80%", "color":(35,139,69), "size":60},

    'mice-df10-t50': { "label":"MICE,#10,50%", "color":(186,228,179), "size":120},
    'mice-df10-t65': { "label":"MICE,#10,65%", "color":(116,196,118), "size":120},
    'mice-df10-t80': { "label":"MICE,#10,80%", "color":(35,139,69), "size":120},

    # Hmisc: blues
    'hmisc-df1-t50': { "label":"Hmisc,#1,50%", "color":(189,215,231), "size":20},
    'hmisc-df1-t65': { "label":"Hmisc,#1,65%", "color":(107,174,214), "size":20},
    'hmisc-df1-t80': { "label":"Hmisc,#1,65%", "color":(33,113,181), "size":20},

    'hmisc-df5-t50': { "label":"Hmisc,#5,50%", "color":(189,215,231), "size":60},
    'hmisc-df5-t65': { "label":"Hmisc,#5,65%", "color":(107,174,214), "size":60},
    'hmisc-df5-t80': { "label":"Hmisc,#5,80%", "color":(33,113,181), "size":60},

    'hmisc-df10-t50': { "label":"Hmisc,#10,50%", "color":(189,215,231), "size":120},
    'hmisc-df10-t65': { "label":"Hmisc,#10,65%", "color":(107,174,214), "size":120},
    'hmisc-df10-t80': { "label":"Hmisc,#10,80%", "color":(33,113,181), "size":120},    
}
opacity = 200

plt.clf()
fig = plt.figure()

plt.xlim([0.5,1.0])
plt.ylim([0.5,1.0])
plt.axes().set_aspect('equal')

plots = []
labels = []
base_dir = "./models"
for dir_name, subdir_list, file_list in os.walk(base_dir):
    name = os.path.split(dir_name)[1]    
    train_files = glob.glob(dir_name + "/training-data-completed-*.csv")
    if not train_files: continue
    parts = name.split("-")
    if len(parts) < 3: continue
    print "**********",name    
    ialg, numf, tperc = parts 
    os.system("python eval.py -N " + name + " -p " + predictor + " -m report > ./out/" + name)
    with open("./out/" + name, "r") as report:
        lines = report.readlines()
        if lines:
            last = lines[-1]
            parts = last.split(",")        
            
            precision_mean = float(parts[1].strip())
            sensitivity_mean = float(parts[2].strip())
            precision_err = float(parts[4].strip()) 
            sensitivity_err = float(parts[5].strip()) 
            f1_mean = float(parts[3].strip())
            f1_err = float(parts[6].strip()) 

            if mode == "ps": 
                x = precision_mean     
                y = sensitivity_mean
            else:
                x = f1_mean
                y = f1_err
            
#             print x, y
             
            size = properties[name]["size"]
            color = [c/255.0 for c in properties[name]["color"]]
            color.append(opacity/255.0)
            label = properties[name]["label"]
            
#             (_, caps, _) = plt.errorbar(precision_mean, sensitivity_mean, precision_err, sensitivity_err, color=color, linewidth=1, capsize=4)
#             for cap in caps:
#                 cap.set_color(c)
#                 cap.set_markeredgewidth(0.5)
            
#             plots.append(plt.scatter(x, y, s=size, color=color, marker='o'))
            plots.append(plt.scatter(x, 1 - y, s=size, color=color, marker='o'))
            labels.append(label)

# plt.legend(tuple(plots),
#            tuple(labels),
#            loc='best',
#            ncol=3,
#            prop={'size':9})

if mode == "ps": 
    plt.xlabel('Mean precision')
    plt.ylabel('Mean sensitivity')
else:
    plt.xlabel('Mean F1 score')
    plt.ylabel('1 - F1 score error')

# plt.show()
fig.savefig("./out/report-" + predictor + ".pdf")