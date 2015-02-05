"""
Renders a graphical representation of the neural network stored in a parameters file.

Based on the code from Jake VanderPlas <vanderplas@astro.washington.edu>
Released under the BSD License

The figure produced by this code is published in the textbook
"Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
For more information, see http://astroML.github.com
http://www.astroml.org/book_figures/appendix/fig_neural_network.html

Important: Only works for 1 hidden layer!
"""

import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import linear_index, thetaMatrix

pred_file = "./data/predictor.txt"
var_file = "./data/variables.txt"

def linear_index(mat_idx, N, L, S, K):
    l = mat_idx[0] # layer
    n = mat_idx[1] # node
    i = mat_idx[2] # input
    if l < 1: 
        return n * N + i
    elif l < L - 1:
        return (S - 1) * N + (l - 1) * (S - 1) * S + n * S + i
    else:
        return (S - 1) * N + (L - 2) * (S - 1) * S + n * S + i

def thetaMatrix(theta, N, L, S, K):
    # The cost argument is a 1D-array that needs to be reshaped into the
    # parameter matrix for each layer:
    thetam = [None] * L
    C = (S - 1) * N
    thetam[0] = theta[0 : C].reshape((S - 1, N))
    for l in range(1, L - 1):
        thetam[l] = theta[C : C + (S - 1) * S].reshape((S - 1, S))
        C = C + (S - 1) * S
    thetam[L - 1] = theta[C : C + K * S].reshape((K, S))
    return thetam

##########################################################################################
# Main
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("param", nargs=1, default=["./data/nnet-params"], help="parameters of neural network")
args = parser.parse_args()

model_variables = []
with open(var_file, "rb") as vfile:
    for line in vfile.readlines(): 
        name = line.split()[0]
        model_variables.append(name)
model_variables[0] = "Bias"

with open(args.param[0], "rb") as pfile:
    i = 0
    for line in pfile.readlines():
        [name, value] = line.strip().split(":")
        if i == 0:
            N = int(value.strip()) + 1
        elif i == 1:
            L = int(value.strip()) + 1
        elif i == 2:
            S = int(value.strip()) + 1
        elif i == 3:
            K = int(value.strip())
            R = (S - 1) * N + (L - 2) * (S - 1) * S + K * S
            theta = np.ones(R)
        else:
            idx = [int(s.strip().split(" ")[1]) for s in name.split(",")]
            n = linear_index(idx, N, L, S, K)
            theta[n] = float(value.strip())
        i = i + 1

fig = plt.figure(facecolor='w')
ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[])
plt.box(False)
circ = plt.Circle((1, 1), 2)

radius = 0.2

arrow_kwargs = dict(head_width=0.05, fc='black')

# function to draw arrows
def draw_connecting_arrow(ax, circ1, rad1, circ2, rad2, coeff):
    theta = np.arctan2(circ2[1] - circ1[1],
                       circ2[0] - circ1[0])

    starting_point = (circ1[0] + rad1 * np.cos(theta),
                      circ1[1] + rad1 * np.sin(theta))

    length = (circ2[0] - circ1[0] - (rad1 + 1.4 * rad2) * np.cos(theta),
              circ2[1] - circ1[1] - (rad1 + 1.4 * rad2) * np.sin(theta))

    arrow_kwargs['fc'] = 'black'
    arrow_kwargs['ec'] = None
    arrow_kwargs['alpha'] = 1
    arrow_kwargs['head_width'] = 0.05
    w = 0.01
    if coeff:
        arrow_kwargs['alpha'] = 0.5
        w = 0.05 * abs(coeff) / max_coeff
        arrow_kwargs['head_width'] = 0.1 * abs(coeff) / max_coeff
        if coeff < 0:
            arrow_kwargs['ec'] = 'red'
            arrow_kwargs['fc'] = 'red'
        else:
            arrow_kwargs['ec'] = 'blue'
            arrow_kwargs['fc'] = 'blue'

    ax.arrow(starting_point[0], starting_point[1],
             length[0], length[1], width=w, **arrow_kwargs)

# function to draw circles
def draw_circle(ax, center, radius, color='none'):
    circ = plt.Circle(center, radius, fc=color, lw=2)
    ax.add_patch(circ)

x1 = -2
x2 = 0
x3 = 2
y3 = 0

#------------------------------------------------------------
# draw circles
for i, y1 in enumerate(np.linspace(1.5, -1.5, N)):
    draw_circle(ax, (x1, y1), radius, 'none' if 0 < i else 'grey')
    if 0 < i:
        ax.text(x1 - 0.9, y1, model_variables[i],
                ha='right', va='center', fontsize=16)
        draw_connecting_arrow(ax, (x1 - 0.9, y1), 0.1, (x1, y1), radius, None)

for i, y2 in enumerate(np.linspace(2, -2, S)):
    draw_circle(ax, (x2, y2), radius, 'none' if 0 < i else 'grey')

draw_circle(ax, (x3, y3), radius)
ax.text(x3 + 0.8, y3, 'Outcome', ha='left', va='center', fontsize=16)
draw_connecting_arrow(ax, (x3, y3), radius, (x3 + 0.8, y3), 0.1, None)

#------------------------------------------------------------
# draw connecting arrows
thetam = thetaMatrix(theta, N, L, S, K)

theta0 = thetam[0]
max_coeff = 0
i1 = 0
for y1 in range(0, N):
    i2 = 0
    for y2 in range(0, S - 1):
        if max_coeff < abs(theta0[y2][y1]): max_coeff = abs(theta0[i2][i1])
        i2 = i2 + 1
    i1 = i1 + 1
i1 = 0
offset = 4.0 / (S - 1)
for i2, y2 in enumerate(np.linspace(2, -2, S)):
    y2 = y2 - offset
    for i1, y1 in enumerate(np.linspace(1.5, -1.5, N)):
        if i2 == S - 1: continue
        draw_connecting_arrow(ax, (x1, y1), radius, (x2, y2), radius, theta0[i2][i1])

# IMPORTANT: no more than 1 hidden layers are supported!

thetaf = thetam[1]
max_coeff = 0
for y2 in range(0, S):
    if max_coeff < abs(thetaf[0][y2]): max_coeff = abs(thetaf[0][y2])
for i2, y2 in enumerate(np.linspace(2, -2, S)):
    draw_connecting_arrow(ax, (x2, y2), radius, (x3, y3), radius, thetaf[0][i2])

#------------------------------------------------------------
# Add text labels
plt.text(x1, 2.7, "Input\nLayer", ha='center', va='top', fontsize=16)
plt.text(x2, 2.7, "Hidden Layer", ha='center', va='top', fontsize=16)
plt.text(x3, 2.7, "Output\nLayer", ha='center', va='top', fontsize=16)

ax.set_aspect('equal')
plt.xlim(-4, 4)
plt.ylim(-3, 3)
plt.show()