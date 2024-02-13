import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch 
# import args
from matplotlib import cm
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import safe
import sys1
import random
# import main

data, prob = sys1.system_data('di')

mpl.rcParams.update(mpl.rcParamsDefault)

barr_nn=torch.load('experiments/di_2_cont/iterations/barr_nn_12')

# s=20
t=25000

# fig,axs = plt.subplots(3,1, figsize=(7, 15))

x = torch.zeros([t,2])

for i in range(t):
    x[i,0] = random.uniform(-5.0, 5.0)
    x[i,1] = random.uniform(-5.0, 5.0)
    


# x[:, 0] = torch.linspace(-5, 5, t)
# x[:, 1] = torch.linspace(-5, 5, t)

h, _ = barr_nn(x, hessian=False)
h = h[:, 0, :]

# print(x.shape, h.shape)
for i in range(t):
    if h[i]>=0:
        plt.scatter(x[i,0], x[i,1], c='blue', s=1)
    else:
        plt.scatter(x[i,0], x[i,1], c='red', s=1)

currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((0 - 1, 0 - 1), 2, 2, fill=None, alpha=1))

currentAxis2 = plt.gca()
currentAxis2.add_patch(Rectangle((0 - 3, 0 - 3), 6, 6, fill=None, alpha=1))

plt.savefig("experiments/di_2_cont/barr_plot.png",dpi=1200)    
plt.show()    






