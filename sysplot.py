import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch 
import args
from matplotlib import cm
import math
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams.update({
    "text.usetex": True})

def sysdyn(x,ctrl_nn):
    tau=0.01;
    f=np.empty([1,2])
    xt=torch.tensor(x)
    cont=ctrl_nn(xt).detach().numpy()
    f[0,0]=xt[0]+ tau*xt[1] 
    f[0,1]=xt[1]+tau*(9.8 * (torch.sin(xt[0])) + cont)
    return f, cont

barr_nn=torch.load('saved_weights/barr_nn')
ctrl_nn=torch.load('saved_weights/ctrl_nn')

s=20
t=51

fig,axs = plt.subplots(3,1, figsize=(7, 15))

x=np.empty([t,2])
cont=np.empty([t,1])

for j in range(s):
    x[0,0]=args.init[0][0]+(args.init[0][1]-args.init[0][0])*np.random.rand(1,1)
    x[0,1]=args.init[1][0]+(args.init[1][1]-args.init[1][0])*np.random.rand(1,1)
    for i in range(t-1):
        x[i+1,:], cont[i,0] =sysdyn(x[i,:],ctrl_nn)
        # if x[i+1,0] >= args.x_max[0]:
        #     x[i+1,0] = args.x_max[0]
        # elif x[i+1,0] <= args.x_min[0]:
        #     x[i+1,0] = args.x_min[0]
        # if x[i+1,1] >= args.x_max[1]:
        #     x[i+1,1] = args.x_max[1]
        # elif x[i+1,1] <= args.x_min[1]:
        #     x[i+1,1] = args.x_min[1]  
    cont[t-1,0]=ctrl_nn(torch.tensor(x[t-1,:]))
    axs[0].step(range(0,t),x[:,0])
    axs[1].step(range(0,t),x[:,1],)
    axs[2].step(range(0,t),cont[:,0])

axs[0].set_xlabel("time step")
axs[1].set_xlabel("time step")
axs[2].set_xlabel("time step")
#axs[0].set_ylabel("$x_1$")
#axs[1].set_ylabel("$x_2$")

plt.savefig("stepplot.png",dpi=1200)    
plt.show()    





