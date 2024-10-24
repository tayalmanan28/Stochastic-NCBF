# Defining safe and unsafe sets- vector field

import torch
import superp_init as superp
import math
from numpy import linalg as la
import unicycle_model.data as data

############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle

SAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

SAFE1=[[-2, -1.5], [-2, 2], [-2, 2]]
SAFE2=[[1.5, 2], [-2, 2], [-2, 2]]
SAFE3=[[-2, 2], [-2, -1.5], [-2, 2]]
SAFE4=[[-2, 2], [1.5, 2], [-2, 2]]

UNSAFE = [[-0.2, 0.2], [-0.2, 0.2], [-2, 2]]
UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

# the the domain in super-rectangle
DOMAIN = [[-2, 2], \
            [-2, 2], [-2, 2]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle


############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_unsafe(x): 
    ini= (x[:,0] >= UNSAFE[0][0]) & (x[:,0 ] <= UNSAFE[0][1]) & (x[:,1] >= UNSAFE[1][0]) & (x[:,1] <= UNSAFE[1][1]) & (x[:,2] >= UNSAFE[2][0]) & (x[:,2] <= UNSAFE[2][1])
    return ini

def cons_safe1(x):
    uns1=(x[:,0] >= SAFE1[0][0]) & (x[:,0 ] <= SAFE1[0][1]) & (x[:,1] >= SAFE1[1][0]) & (x[:,1] <= SAFE1[1][1]) & (x[:,2] >= SAFE1[2][0]) & (x[:,2] <= SAFE1[2][1])
    return uns1

def cons_safe2(x):
    uns2= (x[:,0] >= SAFE2[0][0]) & (x[:,0 ] <= SAFE2[0][1]) & (x[:,1] >= SAFE2[1][0]) & (x[:,1] <= SAFE2[1][1]) & (x[:,2] >= SAFE2[2][0]) & (x[:,2] <= SAFE2[2][1])
    return  uns2

def cons_safe3(x):
    uns3=(x[:,0] >= SAFE3[0][0]) & (x[:,0 ] <= SAFE3[0][1]) & (x[:,1] >= SAFE3[1][0]) & (x[:,1] <= SAFE3[1][1]) & (x[:,2] >= SAFE3[2][0]) & (x[:,2] <= SAFE3[2][1])
    return uns3

def cons_safe4(x):
    uns4= (x[:,0] >= SAFE4[0][0]) & (x[:,0 ] <= SAFE4[0][1]) & (x[:,1] >= SAFE4[1][0]) & (x[:,1] <= SAFE4[1][1]) & (x[:,2] >= SAFE4[2][0]) & (x[:,2] <= SAFE4[2][1])
    return  uns4
 
def cons_domain(x):
    dom = (x[:,0] >= DOMAIN[0][0]) & (x[:,0 ] <= DOMAIN[0][1]) & (x[:,1] >= DOMAIN[1][0]) & (x[:,1] <= DOMAIN[1][1]) & (x[:,2] >= DOMAIN[2][0]) & (x[:,2] <= DOMAIN[2][1])
    return dom

def cons(x):
    return x[:, 0] == x[:, 0]
    

# this function accepts state and input and returns the next state
def func_f(x):
    def f(i, x):
        if i == 1:
            return (torch.cos(x[:, 2]))
        elif i == 2:
            return (torch.sin(x[:, 2]))
        elif i == 3:
            return (0*x[:, 2])
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(data.DIM_S)], dim=1)
    return vf

def func_g(x):
    def g(i):
        if i == 1:
            return 0*x[:, 1]
        elif i == 2:
            return 0*x[:, 1]
        elif i == 3:
            return 0*x[:, 1] + 1
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([g(i + 1) for i in range(data.DIM_S)], dim=1)
    return vf

L_x=1

