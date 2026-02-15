# Defining safe and unsafe sets- vector field

import torch
import superp_init as superp
import math


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[-1, 1], \
            [- 1, 1]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_SAFE = []
SUB_SAFE_SHAPE = []


UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

UNSAFE1=[[-5, -3], [-5, 5]]
UNSAFE2=[[3, 5], [-5, 5]]
UNSAFE3=[[-5, 5], [-5, -3]]
UNSAFE4=[[-5, 5], [3, 5]]


# the the domain in super-rectangle
DOMAIN = [[-5, 5], \
            [-5, 5]
        ]
DOMAIN_SHAPE = 1 # 1 for rectangle



############################################
# set the range constraints
############################################
# accept a two-dimensional tensor and return a 
# tensor of bool with the same number of columns
def cons_safe(x): 
    ini= (x[:,0] >= INIT[0][0]) & (x[:,0 ] <= INIT[0][1]) & (x[:,1] >= INIT[1][0]) & (x[:,1] <= INIT[1][1])
    return ini

def cons_unsafe1(x):
    uns1=(x[:,0] >= UNSAFE1[0][0]) & (x[:,0 ] <= UNSAFE1[0][1]) & (x[:,1] >= UNSAFE1[1][0]) & (x[:,1] <= UNSAFE1[1][1])
    return uns1

def cons_unsafe2(x):
    uns2= (x[:,0] >= UNSAFE2[0][0]) & (x[:,0 ] <= UNSAFE2[0][1]) & (x[:,1] >= UNSAFE2[1][0]) & (x[:,1] <= UNSAFE2[1][1])
    return  uns2

def cons_unsafe3(x):
    uns3=(x[:,0] >= UNSAFE3[0][0]) & (x[:,0 ] <= UNSAFE3[0][1]) & (x[:,1] >= UNSAFE3[1][0]) & (x[:,1] <= UNSAFE3[1][1])
    return uns3

def cons_unsafe4(x):
    uns4= (x[:,0] >= UNSAFE4[0][0]) & (x[:,0 ] <= UNSAFE4[0][1]) & (x[:,1] >= UNSAFE4[1][0]) & (x[:,1] <= UNSAFE4[1][1])
    return  uns4
 
def cons_domain(x):
    dom1= (x[:,0] >= -3) & (x[:,0] <= -1) & (x[:,1] >= -3) & (x[:,1] <= 3)
    dom2= (x[:,0] >= 1) & (x[:,0] <= 5) & (x[:,1] >= -3) & (x[:,1] <= 3)
    dom3= (x[:,0] >= -3) & (x[:,0] <= 5) & (x[:,1] >= -3) & (x[:,1] <= -1)
    dom4= (x[:,0] >= -3) & (x[:,0] <= 5) & (x[:,1] >= 1) & (x[:,1] <= 3)
    return dom1 | dom2 | dom3 | dom4

def cons(x):
    return x[:, 0] == x[:, 0]
    

# this function accepts state and input and returns the next state
def func_f(x):
    def f(i, x):
        if i == 1:
            return x[:, 1]
        elif i == 2:
            return (0*x[:, 0])
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([f(i + 1, x) for i in range(superp.DIM_S)], dim=1)
    return vf

def func_g(x):
    def g(i):
        if i == 1:
            return 0*x[:, 1]
        elif i == 2:
            return 0*x[:, 1] + 1
        else:
            print("Vector function error!")
            exit()

    vf = torch.stack([g(i + 1) for i in range(superp.DIM_S)], dim=1)
    return vf

L_x=1.1
L_u=0.01
#L_f=L_x+L_u*L_c