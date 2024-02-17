# Defining safe and unsafe sets- vector field

import torch
import superp_init as superp
import math


############################################
# set the super-rectangle range
############################################
# set the initial in super-rectangle
INIT = [[- math.pi/15, math.pi/15], \
            [- math.pi/15, math.pi/15]
        ]
INIT_SHAPE = 1 # 2 for circle, 1 for rectangle

SUB_SAFE = []
SUB_SAFE_SHAPE = []


UNSAFE_SHAPE = 1 # 2 for circle, 1 for rectangle

UNSAFE1=[[-1 / 4 * math.pi, -1 / 6 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]]
UNSAFE2=[[1 / 6 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, 1 / 4 * math.pi]]
UNSAFE3=[[-1 / 4 * math.pi, 1 / 4 * math.pi], [-1 / 4 * math.pi, -1 / 6 * math.pi]]
UNSAFE4=[[-1 / 4 * math.pi, 1 / 4 * math.pi], [1 / 6 * math.pi, 1 / 4 * math.pi]]


# the the domain in super-rectangle
DOMAIN = [[-1 / 4 * math.pi, 1 / 4 * math.pi], \
            [-1 / 4 * math.pi, 1 / 4 * math.pi]
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
    dom1= (x[:,0] >= -math.pi/6) & (x[:,0] <= -math.pi/15) & (x[:,1] >= -math.pi/6) & (x[:,1] <= math.pi/6)
    dom2= (x[:,0] >= math.pi/15) & (x[:,0] <= math.pi/4) & (x[:,1] >= -math.pi/6) & (x[:,1] <= math.pi/6)
    dom3= (x[:,0] >= -math.pi/6) & (x[:,0] <= math.pi/4) & (x[:,1] >= -math.pi/6) & (x[:,1] <= -math.pi/15)
    dom4= (x[:,0] >= -math.pi/6) & (x[:,0] <= math.pi/4) & (x[:,1] >= math.pi/15) & (x[:,1] <= math.pi/6)
    return dom1 | dom2 | dom3 | dom4

def cons(x):
    return x[:, 0] == x[:, 0]
    

# this function accepts state and input and returns the next state
def func_f(x):
    def f(i, x):
        if i == 1:
            return x[:, 1]
        elif i == 2:
            return (9.8 * torch.sin(x[:, 0]))
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

L_x=1