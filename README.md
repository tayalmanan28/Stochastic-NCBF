# Stochastic-NCBF
The official repository for Paper "Learning a Formally Verified Control Barrier Function in Stochastic Environment" [paper](https://arxiv.org/pdf/2403.19332), accepted for presentation at CDC'24 Milan, Italy.

## Installation

- Clone the repository:
```
git clone https://github.com/tayalmanan28/Stochastic-NCBF
```
- Install [Pytorch](https://pytorch.org/get-started/locally/)

## Usage 

In `main.py`, define the system you are interested in:

```
system = 'uni'  # 'ip' for inverted pendulum, 'di' for double integrator or 'uni' for unicycle model.
```
In `hyper_para.py` define the hyperparameters:
```
N_H_B = 1 # the number of hidden layers for the barrier
D_H_B = 20 # the number of neurons of each hidden layer for the barrier

###########################################
#Barrier function conditions
#########################################

gamma=0; #first condition <= 0
lamda=0.00001; #this is required for strict inequality >= lambda

############################################
# set loss function definition
############################################
TOL_SAFE = 0.0   # tolerance for safe and unsafe conditions
TOL_UNSAFE = 0.000

TOL_LIE = 0.000 #tolerance for the last condition

TOL_DATA_GEN = 1e-16 #for data generation


############################################
#Lipschitz bound for training
lip_h= 1
lip_dh= 1
lip_d2h= 2
############################################
# number of training epochs
############################################
EPOCHS = 500

############################################
############################################
ALPHA=0.1
BETA = 0 # if beta equals 0 then constant rate = alpha
GAMMA = 0 # when beta is nonzero, larger gamma gives faster drop of rate

#weights for loss function

DECAY_LIE = 1 # decay of lie weight 0.1 works, 1 does not work
DECAY_SAFE = 1
DECAY_UNSAFE = 1
```

Finally, run `main.py` to start training Stochastic Neural CBF:

```
python3 main.py
```
