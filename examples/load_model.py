import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import superp_init as superp # parameters
from safe import calc_safe_u
# import loss # computing loss
# import lrate
# import os
# import time

from deep_differential_network.differential_hessian_network import DifferentialNetwork
from deep_differential_network.replay_memory import PyTorchReplayMemory
from deep_differential_network.utils import jacobian, hessian, jacobian_auto

from utils.logger import DataLog
from utils.make_train_plots import make_train_plots


barr_nn = torch.load('experiments/uni_w_eta/iterations/barr_nn_300', map_location=torch.device('cpu')) 
print(barr_nn)
print('load complete')
barr_nn.eval()

