"""
Smooth SNCBF training for the Unicycle Model system.

Hyperparameters from experiment.tex:
  - epsilon_bar = 0.01
  - L_h = 1, L_dh = 1, L_d2h = 2 => L_max = 4
  - sigma = diag(0.1, 0.1, 0.1)
  - NN: 1 hidden layer, 20 neurons, SoftPlus activation
  - gamma = 1

Usage:
    python -m examples.unicycle
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import superp_init as superp

# Set unicycle-specific hyperparameters
superp.DIM_S = 3
superp.lip_h = 1
superp.lip_dh = 1
superp.lip_d2h = 2

from main import run_training

if __name__ == "__main__":
    run_training('uni')
