"""
Smooth SNCBF training for the Inverted Pendulum system.

Hyperparameters from experiment.tex:
  - epsilon_bar = 0.00016
  - L_h = 0.01, L_dh = 0.4, L_d2h = 2 => L_max = 2.4
  - sigma = diag(0.1, 0.1)
  - NN: 1 hidden layer, 20 neurons, SoftPlus activation
  - gamma = 1

Usage:
    python -m examples.inverted_pendulum
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import superp_init as superp

# Set IP-specific hyperparameters
superp.DIM_S = 2
superp.lip_h = 0.01
superp.lip_dh = 0.4
superp.lip_d2h = 2

from main import run_training

if __name__ == "__main__":
    run_training('ip')
