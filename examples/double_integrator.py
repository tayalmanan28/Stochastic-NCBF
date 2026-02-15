"""
Smooth SNCBF training for the Double Integrator system (bonus example).

Usage:
    python -m examples.double_integrator
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import superp_init as superp

# Set DI-specific hyperparameters
superp.DIM_S = 2
superp.lip_h = 1
superp.lip_dh = 1
superp.lip_d2h = 2

from main import run_training

if __name__ == "__main__":
    run_training('di')
