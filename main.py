import sys
import torch
import torch.nn as nn
import sys1
import train
import time
import superp_init as superp


def run_training(system):
    """Run SNCBF training for the specified system.

    Args:
        system: System identifier ('ip' for inverted pendulum,
                'uni' for unicycle, 'di' for double integrator)
    """
    # Set system-specific hyperparameters
    if system == 'ip':
        superp.DIM_S = 2
        superp.lip_h = 0.01
        superp.lip_dh = 0.4
        superp.lip_d2h = 2
    elif system == 'uni':
        superp.DIM_S = 3
        superp.lip_h = 1
        superp.lip_dh = 1
        superp.lip_d2h = 2
    elif system == 'di':
        superp.DIM_S = 2
        superp.lip_h = 1
        superp.lip_dh = 1
        superp.lip_d2h = 2

    # Generate training data
    data, prob = sys1.system_data(system)
    time_start_data = time.time()
    batches_safe, batches_unsafe, batches_domain = data.gen_batch_data()
    time_end_data = time.time()

    # Number of mini_batches
    BATCHES_S = len(batches_safe)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_S, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_S, BATCHES_U, BATCHES_D, BATCHES]

    # Train and return the learned model
    time_start_train = time.time()
    res = train.itr_train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system)
    time_end_train = time.time()

    print("\nData generation totally costs:", time_end_data - time_start_data)
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")

    return res


if __name__ == "__main__":
    system = sys.argv[1] if len(sys.argv) > 1 else 'uni'
    print(f"Training SNCBF for system: {system}")
    run_training(system)
