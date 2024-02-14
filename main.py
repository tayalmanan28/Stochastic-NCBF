import torch
import torch.nn as nn
import sys1
import train
import time

system = 'ip'

def barr_nn(system):
    # generate training data
    # sys.sys_data(sys)
    data, prob = sys1.system_data(system)
    time_start_data = time.time()
    batches_safe, batches_unsafe, batches_domain = data.gen_batch_data()
    time_end_data = time.time()
    
    ############################################
    # number of mini_batches
    ############################################
    BATCHES_S = len(batches_safe)
    BATCHES_U = len(batches_unsafe)
    BATCHES_D = len(batches_domain)
    BATCHES = max(BATCHES_S, BATCHES_U, BATCHES_D)
    NUM_BATCHES = [BATCHES_S, BATCHES_U, BATCHES_D, BATCHES]
    
    # train and return the learned model
    time_start_train = time.time()
    res = train.itr_train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES, system) 
    time_end_train = time.time()
    
    print("\nData generation totally costs:", time_end_data - time_start_data)
    print("Training totally costs:", time_end_train - time_start_train)
    print("-------------------------------------------------------------------------")

        
    return barr_nn


if __name__ =="__main__":
     
     barr_nn = barr_nn(system)
    #  torch.save(barr_nn,'saved_weights/barr_nn')
