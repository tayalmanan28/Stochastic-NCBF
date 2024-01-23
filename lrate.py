import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import superp_init as superp


############################################
# learn rate scheduling
############################################
def set_scheduler(optimizer, num_batches_per_epoch):
    def lr_lambda(num_sched): # this epoch is not the same of training loop epochs
        
        #rate = superp.ALPHA
        #rate = superp.ALPHA + 1.0 * superp.BETA * epoch / num_batches / superp.EPOCHS
        rate = superp.ALPHA + superp.BETA * torch.sigmoid(torch.tensor(num_sched * 1.0 / num_batches_per_epoch) - superp.GAMMA)        
        ## rate = alpha / (1 + beta * epoch^gamma)
        #rate = superp.ALPHA / (1.0 + superp.BETA * np.power(epoch, superp.GAMMA))
        
        return rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler