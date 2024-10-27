import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import hyper_para as hyp


############################################
# learn rate scheduling
############################################
def set_scheduler(optimizer, num_batches_per_epoch):
    def lr_lambda(num_sched): # this epoch is not the same of training loop epochs
        
        #rate = hyp.ALPHA
        #rate = hyp.ALPHA + 1.0 * hyp.BETA * epoch / num_batches / hyp.EPOCHS
        rate = hyp.ALPHA + hyp.BETA * torch.sigmoid(torch.tensor(num_sched * 1.0 / num_batches_per_epoch) - hyp.GAMMA)        
        ## rate = alpha / (1 + beta * epoch^gamma)
        #rate = hyp.ALPHA / (1.0 + hyp.BETA * np.power(epoch, hyp.GAMMA))
        
        return rate

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    return scheduler