import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import superp_init as superp # parameters
import loss # computing loss
import lrate
import os
import time

from deep_differential_network.differential_hessian_network import DifferentialNetwork
from deep_differential_network.replay_memory import PyTorchReplayMemory
from deep_differential_network.utils import jacobian, hessian, jacobian_auto

from utils.logger import DataLog
from utils.make_train_plots import make_train_plots

LOAD_MODEL = False
RENDER = True
SAVE_MODEL = True
SAVE_PLOT = False


torch.set_printoptions(precision=7)

#################################################
# iterative training: the most important function
# it relies on three assistant functions:
#################################################


# used to output learned model parameters
def print_nn(model):
    for p in model.parameters():
        print(p.data)

def print_nn_matlab(model):
    layer = 0
    for p in model.parameters():
        layer = layer + 1
        arr = p.detach().numpy()
        if arr.ndim == 2:
            print( "w" + str((layer + 1) // 2) + " = [", end="")
            print('; '.join([', '.join(str(curr_int) for curr_int in curr_arr) for curr_arr in arr]), end="];\n")
        elif arr.ndim == 1:
            print( "b" + str(layer // 2) + " = [", end="")
            if layer == 2:
                print(', '.join(str(i) for i in arr), end="]';\n")
            else:
                print(', '.join(str(i) for i in arr), end="];\n")
        else:
            print("Transform error!")

# used for initialization and restart

def initialize_parameters(n_h_b, d_h_b):
    #initialize the eta variable for scenario verification
    lambdas=Variable(torch.normal(mean=10*torch.ones(n_h_b*d_h_b),std=0.001*torch.ones(n_h_b*d_h_b)), requires_grad=True)
    print("Initialize eta")
    eta=Variable(torch.normal(mean=torch.tensor([-0.00075]), std=torch.tensor([0.00001])), requires_grad=True)
    return lambdas, eta


    
def initialize_nn(num_batches):    
    print("Initialize nn parameters!")
    cuda_flag = True
    filename = f"barrier_function"
    n_dof = 2
    # Construct Hyperparameters:
    # Activation must be in ['ReLu', 'SoftPlus']
    hyper = {'n_width': superp.D_H_B,
             'n_depth': superp.N_H_B,
             'learning_rate': 1.0e-03,
             'weight_decay': 1.e-6,
             'activation': "SoftPlus"}

    # Load existing model parameters:
    if LOAD_MODEL:
        load_file = f"./models/{filename}_loss.torch"
        state = torch.load(load_file, map_location='cpu')

        barr_nn = DifferentialNetwork(n_dof, **state['hyper'])
        barr_nn.load_state_dict(state['state_dict'])

    else:
        barr_nn = DifferentialNetwork(n_dof, **hyper)

    if cuda_flag:
        barr_nn.cuda()
        
    # Generate & Initialize the Optimizer:
    t0_opt = time.perf_counter()
    optimizer = torch.optim.Adam(barr_nn.parameters(),
                                    lr=hyper["learning_rate"],
                                    weight_decay=hyper["weight_decay"],
                                    amsgrad=True)

    print("{0:30}: {1:05.2f}s".format("Initialize Optimizer", time.perf_counter() - t0_opt))
    scheduler = lrate.set_scheduler(optimizer, num_batches)

    return barr_nn, optimizer,scheduler

def itr_train(batches_safe, batches_unsafe, batches_domain, NUM_BATCHES):
    logger = DataLog()
    log_dir = "experiments/"
    working_dir = os.getcwd()

    if os.path.isdir(log_dir) == False:
        os.mkdir(log_dir)

    previous_dir = os.getcwd()
    
    os.chdir(log_dir)
    if os.path.isdir('iterations') == False: os.mkdir('iterations')
    if os.path.isdir('logs') ==False: os.mkdir('logs')

    log_dir = os.getcwd()
    os.chdir(working_dir)
    num_restart = -1

    ############################## the main training loop ##################################################################
    while num_restart < 4:
        num_restart += 1
        
        # initialize nn models and optimizers and schedulers
        lambdas, eta = initialize_parameters(superp.N_H_B, superp.D_H_B)
        barr_nn, optimizer_barr, _ = initialize_nn(NUM_BATCHES[3])
        optimizer_eta= torch.optim.SGD([{'params':[eta]}], lr=0.001, momentum=0)


        safe_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[0]  # generate batch indices    # S
        unsafe_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[1]                            # U
        domain_list = np.arange(NUM_BATCHES[3]) % NUM_BATCHES[2]                            # D

        for epoch in range(superp.EPOCHS): # train for a number of epochs
            # initialize epoch
            epoch_loss = 0 # scalar
            lmi_loss = 0 #scalar
            eta_loss = 0
            epoch_gradient_flag = True # gradient is within range
            superp.CURR_MAX_GRAD = 0

            # mini-batches shuffle by shuffling batch indices
            np.random.shuffle(safe_list) 
            np.random.shuffle(unsafe_list)
            np.random.shuffle(domain_list)
            print(NUM_BATCHES[3])

            # train mini-batches
            for batch_index in range(NUM_BATCHES[3]):
                # batch data selection
                batch_safe = batches_safe[safe_list[batch_index]]
                batch_unsafe = batches_unsafe[unsafe_list[batch_index]]
                batch_domain = batches_domain[domain_list[batch_index]]

                ############################## mini-batch training ################################################
                optimizer_barr.zero_grad() # clear gradient of parameters
                optimizer_eta.zero_grad()
                
                curr_batch_loss = loss.calc_loss(barr_nn, batch_safe, batch_unsafe, batch_domain, epoch, batch_index,eta, superp.lip_b)
                # batch_loss is a tensor, batch_gradient is a scalar
                curr_batch_loss.backward() # compute gradient using backward()
                # update weight and bias
                optimizer_barr.step() # gradient descent once
                   
                optimizer_barr.zero_grad()

                curr_lmi_loss= loss.calc_lmi_loss(barr_nn, lambdas, superp.lip_b)
                                
                if curr_lmi_loss >= -5000:
                    curr_lmi_loss.backward()
                    optimizer_barr.step()
                    
                    optimizer_barr.zero_grad()
                
                optimizer_eta.zero_grad()
                
                curr_eta_loss=  loss.calc_eta_loss(eta, superp.lip_b)
                
                if curr_eta_loss > 0:
                    curr_eta_loss.backward()
                    optimizer_eta.step()
                
                # update epoch loss
                epoch_loss += curr_batch_loss.item()
                lmi_loss += curr_lmi_loss
                eta_loss += curr_eta_loss

                if superp.VERBOSE == 1:
                    print("restart: %-2s" % num_restart, "epoch: %-3s" % epoch, "batch: %-5s" % batch_index, "batch_loss: %-25s" % curr_batch_loss.item(), \
                          "epoch_loss: %-25s" % epoch_loss, "lmi loss: % 25s" %lmi_loss, "eta loss: % 25s" %eta_loss, "eta: % 25s" % eta)
                          
            logger.log_kv('epoch', epoch)
            logger.log_kv('epoch_loss', epoch_loss)
            logger.log_kv('lmi_loss', lmi_loss)
            logger.log_kv('eta_loss', eta_loss)

            logger.save_log(log_dir+"/logs")
            make_train_plots(log = logger.log, keys=['epoch', 'epoch_loss'], save_loc=log_dir+"/logs")
            # make_train_plots(log = logger.log, keys=['epoch', 'lmi_loss'], save_loc=log_dir+"logs")


            if (epoch_loss <= 0) and (lmi_loss <= 0) and (eta_loss <= 0):
                print("The last epoch:", epoch, "of restart:", num_restart)
                if superp.VERBOSE == 1:
                    print("\nSuccess! The nn barrier is:")
                    print_nn_matlab(barr_nn) # output the learned model
                    print("\nThe value of eta is:")
                    print(eta)

                return True # epoch success: end of epoch training

    return False
