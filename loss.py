import torch
import torch.nn as nn
import superp_init as superp
import prob
import torch.nn.functional as F
import data
import safe

# from deep_differential_network.utils import jacobian, evaluate

############################################
# given the training data, compute the loss
############################################

def lipschitz(lambdas, lip, model):
    
    weights=[];
    layer=0;
    for p in model.parameters():
        if layer % 2 == 0:
            weights.append(p.data)
        layer = layer+1
    
    T= torch.diag(lambdas)
    
    diag_items= [lip**2*torch.eye(superp.DIM_S),  2*T,  torch.eye(1)]
    
    subdiag_items= [torch.matmul(T, weights[0]), weights[-1]]
    
    dpart = torch.block_diag(diag_items[0],diag_items[1],diag_items[2])
    
    spart= F.pad(torch.block_diag(subdiag_items[0],subdiag_items[1]), (0,1, superp.DIM_S, 0))
    
    return dpart-spart-torch.transpose(spart,0,1)
    
    
def calc_loss(barr_nn, x_safe, x_unsafe, x_domain, epoch, batch_index, eta,lip_b):
    # compute loss of init    
    h_safe, d_h_safe, d2_h_safe = barr_nn(x_safe, hessian=True)
    loss_safe = torch.relu(h_safe - superp.gamma + superp.TOL_SAFE - eta) #tolerance

    # compute loss of unsafe
    h_unsafe, d_h_unsafe, d2_h_unsafe = barr_nn(x_unsafe, hessian=True)
    loss_unsafe = torch.relu((- h_unsafe) + superp.lamda + superp.TOL_UNSAFE - eta) #tolerance
    
    # compute loss of domain
    h_domain, d_h_domain, d2_h_domain = barr_nn(x_domain, hessian=True)
    # h_domain = h_domain[:, 0, :]
    # d_h_domain = d_h_domain[:, 0, :]
    # d2_h_domain = d2_h_domain[:, 0, :]

    u = safe.calc_safe_u(x_domain, h_domain, d_h_domain, d2_h_domain)
    gamma = 1
    
    vector_domain = prob.func_f(x_domain, u) # compute vector field at domain
    print('Shape of del h & dynamics', h_domain.shape, d_h_domain.shape, d2_h_domain.shape, vector_domain.shape, x_domain.shape)
    loss_lie=torch.relu(d_h_domain*vector_domain + gamma*h_domain + superp.TOL_LIE - eta)
        
    total_loss = superp.DECAY_SAFE * torch.sum(loss_safe) + superp.DECAY_UNSAFE * torch.sum(loss_unsafe) \
                    + superp.DECAY_LIE * torch.sum(loss_lie) #+ loss_eta
                    
    # return total_loss is a tensor, max_gradient is a scalar
    return total_loss

def calc_lmi_loss(barr_nn,lambdas_b, lip_b):
    
      lmi_loss = -0.001*(torch.logdet(lipschitz(lambdas_b, lip_b, barr_nn)) ) #+ torch.logdet(lipschitz(lambdas_c, lip_c, ctrl_nn))
    
      return lmi_loss

def calc_eta_loss(eta, lip_b):
    
     loss_eta=torch.relu(torch.tensor(lip_b*(prob.L_x)*data.eps+lip_b*data.eps) + eta)
    
     return loss_eta