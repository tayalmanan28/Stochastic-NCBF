import torch
import torch.nn as nn
import superp_init as superp
import torch.nn.functional as F
import safe

############################################
# given the training data, compute the loss
############################################

def lipschitz(lambdas, lip, model, dim_s):
    alpha = 0
    beta = 0.25
    device = model.device.type
    weights=[];
    layer=0;
    for p in model.parameters():
        if layer % 2 == 0:
            weights.append(p.data)
        layer = layer+1

    weights[0] = weights[0][0, :, :]
    weights[1] = weights[1][0, :, :]

    T= torch.diag(lambdas)

    diag_items= [lip**2*torch.eye(dim_s).to(device) + 2*alpha*beta*torch.matmul(torch.t(weights[0]),torch.matmul(T, weights[0])),  2*T,  torch.eye(1).to(device)]
    subdiag_items= [(alpha+beta)*torch.matmul(T, weights[0]), weights[-1]]

    dpart = torch.block_diag(diag_items[0],diag_items[1],diag_items[2])
    spart= F.pad(torch.block_diag(subdiag_items[0],subdiag_items[1]), (0,1, dim_s, 0))

    return dpart-spart-torch.transpose(spart,0,1)

def lipschitz_diff(lambdas, lip, model, dim_s):
    alpha = 0
    beta = 0.25
    device = model.device.type
    weights=[];
    layer=0;
    for p in model.parameters():
        if layer % 2 == 0:
            weights.append(p.data)
        layer = layer+1

    weights[0] = weights[0][0, :, :]
    weights[1] = weights[1][0, :, :]
    weights[1] = torch.matmul(torch.t(weights[0]), torch.diag(torch.flatten(weights[1])))

    T= torch.diag(lambdas)

    diag_items= [lip**2*torch.eye(dim_s).to(device) + 2*alpha*beta*torch.matmul(torch.t(weights[0]),torch.matmul(T, weights[0])),  2*T,  torch.eye(dim_s).to(device)]
    subdiag_items= [(alpha+beta)*torch.matmul(T, weights[0]), weights[-1]]

    dpart = torch.block_diag(diag_items[0],diag_items[1],diag_items[2])
    spart= F.pad(torch.block_diag(subdiag_items[0],subdiag_items[1]), (0,dim_s, dim_s, 0))

    return dpart-spart-torch.transpose(spart,0,1)

def lipschitz_d_diff(lambdas, lip, model, sigma, dim_s):
    alpha = -0.0962
    beta = 0.0962
    device = model.device.type
    weights=[];
    layer=0;
    for p in model.parameters():
        if layer % 2 == 0:
            weights.append(p.data)
        layer = layer+1

    weights[0] = weights[0][0, :, :]
    weights[1] = weights[1][0, :, :]
    weights[1] = torch.matmul(torch.t(sigma), torch.matmul(torch.t(weights[0]), torch.matmul(torch.diag(torch.flatten(weights[1])), torch.diag(torch.flatten(torch.matmul(weights[0], sigma))))))

    T= torch.diag(lambdas)

    diag_items= [lip**2*torch.eye(dim_s).to(device) + 2*alpha*beta*torch.matmul(torch.t(weights[0]),torch.matmul(T, weights[0])),  2*T,  torch.eye(1).to(device)]
    subdiag_items= [(alpha+beta)*torch.matmul(T, weights[0]), weights[-1]]

    dpart = torch.block_diag(diag_items[0],diag_items[1],diag_items[2])
    spart= F.pad(torch.block_diag(subdiag_items[0],subdiag_items[1]), (0,1, dim_s, 0))

    return dpart-spart-torch.transpose(spart,0,1)


def calc_loss(barr_nn, x_safe, x_unsafe, x_domain, epoch, batch_index, eta, lip_h, sigma, prob):
    # compute loss of init
    h_safe, d_h_safe, d2_h_safe = barr_nn(x_safe, hessian=True)

    device = h_safe.device.type
    if device != 'cpu':
        eta = eta.cuda(device)

    loss_safe = torch.relu(-h_safe + superp.TOL_SAFE -eta) #tolerance

    # compute loss of unsafe
    h_unsafe, d_h_unsafe, d2_h_unsafe = barr_nn(x_unsafe, hessian=True)
    loss_unsafe = torch.relu(h_unsafe + superp.lamda - superp.TOL_UNSAFE -eta) #tolerance

    # compute loss of domain
    h_domain, d_h_domain, d2_h_domain = barr_nn(x_domain, hessian=True)

    h_domain = h_domain[:, 0, :]
    d_h_domain = d_h_domain[:, 0, :]
    d2_h_domain = d2_h_domain[:, :, 0, :]

    f_x = prob.func_f(x_domain)
    g_x = prob.func_g(x_domain)

    gamma = 1

    u, l = safe.calc_safe_u(x_domain, h_domain, d_h_domain, d2_h_domain, f_x, g_x, sigma, gamma, eta)

    loss_lie=torch.relu(-l.to(device) + superp.TOL_LIE -eta)
    loss_lie_eta=torch.relu(-l.to(device))

    total_loss =  superp.DECAY_SAFE * torch.sum(loss_safe) +  superp.DECAY_UNSAFE * torch.sum(loss_unsafe) \
                    + superp.DECAY_LIE * torch.sum(loss_lie)

    return torch.sum(loss_safe), torch.sum(loss_unsafe), torch.sum(loss_lie), torch.sum(loss_lie_eta), total_loss

def calc_lmi_loss(barr_nn, lambda_h, lambda_dh, lambda_d2h, lip_h, lip_dh, lip_d2h, sigma, dim_s):
    lip_h = torch.tensor(lip_h)
    lip_dh = torch.tensor(lip_dh)
    lip_d2h = torch.tensor(lip_d2h)
    device = barr_nn.device.type
    if device != 'cpu':
        lambda_h = lambda_h.cuda(device)
        lip_h = lip_h.cuda(device)
        lambda_dh = lambda_dh.cuda(device)
        lip_dh = lip_dh.cuda(device)
        lambda_d2h = lambda_d2h.cuda(device)
        lip_d2h = lip_d2h.cuda(device)
        sigma = sigma.cuda(device)
    lmi_loss = -0.001*(torch.logdet(lipschitz(lambda_h, lip_h, barr_nn, dim_s)) + torch.logdet(lipschitz_diff(lambda_dh, lip_dh, barr_nn, dim_s)) + torch.logdet(lipschitz_d_diff(lambda_d2h, lip_d2h, barr_nn, sigma, dim_s)))

    return lmi_loss

def calc_eta_loss(eta, lip_h, lip_dh, lip_d2h, L_x, eps):
    loss_eta=torch.relu(torch.tensor((lip_h+lip_dh*L_x + lip_d2h)*eps) + eta)
    return loss_eta
