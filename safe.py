import torch

############################################
# calculate safe controller
############################################

def calc_safe_u(x_domain, h_domain, d_h_domain, d2_h_domain, f_x, g_x, sigma, gamma, eta):
    n_dof = x_domain.shape[1]
    u_ref = (-0)*x_domain[:,0] + (-0)*x_domain[:,1]
    u_ref = u_ref.reshape((-1, 1, 1)).to(d_h_domain.device.type)
    u_safe = 0*u_ref
    l = 0*u_ref
    h_domain = h_domain.reshape((-1, 1, 1))
    d_h_domain = d_h_domain.reshape((-1, 1, n_dof))
    d2_h_domain = d2_h_domain.reshape((-1, n_dof, n_dof))
    f_x = f_x.reshape((-1, n_dof, 1)).to(d_h_domain.device.type)
    g_x = g_x.reshape((-1, n_dof, 1)).to(d_h_domain.device.type)
    A = d_h_domain@f_x
    B = d_h_domain@g_x
    C = torch.matmul(torch.t(sigma).to(d2_h_domain.device.type), torch.matmul(d2_h_domain,sigma.to(d2_h_domain.device.type)))
    psi = A + torch.bmm(B,u_ref) + gamma*h_domain + C +eta

    u_safe = - psi/B
    u_safe[psi>=0] = 0
    u = u_ref + u_safe
    l = A + torch.bmm(B,u) + gamma*h_domain + C

    return u, l
