import torch


############################################
# calculate safe controller
############################################
def calc_safe_u(x_domain, h_domain, d_h_domain, d2_h_domain, f_x, g_x,sigma,gamma):
    u_ref = (-0.1)*x_domain[:,0] + (-0.1)*x_domain[:,1] 
    u_safe = 0*u_ref
    l = 0*u_ref
    for i in range(len(x_domain[0])):
        u_r = u_ref[i]
        h = h_domain[i]
        dh = d_h_domain[i]
        d2h = d2_h_domain[i]
        f = f_x[i].to(dh.device.type)
        g = g_x[i].to(dh.device.type)
        # print('shape of h, dh, d2h', h.shape, dh.shape, d2h.shape, f.shape, g.shape, sigma.shape)
        A = torch.dot(dh,f)
        B = torch.dot(dh,g)
        C = torch.dot(sigma.to(dh.device.type), torch.matmul(d2h,sigma.to(dh.device.type)))

        psi = A + B*u_r + C + gamma*h
        if psi>=0:
            u_safe[i] = - psi/(B)
        
        l[i] = A + B*(u_r + u_safe[i]) + C + gamma*h
        
    u = u_ref + u_safe

    return u, l