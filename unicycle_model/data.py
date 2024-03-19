import torch
import unicycle_model.prob as prob
from numpy import linalg as la


#Grid the state space and sample points $x_c$ such that for any point in those grids there is an $x_c$ with ||x-x_c|| \leq epsilon

X=prob.DOMAIN
DIM_S = 3
    
N=[300,300,600] # size should be the dimension of the system #its also the number of samples per each dimension

data_len= N[0]*N[1]*N[2] #total length of data

block_len=[10,10,20]

#sample center of the grids 

l=[];
ep=[];
for d in range(0, DIM_S):
    l.append(X[d][1]-X[d][0])
    ep.append(l[d]/(2*N[d]))
    # print('ep', ep)
    
eps=la.norm(ep,2)

grid=[]
data=[];

#sample data points from the center of the grids

for d in range(0, DIM_S):
    grid.append([(2*i-1)*ep[d] for i in range(1,N[d]+1) ])
    data.append([X[d][0]+grid[d][j] for j in range(0,N[d]) ])
    
#mesh grid to obtain all the data points   
    
N_data = torch.meshgrid(torch.tensor(data[0]),torch.tensor(data[1]),torch.tensor(data[2]))

##################################################################
#generating training data set and dividing into batches
##################################################################

def gen_full_data(mesh):
    flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))] # flatten the list of meshes
    nn_input = torch.stack(flatten, 1) # stack the list of flattened meshes
    return nn_input

def gen_batch_data():
    
    full_domain=gen_full_data(N_data)
    
    def batch_data(full_data, data_length, data_chunks, filter):
        l = list(data_length)
        batch_list = [torch.reshape(full_data, l + [DIM_S])]
        for i in range(DIM_S):
            batch_list = [tensor_block for curr_tensor in batch_list for tensor_block in list(curr_tensor.chunk(int(data_chunks[i]), i))]
        
        batch_list = [torch.reshape(curr_tensor, [-1, DIM_S]) for curr_tensor in batch_list]
        batch_list = [curr_tensor[filter(curr_tensor)] for curr_tensor in batch_list]
        batch_list = [curr_tensor for curr_tensor in batch_list if len(curr_tensor) > 0]

        return batch_list
    
    batch_unsafe=batch_data(full_domain, N, block_len, prob.cons_unsafe)
    batch_safe1=batch_data(full_domain, N, block_len, prob.cons_safe1)
    batch_safe2=batch_data(full_domain, N, block_len, prob.cons_safe2)
    batch_safe3=batch_data(full_domain, N, block_len, prob.cons_safe3)
    batch_safe4=batch_data(full_domain, N, block_len, prob.cons_safe4)
    batch_safe= batch_safe1+batch_safe2+batch_safe3+batch_safe4
    batch_domain=batch_data(full_domain, N, block_len, prob.cons_domain)
    
    return batch_safe, batch_unsafe, batch_domain

    
    
        

        

    
    















