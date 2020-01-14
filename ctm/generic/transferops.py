import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import ipeps
from ctm.generic.env import ENV
from ctm.generic import corrf

def get_Top_spec(n, coord, direction, state, env, verbosity=0):
    chi= env.chi
    ad= state.get_aux_bond_dims()[0]

    # depending on the direction, get unit-cell length
    if direction==(1,0) or direction==(-1,0):
        N= state.lX
    elif direction==(0,1) or direction==(0,-1):
        N= state.lY
    else:
        raise ValueError("Invalid direction: "+str(direction))

    # multiply vector by transfer-op within torch and pass the result back in numpy
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (chi)
    
    # if state and env are on gpu, the matrix-vector product can be performed
    # there as well. Price to pay is the communication overhead of resulting vector
    def _mv(v):
        c0= coord
        V= torch.as_tensor(v,device=state.device)
        V= V.view(chi,ad*ad,chi)
        for i in range(N):
            V= corrf.apply_TM_1sO(c0,direction,state,env,V,verbosity=verbosity)
            c0= (c0[0]+direction[0],c0[1]+direction[1])
        V= V.view(chi*ad*ad*chi)
        v= V.cpu().numpy()
        return v

    T= LinearOperator((chi*ad*ad*chi,chi*ad*ad*chi), matvec=_mv)
    vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=state.dtype, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    return L
