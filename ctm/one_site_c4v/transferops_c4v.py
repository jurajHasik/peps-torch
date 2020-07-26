import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import ipeps
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import corrf_c4v

def get_Top_spec_c4v(n, state, env_c4v, verbosity=0):
    chi= env_c4v.chi
    ad= state.get_aux_bond_dims()[0]

    # multiply vector by transfer-op within torch and pass the result back in numpy
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (chi)
    def _mv(v):
        V= torch.as_tensor(v,dtype=env_c4v.dtype,device=env_c4v.device)
        V= V.view(chi,ad*ad,chi)
        V= corrf_c4v.apply_TM_1sO(state,env_c4v,V,verbosity=verbosity)
        V= V.view(chi*ad*ad*chi)
        return V.detach().cpu().numpy()

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

def get_Top2_spec_c4v(n, state, env_c4v, verbosity=0):
    chi= env_c4v.chi
    ad= state.get_aux_bond_dims()[0]

    # multiply vector by transfer-op within torch and pass the result back in numpy
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (D^2)
    #  --3 (chi)
    def _mv(v):
        V= torch.as_tensor(v,dtype=env_c4v.dtype,device=env_c4v.device)
        V= V.view(chi,ad*ad,ad*ad,chi)
        V= corrf_c4v.apply_TM_1sO_2(state,env_c4v,V,verbosity=verbosity)
        V= V.view(chi*(ad**4)*chi)
        return V.detach().cpu().numpy()

    T= LinearOperator((chi*(ad**4)*chi,chi*(ad**4)*chi), matvec=_mv)
    vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=state.dtype, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    return L
