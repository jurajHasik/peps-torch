import itertools
import numpy as np
import torch
import yamps.yast as yast
from tn_interface_abelian import contract, permute
try:
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import eigs
    from scipy.sparse.linalg import eigsh
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")
from ctm.one_site_c4v_abelian import corrf_c4v

def get_Top_spec_c4v(n, state, env_c4v, verbosity=0):
    raise NotImplementedError()
    # 0) build edge and get symmetric structure
    # (+)0--
    # (-)1--E 
    # (+)2--
    E= corrf_c4v.get_edge(state, env_c4v, verbosity)

    # get symmetry structure of the (grouped) edge and create a dummy Nx1 vector
    # for serialization
    # (+)--M--(+) (-)--V0--(+) = (+)--V1--(+)
    Cs, Ds= E.get_leg_charges_and_dims()
    # get all the possible sectors given by fusion of E-legs
    Cs1= E.s[0]*np.asarray(Cs[0])
    for x in [s*np.asarray(c) for s,c in zip(E.s[1:], Cs[1:])]:
        X= np.add.outer(Cs1, x)
        Cs1= np.unique(X.reshape(len(Cs1)*len(x),1),axis=0)
    Cs1= tuple(map(tuple,Cs1))

    # build dummy Nx1 vector
    Cs= Cs + (Cs1,)
    Ds= Ds + (tuple([1]*len(Cs1)),)
    v0= yast.zeros(config= E.config, s=np.concatenate((E.s,[1])), n=0, t=Cs, D=Ds)
    # r1d, meta= v0.compress_to_1d()
    r1d, meta= E.compress_to_1d()
    N= r1d.numel()

    # multiply vector by transfer-op and pass the result back to numpy
    # Assume the following index structure
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (chi)
    def _mv(v):
        V1d= torch.as_tensor(v, dtype=r1d.dtype, device=r1d.device)
        # bring 1d vector into edge structure
        # (+)0--
        # (-)1--E--(-) dummy-index
        # (+)2--
        # pdb.set_trace()
        V= yast.decompress_from_1d(V1d, E.config, meta)
        V= corrf_c4v.apply_TM_1sO(state,env_c4v,V,verbosity=verbosity)
        # V= V.flip_signature(inplace=True)
        # bring the edge back into plain 1d representation
        V1d, _meta= V.compress_to_1d()
        return V1d.detach().cpu().numpy()

    T= LinearOperator((N,N), matvec=_mv)
    # vals= eigs(T, k=n, v0=None, return_eigenvectors=False)
    vals= eigsh(T, k=n, return_eigenvectors=False) 

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals

    L= torch.zeros((n,2), dtype=r1d.dtype, device=r1d.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= 0
    # L[:,1]= torch.as_tensor(np.imag(vals))

    return L

# def get_Top2_spec_c4v(n, state, env_c4v, verbosity=0):
#     chi= env_c4v.chi
#     ad= state.get_aux_bond_dims()[0]

#     # multiply vector by transfer-op within torch and pass the result back in numpy
#     #  --0 (chi)
#     # v--1 (D^2)
#     #  --2 (D^2)
#     #  --3 (chi)
#     def _mv(v):
#         V= torch.as_tensor(v,dtype=env_c4v.dtype,device=env_c4v.device)
#         V= V.view(chi,ad*ad,ad*ad,chi)
#         V= corrf_c4v.apply_TM_1sO_2(state,env_c4v,V,verbosity=verbosity)
#         V= V.view(chi*(ad**4)*chi)
#         return V.detach().cpu().numpy()

#     T= LinearOperator((chi*(ad**4)*chi,chi*(ad**4)*chi), matvec=_mv)
#     vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

#     # post-process and return as torch tensor with first and second column
#     # containing real and imaginary parts respectively
#     vals= np.copy(vals[::-1]) # descending order
#     vals= (1.0/np.abs(vals[0])) * vals
#     L= torch.zeros((n,2), dtype=state.dtype, device=state.device)
#     L[:,0]= torch.as_tensor(np.real(vals))
#     L[:,1]= torch.as_tensor(np.imag(vals))

#     return L
