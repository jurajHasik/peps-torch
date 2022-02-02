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

    _test_T= torch.zeros(1,dtype=env_c4v.dtype)
    T= LinearOperator((chi*ad*ad*chi,chi*ad*ad*chi), matvec=_mv, \
        dtype="complex128" if _test_T.is_complex() else "float64")
    vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=torch.float64, device=state.device)
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
    L= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    return L

def get_EH_spec_Ttensor(n, L, state, env_c4v, verbosity=0):
    r"""
    Compute leading part of spectrum of exp(EH), where EH is boundary
    Hamiltonian. Exact exp(EH) is given by the leading eigenvector of 
    transfer matrix ::
          
         ...         PBC                          /
          |           |                  |     --a*--
        --A--       --A(0)--           --A-- =  /| 
        --A--       --A(1)--             |       |/
        --A--        ...                       --a--
          |         --A(L-1)--                  /
         ...          |
                     PBC

        infinite exact TM; exact TM of L-leg cylinder  

    The exp(EH) is then given by reshaping (D^2)^L leading eigenvector of TM
    into D^L x D^L operator.

    We approximate the exp(EH) of L-leg cylinder as MPO formed by T-tensors
    of the CTM environment. Then, the spectrum of this approximate exp(EH)
    is obtained through iterative solver using matrix-vector product

           0
           |            __
         --T(0)----  --|  |
         --T(1)----  --|v0|
          ...       ...|  |
         --T(L-1)--  --|__|
           0(PBC)
    """
    assert L>1,"L must be larger than 1"
    chi= env_c4v.chi
    ad= state.site().size(4)
    T= env_c4v.get_T().view(chi,chi,ad,ad)

    def _mv(v0):
        V= torch.as_tensor(v0,dtype=env_c4v.dtype,device=env_c4v.device)
        V= V.view([ad]*L)
        
        # 0) apply 0th T
        #
        #    0                       0                L-1+2<-0
        # 2--T--3 0--V--1..L-1 -> 2--T--V--3..L-1+2 -> 1<-2--T--V--2..L-1+1
        #    1                       1                    0<-1
        V= torch.tensordot(T,V,([3],[0]))
        V= V.permute([1,2]+list(range(3,L-1+3))+[0])

        # 1) apply 1st to L-2th T
        for i in range(1,L-1):
            #                    _
            #   i<-i-1--T-------|V|--i+1..L-1+1,L-1+2
            # i-1<-1-2--T-------| |
            #          ...     
            #     2<-1--T-------| |
            #           0       | |
            #           0       | |
            #     1<-2--T--3 i--|_|
            #           1->0
            #
            V= torch.tensordot(T,V,([0,3],[0,i+1]))


        # apply L-1th T
        #                     _
        #   L-1--T-----------|V|--L-1+2
        #   L-2--T-----------| |
        #       ...
        #     1--T-----------| |
        #        0           | |
        #        0           | |
        #  0<-2--T--3 L-1+1--|_|
        #        1
        #       PBC
        #
        V= torch.tensordot(T,V,([0,3,1],[0,L-1+1,L-1+2]))
        V= V.permute(list(range(L-1,-1,-1)))
        return V.cpu().numpy()

    _test_T= torch.zeros(1,dtype=env_c4v.dtype)
    expEH= LinearOperator((ad**L,ad**L), matvec=_mv, \
        dtype="complex128" if _test_T.is_complex() else "float64")
    vals= eigs(expEH, k=n, v0=None, return_eigenvectors=False)

    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    S= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    S[:,0]= torch.as_tensor(np.real(vals))
    S[:,1]= torch.as_tensor(np.imag(vals))

    return S