import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import ipeps
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v import corrf_c4v

def get_Top_spec_c4v(n, state, env_c4v, normalize=True, eigenvectors=False, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :param normalize: normalize eigenvalues such that :math:`\lambda_0=1`
    :type normalize: bool
    :param eigenvectors: compute eigenvectors
    :type eigenvectors: bool
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor   

    Compute the leading `n` eigenvalues of width-1 transfer operator of 1-site C4v symmetric iPEPS::

        --T--          --\               /---
        --A-- = \sum_i ---v_i \lambda_i v_i-- 
        --T--          --/               \---

    where `A` is a double-layer tensor.
    """
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
    if eigenvectors:
        vals, vecs= eigs(T, k=n, v0=None, return_eigenvectors=True)
    else:
        vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    ind_sorted= np.argsort(np.abs(vals))[::-1] # descending order
    vals= vals[ind_sorted]
    if normalize: 
        vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    if eigenvectors:
        return L, torch.as_tensor(vecs[:,ind_sorted], device=state.device)
    return L

def get_Top2_spec_c4v(n, state, env_c4v, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor   

    Compute the leading `n` eigenvalues of width-2 transfer operator of 1-site C4v symmetric iPEPS::

        --T--          --\               /---
        --A--          --\               /---
        --A-- = \sum_i ---v_i \lambda_i v_i-- 
        --T--          --/               \---

    where `A` is a double-layer tensor.
    """
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
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param L: width of the cylinder
    :type L: int
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor
    
    Compute the leading part of spectrum of :math:`exp(EH)`, where `EH` is boundary
    Hamiltonian. Exact :math:`exp(EH)` is given by the leading eigenvector of 
    a transfer matrix ::
          
         ...         PBC                          /
          |           |                  |     --a*--
        --A--       --A(0)--           --A-- =  /| 
        --A--       --A(1)--             |       |/
        --A--        ...                       --a--
          |         --A(L-1)--                  /
         ...          |
                     PBC

        infinite exact TM; exact TM of L-leg cylinder  

    The :math:`exp(EH)` is then given by reshaping the :math:`(D^2)^L` leading eigenvector 
    of transfer matrix into :math:`D^L \times D^L` operator.

    We approximate the :math:`exp(EH)` of L-leg cylinder as MPO formed by T-tensors
    of the CTM environment. Then, the spectrum of this approximate :math:`exp(EH)`
    is obtained through iterative solver using matrix-vector product::

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