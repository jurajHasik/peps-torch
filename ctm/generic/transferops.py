import warnings
import torch
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import ipeps
from ctm.generic.env import ENV
from ctm.generic import corrf

def get_Top_w0_spec(n, coord, direction, state, env, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of transfer operator. Choices are: (0,-1) for up, 
                      (-1,0) for left, (0,1) for down, and (1,0) for right 
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: corresponding environment
    :type env: ENV_C4V
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor   

    Compute the leading `n` eigenvalues of width-0 transfer operator of IPEPS::

        --T(x,y)----...--T(x+lX,y)----            --\               /---
          |              |             = \sum_i     v_i \lambda_i v_i 
        --T(x,y+1)--...--T(x+lX,y+1)--            --/               \---

    where `A` is a double-layer tensor. The transfer matrix is given by width-0 channel
    of the same length lX as the unit cell of iPEPS, embedded in environment of T-tensors.

    Other directions are obtained by analogous construction.
    """
    chi= env.chi
    #             up        left       down     right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}

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
        V= V.view(chi,chi)
        for i in range(N):
            V= corrf.apply_TM_0sO(c0,direction,state,env,V,verbosity=verbosity)
            c0= (c0[0]+direction[0],c0[1]+direction[1])
        V= V.view(chi**2)
        v= V.cpu().numpy()
        return v

    _test_T= torch.zeros(1,dtype=env.dtype)
    T= LinearOperator((chi**2,chi**2), matvec=_mv, \
        dtype="complex128" if _test_T.is_complex() else "float64")
    vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    # sort by abs value in ascending order, then reverse order to descending
    ind_sorted= np.argsort(np.abs(vals))
    vals= vals[ ind_sorted[::-1] ]
    # vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    return L

def get_Top_spec(n, coord, direction, state, env, eigenvectors=False, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of transfer operator. Choices are: (0,-1) for up, 
                      (-1,0) for left, (0,1) for down, and (1,0) for right 
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: corresponding environment
    :type env: ENV_C4V
    :param eigenvectors: compute eigenvectors
    :type eigenvectors: bool
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor   

    Compute the leading `n` eigenvalues of width-0 transfer operator of IPEPS::

        --T---------...--T--------            --\               /---
        --A(x,y)----...--A(x+lX,y)-- = \sum_i ---v_i \lambda_i v_i-- 
        --T---------...--T--------            --/               \---

    where `A` is a double-layer tensor. The transfer matrix is given by width-1 channel
    of the same length lX as the unit cell of iPEPS, embedded in environment of T-tensors.

    Other directions are obtained by analogous construction. 
    """
    chi= env.chi
    # if we grow the TM in right direction
    #
    # (-1,0)--A(x,y)--(1,0)--A(x+1,y)--A(x+2,y)--...--A(x+lX-1,y)==A(x,y)--
    #
    #             up        left       down     right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    ad= state.site(coord).size( dir_to_ind[(-direction[0],-direction[1])] )

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

    _test_T= torch.zeros(1,dtype=env.dtype)
    T= LinearOperator((chi*ad*ad*chi,chi*ad*ad*chi), matvec=_mv, \
        dtype="complex128" if _test_T.is_complex() else "float64")
    if eigenvectors:
        vals, vecs= eigs(T, k=n, v0=None, return_eigenvectors=True)
    else:
        vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    # sort by abs value in ascending order, then reverse order to descending
    ind_sorted= np.argsort(np.abs(vals))[::-1]
    vals= vals[ ind_sorted ]
    # vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    L= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= torch.as_tensor(np.imag(vals))

    if eigenvectors:
        return L, torch.as_tensor(vecs[:,ind_sorted], device=state.device)
    return L

def get_EH_spec_Ttensor(n, L, coord, direction, state, env, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param L: width of the cylinder
    :type L: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of the transfer operator. Either
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: torch.Tensor

    Compute leading part of spectrum of :math:`exp(EH)`, where EH is boundary
    Hamiltonian. Exact :math:`exp(EH)` is given by the leading eigenvector of 
    transfer matrix ::
          
         ...                PBC                                /
          |                  |                        |     --a*--
        --A(x,y)----       --A(x,y)------           --A-- =  /| 
        --A(x,y+1)--       --A(x,y+1)----             |       |/
        --A(x,y+2)--        ...                             --a--
          |                --A(x,y+L-1)--                    /
         ...                 |
                            PBC

        infinite exact TM; exact TM of L-leg cylinder  

    The :math:`exp(EH)` is then given by

    .. math::
        
        exp(-H_{ent}) = \sqrt{\sigma_R}\sigma_L\sqrt{\sigma_R}

    where :math:`\sigma_L,\sigma_R` are reshaped :math:`(D^2)^L` left and right 
    leading eigenvectors of TM into :math:`D^L \times D^L` operator. Given that spectrum
    of :math:`AB` is equivalent to :math:`BA`, it is enough to diagonalize
    product :math:`\sigma_R\sigma_L` or :math:`\sigma_R\sigma_L`. 

    We approximate the :math:`\sigma_L,\sigma_R` of L-leg cylinder as MPO formed 
    by T-tensors of the CTM environment. Then, the spectrum of this approximate 
    :math:`exp(EH)` is obtained through iterative solver using matrix-vector product::

           0                    1
           |                    |                    __
         --T[(x,y),(-1,0)]------T[(x,y),(1,0)]------|  |
         --T[(x,y+1),(-1,0)]----T[(x,y+1),(1,0)]----|v0|
          ...                  ...                  |  |
         --T[(x,y+L-1),(-1,0)]--T[(x,y+L-1),(1,0)]--|__|
           0(PBC)               1(PBC)
    """
    assert L>1,"L must be larger than 1"
    assert state.lX==state.lY==1,"only single-site unit cell is supported" #TODO

    chi= env.chi
    #             up        left       down      right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    ind_to_dir= dict(zip(dir_to_ind.values(), dir_to_ind.keys()))
    #
    # TM in direction (1,0) [right], grow in direction (0,1) [down]
    # TM in direction (0,1) [down], grow in direction (-1,0) [left]
    d_grow= ind_to_dir[ dir_to_ind[direction]-1 + ((4-dir_to_ind[direction]+1)//4)*4 ]
    d_opp= (-direction[0],-direction[1])
    ads= [ state.site( (coord[0]+i*d_grow[0],coord[1]+i*d_grow[1]) )\
        .size( dir_to_ind[direction] ) for i in range(L) ]
    if np.prod(ads)<=n:
        warnings.warn("Total dimension of H_ent operator is <= n.",RuntimeWarning)
        return None

    def _get_and_transform_T(c,d=direction):
        if d==(0,-1):
            #                              3
            # 0--T--2->1 => 0--T--1 ==> 0--T--1
            #    1->2          2           2
            return env.T[(c,(0,-1))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(0,-1)])]*2)
        elif d==(-1,0):
            # 0          0
            # T--2 => 2--T--3
            # 1          1
            return env.T[(c,(-1,0))].view([chi]*2\
                + [state.site(c).size(dir_to_ind[(-1,0)])]*2)
        elif d==(0,1):
            #    0->2         2           2
            # 1--T--2   => 0--T--1 ==> 0--T--1
            # ->0   ->1                   3
            return env.T[(c,(0,1))].permute(1,2,0).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(0,1)])]*2)
        elif d==(1,0):
            #       0          0         0
            # 2<-1--T =>    2--T  =>  2--T--3
            #    1<-2          1         1
            return env.T[(c,(1,0))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(1,0)])]*2)

    def mv_sigma(V,d_sigma,d_grow):
        # 0) apply 0th T
        #
        #    0                       0                L-1+2<-0
        # 2--T--3 0--V--1..L-1 -> 2--T--V--3..L-1+2 -> 1<-2--T--V--2..L-1+1
        #    1                       1                    0<-1
        c= state.vertexToSite(coord)
        V= torch.tensordot(_get_and_transform_T(c,d_sigma),V,([3],[0]))
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
            c= state.vertexToSite( (c[0]+d_grow[0],c[1]+d_grow[1]) )
            V= torch.tensordot(_get_and_transform_T(c,d_sigma),\
                V,([0,3],[0,i+1]))


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
        c= state.vertexToSite( (c[0]+d_grow[0],c[1]+d_grow[1]) )
        V= torch.tensordot(_get_and_transform_T(c,d_sigma),\
            V,([0,3,1],[0,L-1+1,L-1+2]))
        V= V.permute(list(range(L-1,-1,-1))).contiguous()
        return V

    def _mv(v0):
        V= torch.as_tensor(v0,dtype=env.dtype,device=env.device)
        V= V.view(ads)
        V= mv_sigma(V,direction,d_grow)
        V= mv_sigma(V,d_opp,d_grow)
        V= V.view(np.prod(ads))
        return V.cpu().numpy()

    _test_T= torch.zeros(1,dtype=env.dtype)
    expEH= LinearOperator((np.prod(ads),np.prod(ads)), matvec=_mv, \
        dtype="complex128" if _test_T.is_complex() else "float64")
    vals= eigs(expEH, k=n, v0=None, return_eigenvectors=False)

    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals
    S= torch.zeros((n,2), dtype=torch.float64, device=state.device)
    S[:,0]= torch.as_tensor(np.real(vals))
    S[:,1]= torch.as_tensor(np.imag(vals))

    return S

def get_full_EH_spec_Ttensor(L, coord, direction, state, env, \
        verbosity=0):
    r"""
    :param L: width of the cylinder
    :type L: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of the transfer operator. Either
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS_C4V
    :param env_c4v: corresponding environment
    :type env_c4v: ENV_C4V
    :return: leading n-eigenvalues, returned as rank-1 tensor 
    :rtype: torch.Tensor

    Compute the leading part of spectrum of :math:`exp(EH)`, where EH is boundary
    Hamiltonian. Exact :math:`exp(EH)` is given by the leading eigenvector of 
    transfer matrix::
          
         ...                PBC                                /
          |                  |                        |     --a*--
        --A(x,y)----       --A(x,y)------           --A-- =  /| 
        --A(x,y+1)--       --A(x,y+1)----             |       |/
        --A(x,y+2)--        ...                             --a--
          |                --A(x,y+L-1)--                    /
         ...                 |
                            PBC

        infinite exact TM; exact TM of L-leg cylinder  

    The :math:`exp(EH)` is then given by

    .. math::
        
        exp(-H_{ent}) = \sqrt{\sigma_R}\sigma_L\sqrt{\sigma_R}

    where :math:`\sigma_L,\sigma_R` are reshaped (D^2)^L left and right 
    leading eigenvectors of TM into :math:`D^L \times D^L` operator. Given that spectrum
    of :math:`AB` is equivalent to :math:`BA`, it is enough to diagonalize
    product :math:`\sigma_R\sigma_L` or :math:`\sigma_R\sigma_L`. 

    We approximate the :math:`\sigma_L,\sigma_R` of L-leg cylinder as MPO formed 
    by T-tensors of the CTM environment. Then, the spectrum of this approximate 
    exp(EH) is obtained through full diagonalization::

           0                    1
           |                    |
         --T[(x,y),(-1,0)]------T[(x,y),(1,0)]------
         --T[(x,y+1),(-1,0)]----T[(x,y+1),(1,0)]----
          ...                  ...
         --T[(x,y+L-1),(-1,0)]--T[(x,y+L-1),(1,0)]--
           0(PBC)               1(PBC)
    """
    chi= env.chi
    #             up        left       down      right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    ind_to_dir= dict(zip(dir_to_ind.values(), dir_to_ind.keys()))
    #
    # TM in direction (1,0) [right], grow in direction (0,1) [down]
    # TM in direction (0,1) [down], grow in direction (-1,0) [left]
    d_grow= ind_to_dir[ dir_to_ind[direction]-1 + ((4-dir_to_ind[direction]+1)//4)*4 ]
    d_opp= (-direction[0],-direction[1])
    if verbosity>0:
        print(f"transferops.get_full_EH_spec_Ttensor direction {direction}"\
            +f" growth d {d}")

    def _get_and_transform_T(c,d=direction):
        #
        # Return T-tensor as rank-4 tensor by permuting (bra,ket) aux index of T-tensor
        # to last position and then opening it
        #
        if d==(0,-1):
            #                              3
            # 0--T--2->1 => 0--T--1 ==> 0--T--1
            #    1->2          2           2
            return env.T[(c,(0,-1))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(0,-1)])]*2)
        elif d==(-1,0):
            # 0          0
            # T--2 => 2--T--3
            # 1          1
            return env.T[(c,(-1,0))].view([chi]*2\
                + [state.site(c).size(dir_to_ind[(-1,0)])]*2)
        elif d==(0,1):
            #    0->2         2           2
            # 1--T--2   => 0--T--1 ==> 0--T--1
            # ->0   ->1                   3
            return env.T[(c,(0,1))].permute(1,2,0).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(0,1)])]*2)
        elif d==(1,0):
            #       0          0         0
            # 2<-1--T =>    2--T  =>  2--T--3
            #    1<-2          1         1
            return env.T[(c,(1,0))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(c).size(dir_to_ind[(1,0)])]*2)

    if L==1:
        c= state.vertexToSite(coord)
        sigma_0= torch.einsum('iilr->lr',_get_and_transform_T(c))
        sigma_1= torch.einsum('iilr->lr',_get_and_transform_T(c,d_opp))
        D,U= torch.eig(sigma_0@sigma_1)
        D_abs, inds= torch.sort(D.abs(),descending=True)
        D_sorted= D[inds]/D_abs[0]

        return D_sorted

    def get_sigma(d_sigma,d_grow):
        c= state.vertexToSite(coord)
        #    0
        # 2--T--3 =>    T--1,2;3
        #    1          0
        sigma= _get_and_transform_T(c,d_sigma).permute(1,2,3,0)
        for i in range(1,L-1):
            #    
            #    T--2i-1,2i;2i+1  =>   T--2i+1,2i+2;2i+3
            #   ...                   ...
            #    T--1,2                T--3,4
            #    0                     |
            #    0                     |
            # 2--T--3                  T--1,2
            #    1                     0
            #
            c= state.vertexToSite( (c[0]+d_grow[0],c[1]+d_grow[1]) )
            sigma= torch.tensordot(_get_and_transform_T(c,d_sigma),sigma,([0],[0]))

        #
        #    T--2L-3,2L-2;2L-1  => T--2L-2,2L-1 
        #   ...                   ...
        #    T--1,2                T--2,3
        #    0                     |
        #    0                     |
        # 2--T--3                  T--0,1
        #    1
        #
        c= state.vertexToSite( (c[0]+d_grow[0],c[1]+d_grow[1]) )
        sigma= torch.tensordot(_get_and_transform_T(c,d_sigma),sigma,([0,1],[0,2*L-1]))
        sigma= sigma.permute(list(range(0,2*L,2))+list(range(1,2*L+1,2))).contiguous()\
            .view(np.prod(sigma.size()[0:L]),np.prod(sigma.size()[L:]))
        return sigma

    sigma_0= get_sigma(direction,d_grow)
    sigma_1= get_sigma(d_opp,d_grow)

    D,U= torch.eig(sigma_0@sigma_1)
    D_abs, inds= torch.sort(D.abs(),descending=True)
    D_sorted= D[inds]/D_abs[0]
    return D_sorted