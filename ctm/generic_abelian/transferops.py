import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
import config as cfg
import yastn.yastn as yastn
from ctm.generic_abelian import corrf

def get_Top_spec(n, coord, direction, state, env, edge_t=None,
    eigenvectors=False, verbosity=0):
    r"""
    :param n: number of leading eigenvalues of a transfer operator to compute
    :type n: int
    :param coord: reference site (x,y)
    :type coord: tuple(int,int)
    :param direction: direction of transfer operator. Choices are: (0,-1) for up, 
                      (-1,0) for left, (0,1) for down, and (1,0) for right 
    :type direction: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS_ABELIAN
    :param env: corresponding environment
    :type env: ENV_ABELIAN
    :param eigenvectors: compute eigenvectors
    :type eigenvectors: bool
    :edge_t: edge charge sectors. If ``None`` only trivial sector is computed.
    :edge_t: tuple(int) or tuple(tuple(int))
    :return: leading n-eigenvalues, returned as `n x 2` tensor with first and second column
             encoding real and imaginary part respectively.
    :rtype: np.ndarray

    Compute the leading `n` eigenvalues of width-0 transfer operator of IPEPS::

        --T---------...--T--------            --\               /---
        --A(x,y)----...--A(x+lX,y)-- = \sum_i ---v_i \lambda_i v_i-- 
        --T---------...--T--------            --/               \---

    where `A` is a double-layer tensor. The transfer matrix is given by width-1 channel
    of the same length lX as the unit cell of iPEPS, embedded in environment of T-tensors.

    Other directions are obtained by analogous construction. 
    """
    if edge_t is None: 
        edge_t= (0,) if state.engine.sym.NSYM==1 else (tuple([0]*state.engine.sym.NSYM),)
    # if we grow the TM in right direction
    #
    # (-1,0)--A(x,y)--(1,0)--A(x+1,y)--A(x+2,y)--...--A(x+lX-1,y)==A(x,y)--
    #
    #             up        left       down     right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    r_dir= (-direction[0],-direction[1])

    # depending on the direction, get unit-cell length
    if direction==(1,0):
        N, i_T1, i_T2= state.lX, [(0,-1),0], [(0,1),1] # indices of T-tensors and their relevant legs
    elif direction==(-1,0):
        N, i_T1, i_T2= state.lX, [(0,-1),2], [(0,1),2]
    elif direction==(0,1):
        N, i_T1, i_T2= state.lY, [(-1,0),0], [(1,0),0]    
    elif direction==(0,-1):
        N, i_T1, i_T2= state.lY, [(-1,0),1], [(1,0),2]
    else:
        raise ValueError("Invalid direction: "+str(direction))

    # get TM legs to build edge E0 and get data for mapping to a dense form
    #
    #   -- --T1        --
    # E0-- --a*a(coord)--
    #   -- --T2        --
    E0= yastn.zeros(state.engine, legs=(
        env.T[(coord,i_T1[0])].get_legs(i_T1[1]).conj(),
        state.site(coord).get_legs(dir_to_ind[r_dir]).conj(),
        state.site(coord).get_legs(dir_to_ind[r_dir]),
        env.T[(coord,i_T2[0])].get_legs(i_T2[1]).conj(),
        yastn.Leg(sym=state.engine.sym, s=1, t=edge_t, D=tuple([1]*len(edge_t)))
    ))
    E0= E0.fuse_legs(axes=(0,(1,2),3,4))
    E0_dense, meta0= yastn.compress_to_1d(E0,meta=None)

    # multiply vector by transfer-op and pass the result back in numpy
    #  --0 (approx chi)
    # v--1 (D^2)
    #  --2 (approx chi)
    
    # if state and env are on gpu, the matrix-vector product can be performed
    # there as well. Price to pay is the communication overhead of resulting vector
    def _mv(v):
        c0= coord
        V= yastn.decompress_from_1d(state.engine.backend.to_tensor(
            v,dtype=E0.config.default_dtype,device=E0.config.default_device),
            meta=meta0)
        for i in range(N):
            V= corrf.apply_TM_1sO(c0,direction,state,env,V,verbosity=verbosity)
            c0= (c0[0]+direction[0],c0[1]+direction[1])

        v, meta_v= yastn.compress_to_1d(V,meta=None)
        v= state.engine.backend.to_numpy(v)
        return v

    T= LinearOperator((E0.size,E0.size), matvec=_mv, dtype=state.dtype)
    if eigenvectors:
        vals, vecs= eigs(T, k=n, v0=None, return_eigenvectors=True)
    else:
        vals= eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    # sort by abs value in ascending order, then reverse order to descending
    ind_sorted= np.argsort(np.abs(vals))[::-1]
    vals= vals[ ind_sorted ]
    vals= (1.0/np.abs(vals[0])) * vals
    L= np.zeros((n,2), dtype='float64')
    L[:,0]= np.real(vals)
    L[:,1]= np.imag(vals)

    if eigenvectors:
        pass
        #return L, torch.as_tensor(vecs[:,ind_sorted], device=state.device)
    return L
