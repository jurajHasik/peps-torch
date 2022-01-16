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
    #             up        left       down     right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    ad= state.site(coord).size( dir_to_ind[direction] )

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

def get_EH_spec_Ttensor(n, L, coord, direction, state, env, verbosity=0):
    r"""
    Compute leading part of spectrum of exp(EH), where EH is boundary
    Hamiltonian. Exact exp(EH) is given by the leading eigenvector of 
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

    The exp(EH) is then given by reshaping (D^2)^L leading eigenvector of TM
    into D^L x D^L operator.

    We approximate the exp(EH) of L-leg cylinder as MPO formed by T-tensors
    of the CTM environment. Then, the spectrum of this approximate exp(EH)
    is obtained through iterative solver using matrix-vector product

           0
           |                __
         --T(x,y)------  --|  |
         --T(x,y+1)----  --|v0|
          ...           ...|  |
         --T(x,y+L-1)--  --|__|
           0(PBC)
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
    d= ind_to_dir[ dir_to_ind[direction]-1 + ((4-dir_to_ind[direction]+1)//4)*4 ]
    ads= [ state.site( (coord[0]+i*d[0],coord[1]+i*d[1]) )\
        .size( dir_to_ind[direction] ) for i in range(L) ]

    def _get_and_transform_T(coord):
        if direction==(0,-1):
            #                              3
            # 0--T--2->1 => 0--T--1 ==> 0--T--1
            #    1->2          2           2
            return env.T[(coord,(0,-1))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(0,-1)])]*2)
        elif direction==(-1,0):
            # 0          0
            # T--2 => 2--T--3
            # 1          1
            return env.T[(coord,(-1,0))].view([chi]*2\
                + [state.site(coord).size(dir_to_ind[(-1,0)])]*2)
        elif direction==(0,1):
            #    0->2         2           2
            # 1--T--2   => 0--T--1 ==> 0--T--1
            # ->0   ->1                   3
            return env.T[(coord,(0,1))].permute(1,2,0).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(0,1)])]*2)
        elif direction==(1,0):
            #       0          0         0
            # 2<-1--T =>    2--T  =>  2--T--3
            #    1<-2          1         1
            return env.T[(coord,(1,0))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(1,0)])]*2)

    def _mv(v0):
        V= torch.as_tensor(v0,dtype=env.dtype,device=env.device)
        V= V.view(ads)
        
        # 0) apply 0th T
        #
        #    0                       0                L-1+2<-0
        # 2--T--3 0--V--1..L-1 -> 2--T--V--3..L-1+2 -> 1<-2--T--V--2..L-1+1
        #    1                       1                    0<-1
        c= state.vertexToSite(coord)
        V= torch.tensordot(_get_and_transform_T(c),V,([3],[0]))
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
            c= state.vertexToSite( (c[0]+d[0],c[1]+d[1]) )
            V= torch.tensordot(_get_and_transform_T(c),\
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
        c= state.vertexToSite( (c[0]+d[0],c[1]+d[1]) )
        V= torch.tensordot(_get_and_transform_T(c),\
            V,([0,3,1],[0,L-1+1,L-1+2]))
        V= V.permute(list(range(L-1,-1,-1)))
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
    Compute leading part of spectrum of exp(EH), where EH is boundary
    Hamiltonian. Exact exp(EH) is given by the leading eigenvector of 
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

    The exp(EH) is then given by reshaping (D^2)^L leading eigenvector of TM
    into D^L x D^L operator.

    We approximate the exp(EH) of L-leg cylinder as MPO formed by T-tensors
    of the CTM environment. Then, the spectrum of this approximate exp(EH)
    is obtained through full diagonalization

           0
           |           
         --T(x,y)------
         --T(x,y+1)----
          ...          
         --T(x,y+L-1)--
           0(PBC)
    """
    chi= env.chi
    #             up        left       down      right
    dir_to_ind= {(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4}
    ind_to_dir= dict(zip(dir_to_ind.values(), dir_to_ind.keys()))
    #
    # TM in direction (1,0) [right], grow in direction (0,1) [down]
    # TM in direction (0,1) [down], grow in direction (-1,0) [left]
    d= ind_to_dir[ dir_to_ind[direction]-1 + ((4-dir_to_ind[direction]+1)//4)*4 ]
    if verbosity>0:
        print(f"transferops.get_full_EH_spec_Ttensor direction {direction}"\
            +f" growth d {d}")

    def _get_and_transform_T(coord):
        if direction==(0,-1):
            #                              3
            # 0--T--2->1 => 0--T--1 ==> 0--T--1
            #    1->2          2           2
            return env.T[(coord,(0,-1))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(0,-1)])]*2)
        elif direction==(-1,0):
            # 0          0
            # T--2 => 2--T--3
            # 1          1
            return env.T[(coord,(-1,0))].view([chi]*2\
                + [state.site(coord).size(dir_to_ind[(-1,0)])]*2)
        elif direction==(0,1):
            #    0->2         2           2
            # 1--T--2   => 0--T--1 ==> 0--T--1
            # ->0   ->1                   3
            return env.T[(coord,(0,1))].permute(1,2,0).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(0,1)])]*2)
        elif direction==(1,0):
            #       0          0         0
            # 2<-1--T =>    2--T  =>  2--T--3
            #    1<-2          1         1
            return env.T[(coord,(1,0))].permute(0,2,1).contiguous().view(\
                [chi]*2 + [state.site(coord).size(dir_to_ind[(1,0)])]*2)

    c= state.vertexToSite(coord)
    if L==1:
        EH= torch.einsum('iilr->lr',_get_and_transform_T(c))
        D,U= torch.eig(EH)
        D_abs, inds= torch.sort(D.abs(),descending=True)
        D_sorted= D[inds]/D_abs[0]
        return D_sorted

    #    0
    # 2--T--3 =>    T--1,2;3
    #    1          0
    EH= _get_and_transform_T(c).permute(1,2,3,0)
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
        c= state.vertexToSite( (c[0]+d[0],c[1]+d[1]) )
        EH= torch.tensordot(_get_and_transform_T(c),EH,([0],[0]))

    #
    #    T--2L-3,2L-2;2L-1  => T--2L-2,2L-1 
    #   ...                   ...
    #    T--1,2                T--2,3
    #    0                     |
    #    0                     |
    # 2--T--3                  T--0,1
    #    1
    #
    c= state.vertexToSite( (c[0]+d[0],c[1]+d[1]) )
    EH= torch.tensordot(_get_and_transform_T(c),EH,([0,1],[0,2*L-1]))
    EH= EH.permute(list(range(0,2*L,2))+list(range(1,2*L+1,2))).contiguous()\
        .view(np.prod(EH.size()[0:L]),np.prod(EH.size()[L:]))
    D,U= torch.eig(EH)
    D_abs, inds= torch.sort(D.abs(),descending=True)
    D_sorted= D[inds]/D_abs[0]
    return D_sorted