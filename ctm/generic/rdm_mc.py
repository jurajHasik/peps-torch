import torch
from torch.utils.checkpoint import checkpoint
from math import prod
from config import _torch_version_check
import config as cfg
from ctm.generic.env import ENV
from ctm.generic.ctm_components import c2x2_LU, c2x2_LD, c2x2_RU, c2x2_RD
from ctm.generic.ctm_projectors import ctm_get_projectors_from_matrices
from ctm.generic.rdm import _cast_to_real, _sym_pos_def_rdm
import ctm.generic.corrf as corrf
import logging

log = logging.getLogger(__name__)

def _col2x3(T_1n1, T_10, a_1n1, a_10, indices):
    i,j,a,b,c,d, m,n,e,f,g,h= indices
    # 0--T_1n1--2->3
    #    |
    #    1->1,2

    #    1
    # 2--a_1n1--4 (or a_10)
    #    3\0 

    # i--T_1n1-----------j =>           0,1
    #    |                              |
    #    0,   1                 i,a,c--|1n1|--j,b,d
    #    0,   1                         |
    #    |    |                         2,3            
    # a--a_1n1(2)--------b 
    # c-------a_1n1*(3)--d
    #    4    5
    V_1n1= torch.einsum(T_1n1[i,:,:,j],[0,1], a_1n1[:,:,a,:,b],[2,0,4], \
        a_1n1[:,:,c,:,d].conj(),[3,1,5], [2,3,4,5])

    #       0->0,1
    #       |
    # 2<-1--T_10--2->3

    #    4     5
    # g--------a_10*()--h
    # e--a_10()---------f
    #    |     |
    #    0     1
    #    0     1
    #    |
    # m--T_10-----------n
    V_10= torch.einsum(T_10[:,:,m,n],[0,1], a_10[:,:,e,:,f],[2,4,0], \
        a_10[:,:,g,:,h].conj(),[3,5,1], [2,3,4,5])

    #         0,1
    #         |
    # i,a,c--|1n1|--j,b,d
    #         |
    #         2,3        
    #         2,3
    #         | 
    # m,e,g--|10 |--n,f,h
    #         |
    #         4,5->2,3
    res=torch.einsum(V_10,[0,1,2,3],V_1n1,[4,5,2,3],[0,1,4,5])
    return res

def _trace_2x3(C2X2_LU, C2X2_RU, T_1n1, T_10, a_1n1, a_10):
    # C2x2_LU--0->0   i
    # |         ->1,2 a,c
    # |
    # |         ->4,5 e,g
    # C2x2_LD--1->3   m
    # \23->6,7

    # j     0<-0--C2x2_RU--1,2->3,4
    # b,d 1,2<-   | 
    # f,h 6,7<-   |
    #   n   5<-3--C2x2_RD

    #         0,1
    #         |
    # i,a,c--|1n1|--j,b,d
    #         |
    #         2,3        
    #         2,3
    #         | 
    # m,e,g--|10 |--n,f,h
    #         |
    #         4,5->2,3

    def _compute_indices(rem,dims):
        vals=[]
        for z in range(1,len(dims)+1):
            vals.append(rem % dims[-z])
            rem= (rem-vals[-1]) // dims[-z]
        vals.reverse()
        return tuple(vals)

    rho_acc= torch.zeros([2]*8,dtype=torch.float64)

    for x in range( prod(C2X2_LU.size()[:6]) ):
        # C2X2_LU
        i,a,c,m,e,g= _compute_indices(x, C2X2_LU.size()[:6])
        print(f"x {x} {(i,a,c,m,e,g)}")

        for y in range( prod(list(C2X2_RU.size()[:3])+list(C2X2_RU.size()[5:])) ):
            # C2X2_RU
            j,b,d,n,f,h= _compute_indices(y, list(C2X2_RU.size()[:3])+list(C2X2_RU.size()[5:]))

            indices= (i,j,a,b,c,d, m,n,e,f,g,h)
            TM= _col2x3(T_1n1, T_10, a_1n1, a_10, indices)

            rho_loc= torch.einsum(C2X2_LU[i,a,c,m,e,g,:,:],[0,1], TM,[2,3,6,7], \
                C2X2_RU[j,b,d,:,:,n,f,h],[4,5], [0,1,2,3,4,5,6,7])
            rho_acc+= rho_loc

            rho_loc= rho_loc.permute(0,2,4,6, 1,3,5,7).contiguous().view(2**4,2**4)
            print(f"y {x} {y} {(i,a,c,m,e,g)} {(j,b,d,n,f,h)} {rho_loc.trace()}")

    return rho_acc

def rdm2x3_mc(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 2x3 patch using strategy:

        1. compute left edge of the network
        2. add extra T-tensor and on-site tensor to the bottom of the left edge
        3. analogously for the right edge, attaching extra T-tensor
           and on-site tensor to the top of the right edge
        4. contract left and right half to obtain final reduced density matrix

    ::

        C--T------------------T-----T-------------------C = C2x2_LU(coord+(0,-1))--T-----C2x2(coord+(2,-1))
        |  |                  |     |                   |   |___________________|--A^+A--|_______________|
        T--A^+A(coord+(0,-1))-A^+A--A^+A(coord+(2,-1))--T   |                   |--A^+A--|               |
        |  |                  |     |                   |   C2x2_LD(coord)---------T-----C2x2(coord+(2,0))
        T--A^+A(coord)--------A^+A--A^+A(coord+(2,0))---T
        |  |                  |     |                   |
        C--T------------------T-----T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s3 s2
        s0 s1 x 

    Left edge <LE(s0;I_l)|           T(s1,s3; I_l,I_r)              |RE(s2;I_r)>

       C--T---\chi                   \chi--T---\chi                 \chi--T---C
       T--x---D^2                     D^2--s3--D^2                   D^2--s2--T
       T--s0--D^2                     D^2--s1--D^2                   D^2--x---T  
       C--T---\chi                   \chi--T---\chi                 \chi--T---C

       (total index s0 x I_l)        total index I_l x s1,s3 x I_r  I_r x s2

    Full reduced density matrix

        rho_{s0,s1,s2,s3} = \sum_{I_l, I_r} <LE(s0;I_l)|T(s1,s3; I_l,I_r)|RE(s2;I_r)>
                          = \sum_{I_l, I_r} rho_{s0,s1,s2,s3; I_l,I_r}

    """
    who="rdm2x3_mc"
    
    #       0->0,1
    #       |
    # 2<-1--T_10--2->3
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_10= env.T[(shift_coord,(0,1))]
    T_10= T_10.view([state.site((shift_coord)).size(3)]*2+[T_10.size(1),T_10.size(2)])
    
    # 0--T_1n1--2->3
    #    |
    #    1->1,2
    vec = (1, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_1n1= env.T[(shift_coord,(0,-1))]
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site((shift_coord)).size(1)]*2+[T_1n1.size(2)])

    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build left part C2x2_LU--C2X2_LD ------------------------------------
    # C2x2_LU--1->0
    # |
    # 0
    # 0
    # C2x2_LD--1
    # \23
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_LD, ([0],[0]))

    
    # C2x2_LU--0->0
    # |         ->1,2
    # |
    # |         ->4,5
    # C2x2_LD--1->3
    # \23->6,7
    C2X2_LU= C2X2_LU.view([T_1n1.size(0)]+[state.site((coord[0],coord[1]-1)).size(4)]*2\
        +[T_10.size(2)]+[state.site(coord).size(4)]*2\
        +[state.site(coord).size(0)]*2)


    # ----- building C2x2_RU ----------------------------------------------------
    vec = (2, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

     # ----- building C2x2_RD ----------------------------------------------------
    vec = (2, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build right part C2X2_RU--C2X2_RD -----------------------------------
    #            0--C2x2_RU--1,2 
    #               1
    #               0
    #         3<-1--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,C2X2_RD,([1],[0]))

    #    0<-0--C2x2_RU--1,2->3,4
    #  1,2<-   | 
    #  6,7<-   |
    #    5<-3--C2x2_RD
    C2X2_RU= C2X2_RU.view([T_1n1.size(3)]+[state.site((coord[0]+2,coord[1]-1)).size(4)]*2\
        +[C2X2_RU.size(1)]*2\
        +[T_10.size(3)]+[state.site((coord[0]+2,coord[1])).size(4)]*2)

    #
    # x   6,7 4,5
    # 0,1 2,3 x
    # rdm= torch.tensordot(C2X2_LU,C2X2_RU,([0,1,2, 5,8,11, 7,10],[0,7,10, 3,4,5, 8,11]))
    rdm= _trace_2x3(C2X2_LU, C2X2_RU, T_1n1, T_10, \
        state.site((coord[0]+1,coord[1]-1)), state.site((coord[0]+1,coord[1])))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = rdm.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm


def rdm2x3_loop(coord, state, env, sym_pos_def=False, use_checkpoint=False, \
    verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 2x3 patch using strategy:

        1. compute left edge of the network
        2. add extra T-tensor and on-site tensor to the bottom of the left edge
        3. analogously for the right edge, attaching extra T-tensor
           and on-site tensor to the top of the right edge
        4. contract left and right half to obtain final reduced density matrix

    ::

        C--T------------------T-----T-------------------C = C2x2_LU(coord+(0,-1))--T-----C2x2(coord+(2,-1))
        |  |                  |     |                   |   |___________________|--A^+A--|_______________|
        T--A^+A(coord+(0,-1))-A^+A--A^+A(coord+(2,-1))--T   |                   |--A^+A--|               |
        |  |                  |     |                   |   C2x2_LD(coord)---------T-----C2x2(coord+(2,0))
        T--A^+A(coord)--------A^+A--A^+A(coord+(2,0))---T
        |  |                  |     |                   |
        C--T------------------T-----T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s3 s2
        s0 s1 x 

    """
    who="rdm2x3_loop"
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build left part C2x2_LU--C2X2_LD ------------------------------------
    # C2x2_LU--1->0
    # |
    # 0
    # 0/23
    # C2x2_LD--1
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_LD, ([0],[0]))

    vec = (1, 0)
    shift_coord_10 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_10= env.T[(shift_coord_10,(0,1))]
    T_10= T_10.view([state.site((shift_coord_10)).size(3)]*2+[T_10.size(1),T_10.size(2)])
    
    vec = (1, -1)
    shift_coord_1n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_1n1= env.T[(shift_coord_1n1,(0,-1))]
    # 0--T_1n1--2->3
    #    1->1,2
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site((shift_coord_1n1)).size(1)]*2+[T_1n1.size(2)])

    # C2x2_LU--0->0
    # |         ->1,2
    # |   
    # |/23->6,7   /->4,5
    # C2x2_LD-----1->3 
    C2X2_LU= C2X2_LU.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(2)]*2\
        +[T_10.size(2)]+[state.site(shift_coord_10).size(2)]*2\
        +[state.site(coord).size(0)]*2)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (2, -1)
    shift_coord_2n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord_2n1,state,env,mode='sl-open',verbosity=verbosity)

     # ----- building C2x2_RD ----------------------------------------------------
    vec = (2, 0)
    shift_coord_20 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord_20,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build right part C2X2_RU--C2X2_RD -----------------------------------
    #         0<-0--C2x2_RU--1,2->3,4 
    #       1,2<-   1
    #       6,7<-   0
    #         5<-1--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,C2X2_RD,([1],[0]))
    C2X2_RU= C2X2_RU.view([T_1n1.size(3)]+[state.site(shift_coord_1n1).size(4)]*2\
        +[state.site(shift_coord_2n1).size(0)]*2\
        +[T_10.size(3)]+[state.site(shift_coord_10).size(4)]*2)

    _loc_bond_dim= state.site(shift_coord_10).size(1)
    rdm_acc=torch.zeros([state.site(coord).size(0)]*8+[_loc_bond_dim**2],\
        device=env.device, dtype=env.dtype)
    
    def _loop_body(C2X2_LU, C2X2_RU, T_10, T_1n1, a_10, a_1n1, i_ip):
        i,ip= i_ip // _loc_bond_dim, i_ip % _loc_bond_dim
        #       i     ip
        # 6(2)--------a_10*(9)--(4)7      
        # 4(2)--a_10(8)---------(4)5
        #       0(3)  1(3) 
        #       0     1
        #       |
        # 2-----T_10---------------3
        TA_10= torch.einsum(T_10,[0,1,2,3], a_10[:,i,:,:,:],[8,4,0,5], a_10[:,ip,:,:,:].conj(),\
            [9,6,1,7], [2,4,6, 3,5,7, 8,9])

        # C2x2_LU--0,1,2
        # |        
        # |        A_i,ip--6,7 
        # C2x2_LD--T_10----5
        # |        | 
        # 3,4      8,9
        TA_10= torch.einsum(C2X2_LU,[0,1,2, 3,4,5, 6,7], TA_10,[3,4,5, 8,9,10, 11,12],\
            [0,1,2, 6,7, 8,9,10, 11,12])

        # 0-----T_1n1---------------3
        #       | 
        #       1     2
        #       1(1)  2(1) 
        # 6(2)--------a_1n1*(9)--(4)7      
        # 4(2)--a_1n1(8)---------(4)5
        #       i     ip
        TA_1n1= torch.einsum(T_1n1,[0,1,2,3], a_1n1[:,:,:,i,:],[8,1,4,5], a_1n1[:,:,:,ip,:].conj(),
            [9,2,6,7], [0,4,6, 3,5,7, 8,9])

        #      3,4      5,6 
        #      |        |    
        # 0----T_1n1----C2x2_RU
        # 1,2--A_i,ip   |
        #               |
        #        7,8,9--C2x2_RD
        TA_1n1= torch.einsum(C2X2_RU,[0,1,2, 3,4, 5,6,7], TA_1n1,[10,11,12, 0,1,2, 8,9],\
            [10,11,12, 8,9, 3,4, 5,6,7,])

        #
        # x   6,7 4,5
        # 0,1 2,3 x
        _loc_rdm= torch.einsum(TA_10,[0,1,2, 3,4, 5,6,7, 8,9], TA_1n1,[0,1,2, 12,13, 10,11, 5,6,7 ],\
            [3,4, 8,9, 10,11, 12,13])
        return _loc_rdm

    tensors= (C2X2_LU, C2X2_RU, T_10, T_1n1, state.site(shift_coord_10), \
        state.site(shift_coord_1n1))
    for i_ip in range(rdm_acc.size()[-1]):
        if use_checkpoint:
            _loc_rdm= checkpoint(_loop_body, *tensors, i_ip)
        else:
            _loc_rdm= _loop_body(*tensors, i_ip)
        rdm_acc[...,i_ip]= _loc_rdm

    rdm= torch.sum(rdm_acc, -1)

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = rdm.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm


def rdm3x2_loop(coord, state, env, sym_pos_def=False, use_checkpoint=False, \
    verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 3x2 patch using strategy:

        1. compute top edge of the network
        2. add extra T-tensor and on-site tensor to the right of the top edge
        3. analogously for the bottom edge, attaching extra T-tensor
           and on-site tensor to the left of the bottom edge
        4. contract top and bottom half to obtain final reduced density matrix

    ::

        C--T-------------------T-------------------C
        |  |                   |                   |
        T--A^+A(coord+(0,-2))--A^+A(coord+(1,-2))--T
        |  |                   |                   |
        T--A^+A(coord+(0,-1))--A^+A(coord+(1,-1))--T
        |  |                   |                   |
        T--A^+A(coord)---------A^+A(coord+(1,0))---T
        |  |                   |                   |
        C--T-------------------T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s2
        s3 s1  
        s0 x 

    """
    who="rdm3x2_loop"

    #         0
    # 1,2<-1--T_1n1
    #      3<-2
    vec = (1, -1)
    shift_coord_1n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_1n1= env.T[shift_coord_1n1,(1,0)]
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(4)]*2+[T_1n1.size(2)])

    # 0
    # T_0n1--2->2,3
    # 1
    vec = (0, -1)
    shift_coord_0n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_0n1= env.T[shift_coord_0n1,(-1,0)]
    T_0n1= T_0n1.view([T_0n1.size(0)]+[T_0n1.size(1)]+[state.site(shift_coord_0n1).size(2)]*2)

    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build top part C2x2_LU--C2X2_RU ------------------------------------
    # C2x2_LU--1 0--C2x2_RU--2,3->6,7
    # |                  |
    # 0->                1->  
    # 0, 1,2             3, 4,5     
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_RU, ([1],[0]))

    C2X2_LU= C2X2_LU.view([T_0n1.size(0)]+[state.site(shift_coord_0n1).size(1)]*2\
        +[T_1n1.size(0)]+[state.site(shift_coord_1n1).size(1)]*2\
        +[C2X2_LU.size(2), C2X2_LU.size(3)])


    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build bottom part C2X2_LD--C2X2_RD -----------------------------------
    #        
    #               3, 4,5     0, 1,2
    #            0->1->        0->
    #  6,7<-2,3--C2x2_LD--1 1--C2x2_RD
    C2X2_RD= torch.tensordot(C2X2_RD,C2X2_LD,([1],[1]))

    C2X2_RD= C2X2_RD.view([T_1n1.size(3)]+[state.site(shift_coord_1n1).size(3)]*2\
        +[T_0n1.size(1)]+[state.site(shift_coord_0n1).size(3)]*2\
        +[C2X2_RD.size(2),C2X2_RD.size(3)])


    # contract two parts
    #          __________  
    #   C2X2_LU  7,8(3,4)|     <=>  x  s2
    #   |_x______5,6(1,2)|          s3 s1
    #   |           |____|          s0 x 
    #   0           2    1
    #   0(1)        2    1(0)
    #   |9,10_______|____|
    #   |3,4        x    |
    #   |C2x2_RD_________|


    _loc_bond_dim= state.site(shift_coord_0n1).size(4)
    rdm_acc=torch.zeros([state.site(coord).size(0)]*8+[_loc_bond_dim**2],\
        device=env.device, dtype=env.dtype)
    
    def _loop_body(C2X2_LU, C2X2_RD, T_0n1, T_1n1, a_0n1, a_1n1, i_ip):
        i,ip= i_ip // _loc_bond_dim, i_ip % _loc_bond_dim
        
        # 0              4(1)     6(1)
        # |              |        |
        # T_0n1--2 2(2)--a_0n1(8)------------i
        # |      3 3(2)-----------a_0n1*(9)--ip
        # 1              5(3)     7(3)
        TA_0n1= torch.einsum(T_0n1,[0,1,2,3], a_0n1[:,:,:,:,i],[8,4,2,5], a_0n1[:,:,:,:,ip].conj(),\
            [9,6,3,7], [0,4,6, 1,5,7, 8,9])

        #       10, 11,12  
        #       |
        #  8,9--TA_0n1--i,ip
        #       |            0, 1,2
        #       |            |
        #  6,7--C2x2_LD------C2x2_RD

        TA_0n1= torch.einsum(C2X2_RD,[0,1,2, 3,4,5, 6,7], TA_0n1,[10,11,12, 3,4,5, 8,9],\
            [0,1,2, 6,7, 10,11,12, 8,9])

        #     4(1)    6(1)            0
        #     |       |               |
        # i---a_1n1(8)-----------1 1--T_1n1
        # ip----------a_1n1*(9)--2 2  |
        #     5(3)    7(3)            3
        TA_1n1= torch.einsum(T_1n1,[0,1,2,3], a_1n1[:,:,i,:,:],[8,4,5,1], a_1n1[:,:,ip,:,:].conj(),
            [9,6,7,2], [0,4,6, 3,5,7, 8,9])

        # C2x2_LU------C2x2_RU--6,7
        # |                  | 
        # 0, 1,2             |
        #              i,ip--TA_1n1--8,9
        #                    |
        #            11,12,  10
        TA_1n1= torch.einsum(C2X2_LU,[0,1,2, 3,4,5, 6,7], TA_1n1,[3,4,5, 10,11,12, 8,9],\
            [0,1,2, 6,7, 10,11,12, 8,9])

        #   x  s2
        #   s3 s1
        #   s0 x 
        _loc_rdm= torch.einsum(TA_1n1,[0,1,2, 6,7, 10,11,12, 8,9], TA_0n1,[10,11,12, 3,4, 0,1,2, 13,14],\
            [3,4, 8,9, 6,7, 13,14])
        return _loc_rdm

    tensors= (C2X2_LU, C2X2_RD, T_0n1, T_1n1, state.site(shift_coord_0n1), \
        state.site(shift_coord_1n1))
    for i_ip in range(rdm_acc.size()[-1]):
        if use_checkpoint:
            _loc_rdm= checkpoint(_loop_body, *tensors, i_ip)
        else:
            _loc_rdm= _loop_body(*tensors, i_ip)
        rdm_acc[...,i_ip]= _loc_rdm

    rdm= torch.sum(rdm_acc, -1)

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = rdm.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm