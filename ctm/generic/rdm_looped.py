import opt_einsum as oe
import torch
import warnings
from torch.utils.checkpoint import checkpoint
from math import prod
from config import _torch_version_check
import config as cfg
from ctm.generic.env import ENV
from ctm.generic.ctm_components import c2x2_LU, c2x2_LD, c2x2_RU, c2x2_RD
from ctm.generic.ctm_projectors import ctm_get_projectors_from_matrices
from ctm.generic.rdm import _cast_to_real, _sym_pos_def_rdm, get_contraction_path, contract_with_unroll
import ctm.generic.corrf as corrf
try:
    import opt_einsum as oe
    from oe_ext.oe_ext import get_contraction_path, contract_with_unroll, _debug_allocated_tensors
except:
    oe=False
    warnings.warn("opt_einsum not available.")
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

def _find_unrolled(to_unroll,*interleaved_exp): 
    if (to_unroll is None) or not to_unroll: return []
    indices= set(sum(interleaved_exp[1::2],start=[]))
    intersection= set.intersection(set(to_unroll),indices)
    return list(intersection)

# mode 1: all tensors are moved to CPU and contraction is evaluated on CPU
# mode 2: all is evaluated on current device (assumed to be the same for all tensors)
# mode 3: all is stored on CPU, all is evaluated on GPU under checkpointing (move to GPU happens within checkpointed section)

def rdm2x3_loop(coord, state, env, sym_pos_def=False, checkpoint_unrolled=False, \
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
    :return: 6-site reduced density matrix with indices 
             :math:`s_0...s5;s'_0...s'_5`
    :rtype: torch.tensor

    Computes 6-site reduced density matrix :math:`\rho` of six-site subsystem, 
    specified by the vertex ``coord`` of its lower-left 
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
    are left uncontracted and given in the order::

        s3 s4 s5
        s0 s1 s2

    """
    who="rdm2x3_loop"
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build left part C2x2_LU--C2X2_LD ------------------------------------
    #                         /23(LU),45(LD)
    # C2x2_LU--1->0  permute  C2x2_LU--0
    # |\23->12                |
    # 0                       |
    # 0/23->45                |
    # C2x2_LD--1->3           C2x2_LD--1
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_LD, ([0],[0]))
    C2X2_LU= C2X2_LU.permute(0,3,1,2,4,5)

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

    # /23,45->67(LU),89(LD)
    # C2x2_LU--0->0
    # |         ->1,2
    # |   
    # |           /->4,5
    # C2x2_LD-----1->3 
    C2X2_LU= C2X2_LU.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(2)]*2\
        +[T_10.size(2)]+[state.site(shift_coord_10).size(2)]*2\
        +[state.site(shift_coord).size(0)]*2+[state.site(coord).size(0)]*2 )

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (2, -1)
    shift_coord_2n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord_2n1,state,env,mode='sl-open',verbosity=verbosity)

     # ----- building C2x2_RD ----------------------------------------------------
    vec = (2, 0)
    shift_coord_20 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord_20,state,env,mode='sl-open',verbosity=verbosity)
    
    # ----- build right part C2X2_RU--C2X2_RD -----------------------------------
    #         0<-0--C2x2_RU--2,3->1,2  permute  0--C2x2_RU--23(RU),45(RD)
    #               1                              |
    #               0                              |
    #         3<-1--C2x2_RD--2,3->4,5           1--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,C2X2_RD,([1],[0]))
    C2X2_RU= C2X2_RU.permute(0,3,1,2,4,5)
    #         0<-0--C2x2_RU--23,45->67(RU),89(RD)
    #       1,2<-   |
    #       4,5<-   |
    #         3<-1--C2x2_RD
    C2X2_RU= C2X2_RU.view([T_1n1.size(3)]+[state.site(shift_coord_1n1).size(4)]*2\
        +[T_10.size(3)]+[state.site(shift_coord_10).size(4)]*2\
        +[state.site(shift_coord_2n1).size(0)]*2+[state.site(shift_coord_20).size(0)]*2)

    _loc_bond_dim= state.site(shift_coord_10).size(1)
    rdm_acc=torch.zeros([state.site(coord).size(0)]*12+[_loc_bond_dim**2],\
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
        # |                13,14
        # |                |
        # |                A_i,ip--11,12 
        # C2x2_LD----------T_10----10
        # |        
        # 67(LU),89(LD)
        TA_10= torch.einsum(C2X2_LU,[0,1,2, 3,4,5, 6,7,8,9], TA_10,[3,4,5, 10,11,12, 13,14],\
            [0,1,2, 10,11,12, 6,7,8,9,13,14])

        # 0-----T_1n1---------------3
        #       | 
        #       1     2
        #       1(1)  2(1) 
        # 6(2)--------a_1n1*(9)--(4)7      
        # 4(2)--a_1n1(8)---------(4)5
        #       i     ip
        TA_1n1= torch.einsum(T_1n1,[0,1,2,3], a_1n1[:,:,:,i,:],[8,1,4,5], a_1n1[:,:,:,ip,:].conj(),
            [9,2,6,7], [0,4,6, 3,5,7, 8,9])

        #                       67(RU),89(RD)
        #                       |    
        #  10----T_1n1----------C2x2_RU
        # 11,12--A_i,ip         |
        #        |              |
        #        13,14          |
        #                3,4,5--C2x2_RD
        TA_1n1= torch.einsum(C2X2_RU,[0,1,2, 3,4,5, 6,7,8,9], TA_1n1,[10,11,12, 0,1,2, 13,14],\
            [10,11,12, 3,4,5, 6,7,8,9,13,14])

        #
        # 6,7 8,9 10,11
        # 0,1 2,3 4,5
        _loc_rdm= torch.einsum(TA_10,[0,1,2, 10,11,12, 6,7,8,9,13,14], TA_1n1,[0,1,2, 10,11,12, 15,16,17,18,19,20 ],\
            [8,9, 13,14, 17,18, 6,7, 19,20, 15,16])
        return _loc_rdm

    tensors= (C2X2_LU, C2X2_RU, T_10, T_1n1, state.site(shift_coord_10), \
        state.site(shift_coord_1n1))
    for i_ip in range(rdm_acc.size()[-1]):
        if checkpoint_unrolled:
            _loc_rdm= checkpoint(_loop_body, *tensors, i_ip)
        else:
            _loc_rdm= _loop_body(*tensors, i_ip)
        rdm_acc[...,i_ip]= _loc_rdm

    rdm= torch.sum(rdm_acc, -1)

    # permute into order of s0,...,s5;s0',...,s5' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = rdm.permute(0,2,4,6,8,10, 1,3,5,7,9,11).contiguous()
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm

def rdm2x3_loop_trglringex_manual(coord, state, env, sym_pos_def=False, checkpoint_unrolled=False, \
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
        if checkpoint_unrolled:
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

def rdm2x3_loop_oe(coord, state, env, open_sites=[0,1,2,3,4,5], unroll=True,\
    sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
    checkpoint_on_device=False,verbosity=0):
    # C1------(1)1 1(0)----T1----(3)44 44(0)----T1_x----(3)39 39(0)---T1_2x---(3)24 24(0)--C2_2x
    # 0(0)               (1,2)                 (1,2)                  (1,2)                25(1)
    # 0(0)           100  2  5             102 40 42              104 26 28                25(0)
    # |                 \ 2  5               \ |  |                  \ |  |                 |  
    # T4-------(2)3 3-----a--|------45 45----a_x--6(1)----41  41-------a_2x-------27 27(1)--T2_2x
    # |                   |  |                 |  |                    |  |                 |
    # |        (3)6 6-------a*------46 46--------a*_x-----43  43----------a*_2x---29 29(2)  |
    # 15(1)               16 17 \101          47 48 \103               37 38 \105          36(3)           
    # 15(0)          106  16 17           108 47 48                    37 38               36(0) 
    # |                 \ |   |              \ |  |               110\ |  |                 |
    # T4_y--(2)9 9--------a_y-------20 20-----a_xy--------49 49(1)----a_2xy-------33 33(1)--T2_2xy
    # |                   |   |                |  |                    |  |                 |
    # |     (3)12 12---------a*_y---22 22------- a*_xy----50 50(2)--------a*_2xy--35 35(2)  |
    # |                   10 13 \107           21 23 \109             32 34 \111            |                   
    # 8(1)                10 13                21 23                  32 34                 31(3)
    # 8(0)                (0,1)                (0,1)                  (0,1)                 31(0)
    # C4_y---(1)7 7(2)-----T3_y--(3)19 19(2)----T3_xy---(3)51 51(2)---T3_2xy--(3)30 30(1)---C3_2xy
    ind_os= set(sorted(open_sites))
    assert len(ind_os)==len(open_sites),"contains repeated elements"
    assert ind_os <= {0,1,2,3,4,5},"allowed site labels are 0,1,2,3,4, and 5"
    I= sum([[100+2*x,100+2*x+1] if x in ind_os else [100+2*x]*2 for x in [0,1,2,3,4,5]],[])
    I_out= [100+2*x for x in ind_os]+[100+2*x+1 for x in ind_os]

    who=f"rdm2x3_{ind_os}"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_y= state.site( (coord[0],coord[1]+1) )
    a_xy= state.site( (coord[0]+1,coord[1]+1) )
    a_2x= state.site( (coord[0]+2,coord[1]) )
    a_2xy= state.site( (coord[0]+2,coord[1]+1) )
    C1, C2_2x, C3_2xy, C4_y= env.C[(state.vertexToSite( coord ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+2,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,1))]
    T1, T4, T3_y, T4_y, T3_xy, T2_2xy, T3_2xy, T1_2x, T2_2x, T1_x= \
        env.T[(state.vertexToSite( coord ),(0,-1))],\
        env.T[(state.vertexToSite( coord ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))]
       
    t= C1, C2_2x, C3_2xy, C4_y, T1, T4, T3_y, T4_y, T3_xy, T2_2xy, T3_2xy, T1_2x, T2_2x, T1_x,\
        a, a_x, a_y, a_xy, a_2x, a_2xy
    if force_cpu:
       t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T1_2x= T1_2x.view(T1_2x.size(0),a_2x.size(1),a_2x.size(1),T1_2x.size(2))
    T2_2x= T2_2x.view(T2_2x.size(0),a_2x.size(4),a_2x.size(4),T2_2x.size(2))
    T2_2xy= T2_2xy.view(T2_2xy.size(0),a_2xy.size(4),a_2xy.size(4),T2_2xy.size(2))
    T3_y= T3_y.view(a_y.size(3),a_y.size(3),T3_y.size(1),T3_y.size(2))
    T3_xy= T3_xy.view(a_xy.size(3),a_xy.size(3),T3_xy.size(1),T3_xy.size(2))
    T3_2xy= T3_2xy.view(a_2xy.size(3),a_2xy.size(3),T3_2xy.size(1),T3_2xy.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))

    contract_tn= C1,[0,1],T1,[1,2,5,44],T4,[0,15,3,6],a,[I[0],2,3,16,45],a.conj(),[I[1],5,6,17,46],\
        T4_y,[15,8,9,12],C4_y,[8,7],T3_y,[10,13,7,19],a_y,[I[6],16,9,10,20],a_y.conj(),[I[7],17,12,13,22],\
        T3_xy,[21,23,19,51],a_xy,[I[8],47,20,21,49],a_xy.conj(),[I[9],48,22,23,50],\
        T1_2x,[39,26,28,24],C2_2x,[24,25],T2_2x,[25,27,29,36],a_2x,[I[4],26,41,37,27],a_2x.conj(),[I[5],28,43,38,29],\
        T2_2xy,[36,33,35,31],C3_2xy,[31,30],T3_2xy,[32,34,51,30],a_2xy,[I[10],37,49,32,33],a_2xy.conj(),[I[11],38,50,34,35],\
        T1_x,[44,40,42,39],a_x,[I[2],40,45,47,41],a_x.conj(),[I[3],42,46,48,43],I_out
    names= tuple(x.strip() for x in ("C1, T1, T4, a, a*, T4_y, C4_y, T3_y, a_y, a_y*, T3_xy, a_xy, a_xy*, "\
        +"T1_2x, C2_2x, T2_2x, a_2x, a_2x*, T2_2xy, C3_2xy, T3_2xy, a_2xy, a_2xy*, T1_x, a_x, a_x*").split(','))
    
    # Memory limit avoids following contraction
    #
    # |C2X2  |--
    # |      |==
    # |      |==   ||
    # |C2X2_y|-- --T_xy-- 
    #
    _tmp_a=(a,a_x,a_2x,a_y,a_xy,a_2xy)
    mem_limit= env.chi**2 * a.size(4)**2 * max(a_y.size(4)**2,a_xy.size(4)**2) \
        * prod([_tmp_a[x].size(0)**2 for x in ind_os if x in [0,3,4]])
    if type(unroll)==bool and unroll:
        unroll= [47,48]
    path, path_info= get_contraction_path(*contract_tn,unroll=unroll if unroll else [],\
        names=names,path=None,who=who,\
        memory_limit=mem_limit if unroll else None,optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,optimize=path,backend='torch',\
        unroll=unroll if unroll else [],checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R

def rdm2x3_loop_oe_semimanual(coord, state, env, open_sites=[0,1,2,3,4,5], unroll=True,\
    sym_pos_def=False, force_cpu=False, 
    checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0):
    # C1------(1)1 1(0)----T1----(3)44 44(0)----T1_x----(3)39 39(0)---T1_2x---(3)24 24(0)--C2_2x
    # 0(0)               (1,2)                 (1,2)                  (1,2)                25(1)
    # 0(0)           100  2  5             102 40 42              104 26 28                25(0)
    # |                 \ 2  5               \ |  |                  \ |  |                 |  
    # T4-------(2)3 3-----a--|------45 45----a_x--6(1)----41  41-------a_2x-------27 27(1)--T2_2x
    # |                   |  |                 |  |                    |  |                 |
    # |        (3)6 6-------a*------46 46--------a*_x-----43  43----------a*_2x---29 29(2)  |
    # 15(1)               16 17 \101          47 48 \103               37 38 \105          36(3)           
    # 15(0)          106  16 17           108 47 48                    37 38               36(0) 
    # |                 \ |   |              \ |  |               110\ |  |                 |
    # T4_y--(2)9 9--------a_y-------20 20-----a_xy--------49 49(1)----a_2xy-------33 33(1)--T2_2xy
    # |                   |   |                |  |                    |  |                 |
    # |     (3)12 12---------a*_y---22 22------- a*_xy----50 50(2)--------a*_2xy--35 35(2)  |
    # |                   10 13 \107           21 23 \109             32 34 \111            |                   
    # 8(1)                10 13                21 23                  32 34                 31(3)
    # 8(0)                (0,1)                (0,1)                  (0,1)                 31(0)
    # C4_y---(1)7 7(2)-----T3_y--(3)19 19(2)----T3_xy---(3)51 51(2)---T3_2xy--(3)30 30(1)---C3_2xy
    ind_os= set(sorted(open_sites))
    assert len(ind_os)==len(open_sites),"contains repeated elements"
    assert ind_os <= {0,1,2,3,4,5},"allowed site labels are 0,1,2,3,4, and 5"
    I= sum([[100+2*x,100+2*x+1] if x in ind_os else [100+2*x]*2 for x in [0,1,2,3,4,5]],[])
    I_out= [100+2*x for x in ind_os]+[100+2*x+1 for x in ind_os]
    I_left_out= [100+2*x for x in ind_os if x in [0,3]]+[100+2*x+1 for x in ind_os if x in [0,3]]
    I_right_out= [100+2*x for x in ind_os if x in [2,5]]+[100+2*x+1 for x in ind_os if x in [2,5]]

    who=f"rdm2x3_oe_semimanual_{ind_os}"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_y= state.site( (coord[0],coord[1]+1) )
    a_xy= state.site( (coord[0]+1,coord[1]+1) )
    a_2x= state.site( (coord[0]+2,coord[1]) )
    a_2xy= state.site( (coord[0]+2,coord[1]+1) )
    C1, C2_2x, C3_2xy, C4_y= env.C[(state.vertexToSite( coord ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+2,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,1))]
    T1, T4, T3_y, T4_y, T3_xy, T2_2xy, T3_2xy, T1_2x, T2_2x, T1_x= \
        env.T[(state.vertexToSite( coord ),(0,-1))],\
        env.T[(state.vertexToSite( coord ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+2,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))]
       
    t= C1, C2_2x, C3_2xy, C4_y, T1, T4, T3_y, T4_y, T3_xy, T2_2xy, T3_2xy, T1_2x, T2_2x, T1_x,\
        a, a_x, a_y, a_xy, a_2x, a_2xy
    if force_cpu:
       t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T1_2x= T1_2x.view(T1_2x.size(0),a_2x.size(1),a_2x.size(1),T1_2x.size(2))
    T2_2x= T2_2x.view(T2_2x.size(0),a_2x.size(4),a_2x.size(4),T2_2x.size(2))
    T2_2xy= T2_2xy.view(T2_2xy.size(0),a_2xy.size(4),a_2xy.size(4),T2_2xy.size(2))
    T3_y= T3_y.view(a_y.size(3),a_y.size(3),T3_y.size(1),T3_y.size(2))
    T3_xy= T3_xy.view(a_xy.size(3),a_xy.size(3),T3_xy.size(1),T3_xy.size(2))
    T3_2xy= T3_2xy.view(a_2xy.size(3),a_2xy.size(3),T3_2xy.size(1),T3_2xy.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))
    
    # left edge
    left_tn= C1,[0,1],T1,[1,2,5,44],T4,[0,15,3,6],a,[I[0],2,3,16,45],a.conj(),[I[1],5,6,17,46],\
        T4_y,[15,8,9,12],C4_y,[8,7],T3_y,[10,13,7,19],a_y,[I[6],16,9,10,20],a_y.conj(),[I[7],17,12,13,22],\
        [44,45,46,20,22,19]+I_left_out
    left_names= tuple(x.strip() for x in ("C1, T1, T4, a, a*, T4_y, C4_y, T3_y, a_y, a_y*").split(','))
    unroll_L=_find_unrolled(unroll,*left_tn)
    path, path_info= get_contraction_path(*left_tn,\
        names=left_names,path=None,unroll=unroll_L,who=who+"_L",memory_limit=None,optimizer="default" if env.chi>1 else "auto")
    L= contract_with_unroll(*left_tn,optimize=path,who=who+"_L",backend='torch',
        unroll=unroll_L,checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,verbosity=verbosity)

    # right edge
    right_tn= T1_2x,[39,26,28,24],C2_2x,[24,25],T2_2x,[25,27,29,36],a_2x,[I[4],26,41,37,27],a_2x.conj(),[I[5],28,43,38,29],\
        T2_2xy,[36,33,35,31],C3_2xy,[31,30],T3_2xy,[32,34,51,30],a_2xy,[I[10],37,49,32,33],a_2xy.conj(),[I[11],38,50,34,35],\
        [39,41,43,49,50,51]+I_right_out
    right_names= tuple(x.strip() for x in ("T1_2x, C2_2x, T2_2x, a_2x, a_2x*, T2_2xy, C3_2xy, T3_2xy, a_2xy, a_2xy*").split(','))
    unroll_R=_find_unrolled(unroll,*right_tn)
    path, path_info= get_contraction_path(*right_tn,\
        names=right_names,path=None,unroll=unroll_R,who=who+"_R",memory_limit=None,optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*right_tn,optimize=path,who=who+"_R",backend='torch',
        unroll=unroll_R,checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,verbosity=verbosity)

    joint_tn= L,[44,45,46,20,22,19]+I_left_out,\
        T3_xy,[21,23,19,51],a_xy,[I[8],47,20,21,49],a_xy.conj(),[I[9],48,22,23,50],\
        T1_x,[44,40,42,39],a_x,[I[2],40,45,47,41],a_x.conj(),[I[3],42,46,48,43],\
        R,[39,41,43,49,50,51]+I_right_out,I_out
    names= tuple(x.strip() for x in ("L, T3_xy, a_xy, a*_xy, T1_x, a_x, a*_x, R").split(','))
    
    # Memory limit forces following contraction
    #
    # |C2X2  |--                      |C2X2  |--
    # |      |==                      |      |==
    # |      |== ==a_xy==             |      |==   ||
    # |C2X2_y|-- --T_xy--  instead of |C2X2_y|-- --T_xy-- 
    # 
    if type(unroll)==bool and unroll:
        unroll= [47,48]
    path, path_info= get_contraction_path(*joint_tn,unroll=unroll if unroll else [],\
        names=names,path=None,who=who,memory_limit=L.numel()*a_xy.size(0)**2 if unroll else None,\
            optimizer="default" if env.chi>1 else "auto") 
    res= contract_with_unroll(*joint_tn,optimize=path,backend='torch',
        unroll=unroll if unroll else [],checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,who=who,verbosity=verbosity)

    res = _sym_pos_def_rdm(res, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        res= res.to(env.device)
    return res


def rdm3x2_loop_trglringex_manual(coord, state, env, sym_pos_def=False, checkpoint_unrolled=False, \
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
        if checkpoint_unrolled:
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

def rdm3x2_loop_oe_semimanual(coord, state, env, open_sites=[0,1,2,3,4,5], unroll=True,\
    sym_pos_def=False, force_cpu=False, 
    checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0):
    # C1------(1)1 1(0)----T1----(3)13 13(0)----T1_x-----(3)7 7(0)-----C2_x
    # 0(0)               (1,2)                 (1,2)                   8(1)
    # 0(0)           100  2  5             106 9 11                    8(0)
    # |                 \ 2  5               \ |  |                     |  
    # T4-------(2)3 3-----a--|------14 14----a_x--6(1)----10 10(1)-----T2_x
    # |                   |  |                 |  |                     |
    # |        (3)6 6-------a*------15 15--------a*_x-----12 12(2)      |
    # 18(1)               19 20 \101          85 86 \107               87(3)           
    # 18(0)          102  19 20           108 85 86                    87(0) 
    # |                 \ |   |              \ |  |                     |
    # T4_y--(2)16 16-----a_y--------83 83-----a_xy--------55 55(1)------T2_xy
    # |                   |   |                |  |                     |
    # |     (3)17 17---------a*_y---84 84--------a*_xy----56 56(2)------|
    # 80(1)               81 82 \103          58 59 \109               57(3)     
    # 80(0)           104 81 82           110 58 59                    57(0) 
    # |                 \ |   |              \ |  |                     |
    # T4_2y--(2)48 48----a_2y-------53 53------a_x2y------43 43(1)-----T2_x2y
    # |                   |   |                |  |                     |
    # |      (3)50 50--------a*_2y--54 54------- a*_x2y---45 45(2)------|
    # |                   49 51 \105           42 44 \111               |                   
    # 47(1)               49 51                42 44                   41(3)
    # 47(0)               (0,1)                (0,1)                   41(0)
    # C4_2y--(1)46 46(2)--T3_2y---(3)52 52(2)--T3_x2y---(3)40 40(1)---C3_x2y
    ind_os= set(sorted(open_sites))
    assert len(ind_os)==len(open_sites),"contains repeated elements"
    assert ind_os <= {0,1,2,3,4,5},"allowed site labels are 0,1,2,3,4, and 5"
    I= sum([[100+2*x,100+2*x+1] if x in ind_os else [100+2*x]*2 for x in [0,1,2,3,4,5]],[])
    I_out= [100+2*x for x in ind_os]+[100+2*x+1 for x in ind_os]
    I_top_out= [100+2*x for x in ind_os if x in [0,3]]+[100+2*x+1 for x in ind_os if x in [0,3]]
    I_bottom_out= [100+2*x for x in ind_os if x in [2,5]]+[100+2*x+1 for x in ind_os if x in [2,5]]

    who=f"rdm3x2_oe_semimanual_{ind_os}"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_y= state.site( (coord[0],coord[1]+1) )
    a_xy= state.site( (coord[0]+1,coord[1]+1) )
    a_2y= state.site( (coord[0],coord[1]+2) )
    a_x2y= state.site( (coord[0]+1,coord[1]+2) )
    C1, C2_x, C3_x2y, C4_2y= env.C[(state.vertexToSite( coord ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+2) ), (-1,1))]
    T1, T4, T4_y, T4_2y, T1_x, T2_x, T2_xy, T2_x2y, T3_x2y, T3_2y= \
        env.T[(state.vertexToSite( coord ),(0,-1))],\
        env.T[(state.vertexToSite( coord ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+2) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+2) ), (0,1))]

       
    t= C1, C2_x, C3_x2y, C4_2y, T1, T4, T4_y, T4_2y, T1_x, T2_x, T2_xy, T2_x2y, T3_x2y, T3_2y,\
        a, a_x, a_y, a_xy, a_2y, a_x2y
    if force_cpu:
       t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T2_x= T2_x.view(T2_x.size(0),a_x.size(4),a_x.size(4),T2_x.size(2))
    T2_xy= T2_xy.view(T2_xy.size(0),a_xy.size(4),a_xy.size(4),T2_xy.size(2))
    T2_x2y= T2_x2y.view(T2_x2y.size(0),a_x2y.size(4),a_x2y.size(4),T2_x2y.size(2))
    T3_2y= T3_2y.view(a_2y.size(3),a_2y.size(3),T3_2y.size(1),T3_2y.size(2))
    T3_x2y= T3_x2y.view(a_x2y.size(3),a_x2y.size(3),T3_x2y.size(1),T3_x2y.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))
    T4_2y= T4_2y.view(T4_2y.size(0),T4_2y.size(1),a_2y.size(2),a_2y.size(2))
    
    # top edge
    top_tn= C1,[0,1],T1,[1,2,5,13],T4,[0,18,3,6],a,[I[0],2,3,19,14],a.conj(),[I[1],5,6,20,15],\
        T1_x,[13,9,11,7],C2_x,[7,8],T2_x,[8,10,12,87],a_x,[I[6],9,14,85,10],a_x.conj(),[I[7],11,15,86,12],\
        [18,19,20,85,86,87]+I_top_out
    top_names= tuple(x.strip() for x in ("C1, T1, T4, a, a*, T1_x, C2_x, T2_y, a_x, a_x*").split(','))
    unroll_TE=_find_unrolled(unroll,*top_tn)
    path, path_info= get_contraction_path(*top_tn,\
        names=top_names,path=None,unroll=unroll_TE,who=who+"_TE",memory_limit=None,\
            optimizer="default" if env.chi>1 else "auto")
    TE= contract_with_unroll(*top_tn,optimize=path,who=who+"_TE",backend='torch',
        unroll=unroll_TE,checkpoint_unrolled=checkpoint_unrolled,\
        checkpoint_on_device=checkpoint_on_device,verbosity=verbosity)

    # bottom edge
    bottom_tn= T3_x2y,[42,44,52,40],C3_x2y,[41,40],T2_x2y,[57,43,45,41],a_x2y,[I[10],58,53,42,43],a_x2y.conj(),[I[11],59,54,44,45],\
        T4_2y,[80,47,48,50],C4_2y,[47,46],T3_2y,[49,51,46,52],a_2y,[I[4],81,48,49,53],a_2y.conj(),[I[5],82,50,51,54],\
        [80,81,82,58,59,57]+I_bottom_out
    bottom_names= tuple(x.strip() for x in ("T3_x2y, C3_x2y, T2_x2y, a_x2y, a_x2y*, T4_2y, C4_2y, T3_2y, a_2y, a_2y*").split(','))
    unroll_BE=_find_unrolled(unroll,*bottom_tn)
    path, path_info= get_contraction_path(*bottom_tn,\
        names=bottom_names,path=None,unroll=unroll_BE,who=who+"_BE",memory_limit=None,\
            optimizer="default" if env.chi>1 else "auto")
    BE= contract_with_unroll(*bottom_tn,optimize=path,who=who+"_BE",backend='torch',
        unroll=unroll_BE,checkpoint_unrolled=checkpoint_unrolled,\
        checkpoint_on_device=checkpoint_on_device,verbosity=verbosity)

    joint_tn= TE,[18,19,20,85,86,87]+I_top_out,\
        T4_y,[18,80,16,17],a_y,[I[2],19,16,81,83],a_y.conj(),[I[3],20,17,82,84],\
        T2_xy,[87,55,56,57],a_xy,[I[8],85,83,58,55],a_xy.conj(),[I[9],86,84,59,56],\
        BE,[80,81,82,58,59,57]+I_bottom_out,I_out
    names= tuple(x.strip() for x in ("TE, T4_y, a_y, a*_y, T2_xy, a_xy, a*_xy, BE").split(','))
    
    # Memory limit forces following contraction (rotated by pi/2 clockwise)
    #
    # |C2X2  |--                   |C2X2  |--
    # |      |==                   |      |==
    # |      |== ==a==             |      |==   ||
    # |C2X2  |-- --T--  instead of |C2X2  |-- --T-- 
    # 
    if type(unroll)==bool and unroll:
        unroll= [83,84]
    path, path_info= get_contraction_path(*joint_tn,unroll=unroll if unroll else [],\
        names=names,path=None,who=who,memory_limit=TE.numel()*a_y.size(0)**2 if unroll else None,\
            optimizer="default" if env.chi>1 else "auto") 
    res= contract_with_unroll(*joint_tn,optimize=path,backend='torch',
        unroll=unroll if unroll else [],checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,who=who,verbosity=verbosity)

    res = _sym_pos_def_rdm(res, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        res= res.to(env.device)
    return res

def rdm3x2_loop_oe(coord, state, env, open_sites=[0,1,2,3,4,5], unroll=True,\
    sym_pos_def=False, force_cpu=False, checkpoint_unrolled=False, 
    checkpoint_on_device=False, verbosity=0):
    # C1------(1)1 1(0)----T1----(3)13 13(0)----T1_x-----(3)7 7(0)-----C2_x
    # 0(0)               (1,2)                 (1,2)                   8(1)
    # 0(0)           100  2  5             106 9 11                    8(0)
    # |                 \ 2  5               \ |  |                     |  
    # T4-------(2)3 3-----a--|------14 14----a_x--6(1)----10 10(1)-----T2_x
    # |                   |  |                 |  |                     |
    # |        (3)6 6-------a*------15 15--------a*_x-----12 12(2)      |
    # 18(1)               19 20 \101          85 86 \107               87(3)           
    # 18(0)          102  19 20           108 85 86                    87(0) 
    # |                 \ |   |              \ |  |                     |
    # T4_y--(2)16 16-----a_y--------83 83-----a_xy--------55 55(1)------T2_xy
    # |                   |   |                |  |                     |
    # |     (3)17 17---------a*_y---84 84--------a*_xy----56 56(2)------|
    # 80(1)               81 82 \103          58 59 \109               57(3)     
    # 80(0)           104 81 82           110 58 59                    57(0) 
    # |                 \ |   |              \ |  |                     |
    # T4_2y--(2)48 48----a_2y-------53 53------a_x2y------43 43(1)-----T2_x2y
    # |                   |   |                |  |                     |
    # |      (3)50 50--------a*_2y--54 54------- a*_x2y---45 45(2)------|
    # |                   49 51 \105           42 44 \111               |                   
    # 47(1)               49 51                42 44                   41(3)
    # 47(0)               (0,1)                (0,1)                   41(0)
    # C4_2y--(1)46 46(2)--T3_2y---(3)52 52(2)--T3_x2y---(3)40 40(1)---C3_x2y
    ind_os= set(sorted(open_sites))
    assert len(ind_os)==len(open_sites),"contains repeated elements"
    assert ind_os <= {0,1,2,3,4,5},"allowed site labels are 0,1,2,3,4, and 5"
    I= sum([[100+2*x,100+2*x+1] if x in ind_os else [100+2*x]*2 for x in [0,1,2,3,4,5]],[])
    I_out= [100+2*x for x in ind_os]+[100+2*x+1 for x in ind_os]

    who=f"rdm3x2_{ind_os}"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_y= state.site( (coord[0],coord[1]+1) )
    a_xy= state.site( (coord[0]+1,coord[1]+1) )
    a_2y= state.site( (coord[0],coord[1]+2) )
    a_x2y= state.site( (coord[0]+1,coord[1]+2) )
    C1, C2_x, C3_x2y, C4_2y= env.C[(state.vertexToSite( coord ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+2) ), (-1,1))]
    T1, T4, T4_y, T4_2y, T1_x, T2_x, T2_xy, T2_x2y, T3_x2y, T3_2y= \
        env.T[(state.vertexToSite( coord ),(0,-1))],\
        env.T[(state.vertexToSite( coord ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+2) ), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+2) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+2) ), (0,1))]

       
    t= C1, C2_x, C3_x2y, C4_2y, T1, T4, T4_y, T4_2y, T1_x, T2_x, T2_xy, T2_x2y, T3_x2y, T3_2y,\
        a, a_x, a_y, a_xy, a_2y, a_x2y
    if force_cpu:
       t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T2_x= T2_x.view(T2_x.size(0),a_x.size(4),a_x.size(4),T2_x.size(2))
    T2_xy= T2_xy.view(T2_xy.size(0),a_xy.size(4),a_xy.size(4),T2_xy.size(2))
    T2_x2y= T2_x2y.view(T2_x2y.size(0),a_x2y.size(4),a_x2y.size(4),T2_x2y.size(2))
    T3_2y= T3_2y.view(a_2y.size(3),a_2y.size(3),T3_2y.size(1),T3_2y.size(2))
    T3_x2y= T3_x2y.view(a_x2y.size(3),a_x2y.size(3),T3_x2y.size(1),T3_x2y.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))
    T4_2y= T4_2y.view(T4_2y.size(0),T4_2y.size(1),a_2y.size(2),a_2y.size(2))
    
    contract_tn= C1,[0,1],T1,[1,2,5,13],T4,[0,18,3,6],a,[I[0],2,3,19,14],a.conj(),[I[1],5,6,20,15],\
        T1_x,[13,9,11,7],C2_x,[7,8],T2_x,[8,10,12,87],a_x,[I[6],9,14,85,10],a_x.conj(),[I[7],11,15,86,12],\
        T4_y,[18,80,16,17],a_y,[I[2],19,16,81,83],a_y.conj(),[I[3],20,17,82,84],\
        T2_xy,[87,55,56,57],a_xy,[I[8],85,83,58,55],a_xy.conj(),[I[9],86,84,59,56],\
        T3_x2y,[42,44,52,40],C3_x2y,[41,40],T2_x2y,[57,43,45,41],a_x2y,[I[10],58,53,42,43],a_x2y.conj(),[I[11],59,54,44,45],\
        T4_2y,[80,47,48,50],C4_2y,[47,46],T3_2y,[49,51,46,52],a_2y,[I[4],81,48,49,53],a_2y.conj(),[I[5],82,50,51,54],I_out
    names= tuple(x.strip() for x in ("C1, T1, T4, a, a*, T1_x, C2_x, T2_x, a_x, a_x*, T4_y, a_y, a_y*, "\
        +"T2_xy, a_xy, a_xy*, T3_x2y, C3_x2y, T2_x2y, a_x2y, a_x2y*, T4_2y, C4_2y, T3_2y, a_2y, a_2y*").split(','))
    
    # Memory limit avoids following contraction (rotated by pi/2 clockwise)
    #
    # |C2X2  |--
    # |      |==
    # |      |==   ||
    # |C2X2  |-- --T-- 
    #
    _tmp_a=(a, a_y, a_2y, a_x, a_xy, a_x2y)
    mem_limit= env.chi**2 * a_x.size(3)**2 * max(a.size(3)**2,a_y.size(3)**2) \
        * prod([_tmp_a[x].size(0)**2 for x in ind_os if x in [0,1,3]])
    if type(unroll)==bool and unroll:
        unroll= [83,84]
    path, path_info= get_contraction_path(*contract_tn,unroll=unroll if unroll else [],\
        names=names,path=None,who=who,\
        memory_limit=mem_limit if unroll else None,\
            optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,optimize=path,backend='torch',\
        unroll=unroll if unroll else [],checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R

# ----- deprecated forms ------
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

