import itertools
import torch
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import ENV_C4V
from ctm.one_site_c4v.ctm_components_c4v import c2x2_dl
from ctm.one_site_c4v.rdm_c4v import _log_cuda_mem, _sym_pos_def_rdm
import logging
log = logging.getLogger(__name__)

def rdm2x1_tiled(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of a horizontal 
    2x1 subsystem using following strategy:
    
        1. compute upper left 2x2 corner and lower left 2x1 corner
        2. construct left half of the network
        3. contract left and right half (identical to the left) to obtain final 
           reduced density matrix

    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |  
        T--a^+a--a^+a--T   C2x1--C2x1
        |  |     |     |
        C--T-----T-----C 

    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1

    The resulting reduced density matrix is identical to the one of vertical 1x2 subsystem
    due to the C4v symmetry. 
    """
    who= "rdm2x1_tiled"
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))

    loc_device=a.device
    is_cpu= loc_device==torch.device('cpu')
    log_gpu_mem= (not is_cpu and verbosity>0)

    def ten_size(t):
        return t.element_size() * t.numel()

    #----- building C2x2_LU ----------------------------------------------------
    dimsa = a.size()

    #       ---->
    # C--1 1--T--0->1
    # 0       2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_init")
    C2x1 = torch.tensordot(C, T, ([1],[1]))
    # C2x1= torch.einsum('i,ijk->ijk',torch.diag(C), T)
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_end")

    #        ---->
    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_init")
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_end")

    # unfuse auxiliary indices of T's
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T2--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0], dimsa[1], dimsa[1], C2x2.size()[2], \
        dimsa[2], dimsa[2])

    # prepare tensor to store results chi^2 x D^2 x dimsa[0]^2
    #  _______
    # |       |--1 (chi)
    # |       |--2 (D^2)
    # left_half--3 (dimsa[0]^2)
    # |_______|--0 (chi)
    #
    left_half= torch.zeros(C.size(0),C.size(0),(dimsa[4]**2),dimsa[0],dimsa[0],\
        device=C2x2.device, dtype=C2x2.dtype)

    # 0       2     
    # C--1 0--T--1
    #       <----
    C2x1= torch.tensordot(C, T, ([1],[0]))
    
    # enter tiled loop over physical index of A
    for p0,p1 in itertools.product(range(dimsa[0]),range(dimsa[0])):

        # first layer "ket"
        # 
        # C---------T----0
        # |         |\---2->1
        # |         1    
        # |         0
        # T----4 1--a(p1)--3->5 
        # | |       2->4
        # |  --5->3
        # 3->2
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTa_{p0}{p1}_init")
        tmp= torch.tensordot(C2x2, a[p0,:,:,:,:], ([1,4],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTa_{p0}{p1}_end")

        # second layer "bra"
        # 
        # C----T----------0
        # |    |\-----\
        # |    |       1
        # |    |       |
        # T----a----------6->3 
        # | |  |       0
        # |  -----3 1--a--3->5
        # |    |       2->4
        # |    |
        # 2->1 5->2
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaa_{p0}{p1}_init")
        tmp= torch.tensordot(tmp, a[p1,:,:,:,:].conj(), ([1,3],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaa_{p0}{p1}_end")

        # permute 012345->124035
        # reshape 1(24)0(35)->0123
        # -->         -->
        # tmp--0      tmp--2
        # |_|--3,5 => |_|--3
        # | |         | |
        # 1 2,4       0 1
        tmp= tmp.permute(1,2,4,0,3,5).contiguous().view(\
            T.size(1),dimsa[3]**2,T.size(1),dimsa[4]**2)

        #----- build left part C2x2_LU--C2x1_LD ------------------------------------
        # -->
        # tmp--2->1
        # |_|--3->2
        # | |\p1,p2
        # 0 1
        # 0 2
        # C2x1--1->0
        # <---
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaaTC_{p0}{p1}_init")
        left_half[:,:,:,p0,p1]= torch.tensordot(C2x1, tmp, ([0,2],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaaTC_{p0}{p1}_end")

    # construct reduced density matrix by contracting left and right halfs
    # --->         --->
    # C2x2--1 0----C2x1
    # |__|--2 2----|
    # |\3,4->0,1   |___/3,4->2,3
    # |            |  |
    # C2x1--0 1----C2x2
    # <---         <---
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm_init")
    rdm = torch.tensordot(left_half,left_half,([0,1,2],[1,0,2]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm_end")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= rdm.permute(0,2,1,3).contiguous()

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.to(env.device)

    return rdm

def _get_open_c2x2_LU_sl_elem(C, T, a, idxs, verbosity=0):
    who="_get_open_c2x2_LU_sl_elem"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    log_gpu_mem= (not is_cpu and verbosity>0)

    assert len(idxs.size())==1 and idxs.size()[0]==2,"Invalid physical indices"
    dimsa= a.size()

    #       ---->
    # C--1 1--T--0->1
    # 0       2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_init")
    C2x2 = torch.tensordot(C, T, ([1],[1]))
    # C2x2= torch.einsum('i,ijk->ijk',torch.diag(C), T)
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_end")

    #        ---->
    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_init")
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_end")

    # unfuse auxiliary indices of T's
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T2--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size(0), dimsa[1], dimsa[1], C2x2.size(2), \
        dimsa[2], dimsa[2])

    # first layer "bra" (in principle conjugate)
    # 
    # C---------T----0
    # |         |\---2->1
    # |         1    
    # |         0
    # T----4 1--a(p1)--3->5 
    # | |       2->4
    # |  --5->3
    # 3->2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTa_{idxs[0]}_init")
    C2x2= torch.tensordot(C2x2, a[idxs[0],:,:,:,:], ([1,4],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTa_{idxs[0]}_end")

    # second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |       |
    # T----a----------6->3 
    # | |  |       0
    # |  -----3 1--a--3->5
    # |    |       2->4
    # |    |
    # 2->1 5->2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaa_{idxs[1]}_init")
    C2x2= torch.tensordot(C2x2, a[idxs[1],:,:,:,:].conj(), ([1,3],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,f"CTTaa_{idxs[1]}_end")

    # permute 012345->124035
    # reshape (124)(035)->01
    # tmp--0      tmp--1
    # |_|--3,5 => |
    # | |         0
    # 1 2,4
    C2x2= C2x2.permute(1,2,4,0,3,5).contiguous().view(\
        T.size()[1]*(dimsa[3]**2),T.size()[1]*(dimsa[4]**2))

    return C2x2

def rdm2x2_NN_tiled(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    who= "rdm2x2_NN_tiled"
    if force_cpu:
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))
    
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    log_gpu_mem= (not is_cpu and verbosity>0)

    def ten_size(t):
        return t.element_size() * t.numel()

    # build basic building blocks of RDM elements
    #----- building C2x2_LU ----------------------------------------------------
    dimsa = a.size()
    A = torch.einsum('sefgh,sabcd->eafbgchd',a,a.conj()).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    # ---->
    # C2x2c--1
    # |
    # 0
    C2x2c= c2x2_dl(A,C,T,verbosity=verbosity)

    # pre-allocate the result
    rdm= torch.zeros(tuple([dimsa[0]]*4), device=a.device, dtype=a.dtype)

    # loop over all combinations of physical indices
    for p00,p01,p10,p11 in itertools.product(*tuple([range(dimsa[0])]*4)):
        uuid=f"{p00}{p01}{p10}{p11}"
        idxs0= torch.tensor([p00,p01])
        idxs1= torch.tensor([p10,p11])

        # --->
        # C2x2--1
        # |\p0
        # 0
        C2x2= _get_open_c2x2_LU_sl_elem(C, T, a, idxs0, verbosity=verbosity)

        #----- build upper part C2x2 -- C2x2c ----------------------------------
        # ---->       --->    ---->  --->
        # C2x2c--1 0--C2x2    C2x2c--C2x2
        # |           | \     |      | \
        # 0           1  p0   0      1 p0
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"C2x2c-C2x2p0_init")
        tmp= torch.mm(C2x2c,C2x2)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"C2x2c-C2x2p0_end")

        #----- complete lower part C2x2 -- C2x2c -------------------------------
        #     -->
        #  ---tmp-- --p0
        # |        |
        # 0        1
        # 1        
        # |        
        # C2x2c--0
        # <----
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"tmp-C2x2p1_init")
        C2x2= torch.mm(C2x2c,tmp)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"tmp-C2x2p1_end")
        
        # -->
        # tmp--1
        # |\p1
        # 0
        tmp= _get_open_c2x2_LU_sl_elem(C, T, a, idxs1, verbosity=verbosity)

        #     --->
        #  ---C2x2---- --p0
        # |           |
        # |           1
        # |           0
        # |           |
        #  -----0 1---tmp--p1
        #             <--
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm-p0p1_init")
        rdm[p00,p10,p01,p11]= torch.einsum('ij,ji',C2x2,tmp)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm-p0p1_end")

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.to(env.device)

    return rdm

def rdm2x2_NNN_tiled(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    who= "rdm2x2_NNN_tiled"
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))

    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    log_gpu_mem= (not is_cpu and verbosity>0)

    def ten_size(t):
        return t.element_size() * t.numel()

    # build basic building blocks of RDM elements
    #----- building C2x2_LU ----------------------------------------------------
    dimsa = a.size()
    A = torch.einsum('sefgh,sabcd->eafbgchd',a,a.conj()).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    # ---->
    # C2x2c--1
    # |
    # 0
    C2x2c= c2x2_dl(A,C,T,verbosity=verbosity)

    # pre-allocate the result
    rdm= torch.zeros(tuple([dimsa[0]]*4), device=a.device, dtype=a.dtype)

    # loop over all combinations of physical indices
    for p00,p01,p10,p11 in itertools.product(*tuple([range(dimsa[0])]*4)):
        uuid=f"{p00}{p01}{p10}{p11}"
        idxs0= torch.tensor([p00,p01])
        idxs1= torch.tensor([p10,p11])

        # --->
        # C2x2--1
        # |\p0
        # 0
        C2x2= _get_open_c2x2_LU_sl_elem(C, T, a, idxs0, verbosity=verbosity)

        #----- build upper part C2x2 -- C2x2c ----------------------------------
        # ---->       --->    ---->  --->
        # C2x2c--1 0--C2x2    C2x2c--C2x2
        # |           | \     |      | \
        # 0           1  p0   0      1 p0
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"C2x2c-C2x2p0_init")
        tmp= torch.mm(C2x2c,C2x2)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"C2x2c-C2x2p0_end")

        #----- complete lower part C2x2 -- C2x2c -------------------------------
        # --->
        # C2x2--1
        # |\p1
        # 0
        C2x2= _get_open_c2x2_LU_sl_elem(C, T, a, idxs1, verbosity=verbosity)
        #     -->
        #  ---tmp-----
        # |          /|
        # 0    p00p01 1
        # 1 p10p11   
        # |/           
        # C2x2--0
        # <---
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"tmp-C2x2p1_init")
        tmp= torch.mm(C2x2,tmp)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"tmp-C2x2p1_end")
        #     -->
        #  ---tmp-----
        # |          /|
        # |    p00p01 1
        # | p10p11    0
        # |/          | 
        #  -----0 1---C2x2c
        #             <----
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm-p0p1_init")
        tmp= torch.einsum('ij,ji',tmp,C2x2c)
        if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm-p0p1_end")

        rdm[p00,p10,p01,p11]= tmp

    # normalize and symmetrize and move to original device
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.to(env.device)

    return rdm