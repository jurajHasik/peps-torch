import itertools
import torch
from ctm.one_site_c4v.rdm_c4v import _log_cuda_mem, _sym_pos_def_rdm
import logging
from linalg.eig_sym import SYMEIG
log = logging.getLogger(__name__)

def _get_open_C2x2_LU_sl(C, T, a, verbosity=0):
    who= "_get_open_C2x2_LU_sl"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    
    #      ------>   
    # C--1 1--T--0->1
    # 0       2
    # C2x2= torch.tensordot(C, T, ([1],[0]))
    C2x2= torch.tensordot(C, T, ([1],[1]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CT")
        

    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTT")

    # 4i) untangle the fused D^2 indices
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size(0),a.size(2),a.size(2),C2x2.size(2),a.size(3),\
        a.size(3))

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # C---------T----0
    # |         |\---2->1
    # |         1    
    # |         2 /1->5
    # T----4 3--a--5->7 
    # | |      /4->6
    # | |     0->4
    # | |
    # |  --5->3
    # 3->2
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTa_init")
    C2x2= torch.tensordot(C2x2, a,([1,4],[2,3]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTa_end")
    
    # 4iii) second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |/5->2  |
    # T----a----------6->4 
    # | |  |\-4 0-\1/1->5
    # |  -----3 2--a--4->7
    # |    |       3->6
    # |    |
    # 2->1 5->3
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTaa_init")
    C2x2= torch.tensordot(C2x2, a.conj(),([1,3,4],[2,3,0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTaa_end")

    # 4iv) fuse pairs of aux indices
    #
    #  C------T----0\
    #  |    2\|\     \
    #  T------a----4\ \ 
    #  | \    | |    ->->1
    #  |  ------a--7/
    #  |      | |\5->3
    # (1     (3 6))->0
    # 
    # permute and reshape 01234567->(136)(047)25->0123
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size(1)*(a.size(4)**2),\
        C2x2.size(0)*(a.size()[4]**2),a.size(1),a.size(1))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    return C2x2

def _get_open_C2x2_LU_dl(C, T, a, verbosity=0):
    who= "_get_open_C2x2_LU_dl"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    dimsa = a.size()
    A = torch.einsum('tmefgh,tnabcd->eafbgchdmn',a,a.conj()).contiguous()\
        .view(dimsa[2]**2, dimsa[3]**2, dimsa[4]**2, dimsa[5]**2, dimsa[1], dimsa[1])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    #      ------>
    # C--1 1--T--0->1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[1]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3
    # 2->1    2\45
    C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2= C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size(1)*A.size(2),T.size(1)*A.size(3),dimsa[1],dimsa[1])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)
    
    return C2x2

def entropy(rdm, ad_decomp_reg, verbosity=0):
    assert len(rdm.size())%2==0, "invalid rank of RDM"
    nsites= len(rdm.size())//2

    rdm= rdm.reshape(torch.prod(torch.as_tensor(rdm.size())[:nsites]),-1)

    rdm_asym= 0.5*(rdm-rdm.conj().t())
    rdm= 0.5*(rdm+rdm.conj().t())
    if verbosity>0: 
        log.info(f"{who} norm(rdm_sym) {rdm.norm()} norm(rdm_asym) {rdm_asym.norm()}")

    reg= torch.as_tensor(ad_decomp_reg, dtype=rdm.dtype, device=rdm.device)
    D, U= SYMEIG.apply(rdm, reg)
    if D.min() < 0:
        log.info(f"{who} max(diag(rdm)) {D.max()} min(diag(rdm)) {D.min()}")

    # assume normalized
    # S = -Tr(rho log rho)
    S= -torch.sum(D*torch.log(D))

    return S

def rdm1x1_sl(state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V_THERMAL
    :type env: ENV_C4V
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor
    
    Evaluates 1-site operator :math:`Tr(\rho_{1x1}O)` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--a^+a--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`a` 
    and its hermitian conjugate :math:`a^\dagger` are left uncontracted
    """
    who= "rdm1x1_sl"
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    #   C--1->0
    #   0
    # A 0
    # | T--2
    # | 1
    CTC = torch.tensordot(C,T,([0],[0]))
    #   C--0
    # A |
    # | T--2->1
    # | 1
    #   0
    #   C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    #      <------
    rdm = torch.tensordot(CTC,T,([2],[0]))
    
    # 4i) untangle the fused D^2 indices
    #
    # C--0
    # |
    # T--1->1,2
    # |    3->4,5
    # C----T--2->3
    a= next(iter(state.sites.values()))
    rdm= rdm.view(rdm.size(0),a.size(3),a.size(3),rdm.size(2),a.size(4),\
        a.size(4))

    #    /
    # --a--
    #  /|s'
    #
    #  s|/
    # --a--
    #  /
    #

    # 4ii) first layer "bra"
    # C--0    2->6
    # |       |/1->5
    # T--1 3--a--5->7
    # |\2->1  4\0->4
    # |       4 5->3
    # |       |/
    # C-------T--3->2
    rdm= torch.tensordot(rdm,a,([1,4],[2,3]))
    
    # 4iii) second layer "ket"
    # C--0         2->6
    # |   6->3     |/1->5
    # |  -|---1 3--a--5->7
    # | / |       /4
    # |/  |/-4 0-/ 3
    # T---a-----------7->4
    # |   |\-5->2  |
    # |   | ------/
    # |   |/
    # C---T-----------2->1
    rdm= torch.tensordot(rdm,a.conj(),([1,3,4],[3,4,0]))

    # 4iv) fuse pairs of aux indices    
    # C--0 (3 6)->2
    # |     | |/5
    # | ----|-a--7\
    # |/    | |    >3
    # T-----a----4/
    # |4<-2/| |
    # |     |/
    # C-----T----1
    rdm= rdm.permute(0,1,3,6,4,7,2,5).contiguous().view(rdm.size(0),rdm.size(1),\
        a.size(2)**2,a.size(5)**2,a.size(1),a.size(1))

    #      ------>
    # C--0 1--T--0
    # |       2
    # |       2
    # T-------a--3->2
    # |       |\45->34(s,s')
    # |       |
    # C-------T--1
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))

    # C--T--0 2---------C
    # |  |              |
    # |  |              | |
    # T--a--2 1---------T |
    # |  |\34->01(s,s') | V
    # |  |              |
    # C--T--1 0---------C
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[2,0,1]))

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm2x1_sl(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V_THERMAL
    :type env: ENV_C4V
    :type force_cpu: bool
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
    who= "rdm2x1_sl"
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))
        
    #----- building C2x2_LU ----------------------------------------------------
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    def ten_size(t):
        return t.element_size() * t.numel()

    #      ------>
    # C--1 1--T--0->1
    # 0       2
    C2x1 = torch.tensordot(C, T, ([1],[1]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1: {ten_size(C2x1)}")

    # see _get_open_C2x2_LU_sl
    C2x2= torch.tensordot(C2x1, T, ([0],[0]))
    C2x2= C2x2.view(C2x2.size(0),a.size(2),a.size(2),C2x2.size(2),a.size(3),\
        a.size(3))
    C2x2= torch.tensordot(C2x2, a,([1,4],[2,3]))
    C2x2= torch.tensordot(C2x2, a.conj(),([1,3,4],[2,3,0]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1,C2x2: {ten_size(C2x1)+ten_size(C2x2)}")

    # 4iv) fuse (some) pairs of aux indices
    #
    # C------T----0->2
    # | 4<-2\|\    
    # T------a----4\
    # | \    | |    ->->3
    # |  ------a--7/
    # |      | |\5
    # 1->0  (3 6)->1
    # 
    # permute and reshape 01234567->1(36)0(47)25->012345
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size(1),a.size(4)**2,\
        C2x2.size(0),a.size(5)**2,a.size(1),a.size(1))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1,C2x2: {ten_size(C2x1)+ten_size(C2x2)}")
                  
    # 0       2          0  2
    # C--1 0--T--1->1 -> C--T--1
    #      <------
    C2x1= torch.tensordot(C, T, ([1],[0]))

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    #   --->
    # A C2x2--2->1
    # | |__|--3->2
    #   | | \45->34
    #   0 1
    #   0 2
    #   C2x1--1->0
    #   <---
    left_half = torch.tensordot(C2x1, C2x2, ([0,2],[0,1]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} left_half: {ten_size(left_half)}")

    # construct reduced density matrix by contracting left and right halfs
    # --->         --->
    # C2x2--1 0----C2x1
    # |__|--2 2--\ |
    # |\34->01    \|__/34->23
    # |            |  |
    # C2x1--0 1----C2x2
    # <---         <---
    rdm = torch.tensordot(left_half,left_half,([0,1,2],[1,0,2]))
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} rdm: {ten_size(rdm)}")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= rdm.permute(0,2,1,3).contiguous()

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.to(env.device)

    return rdm

def rdm2x2(state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type verbosity: int
    :return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix
    
    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |
        T--a^+a--a^+a--T   C2x2--C2x2
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1
        s2 s3

    """
    #----- building C2x2_LU ----------------------------------------------------
    who= "rdm2x2"
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a = next(iter(state.sites.values()))
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    #   --->
    # A C2x2--1
    # | |\23
    #   0
    C2x2= _get_open_C2x2_LU_dl(C,T,a,verbosity=verbosity)
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # --->       --->                 --->      --->
    # C2x2--1 0--C2x2                 C2x2------C2x2
    # |\23->12   |\23->45   & permute |\12->23  |\45
    # 0          1->3                 0         3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    upper_half = torch.tensordot(C2x2, C2x2, ([1],[0]))
    upper_half = upper_half.permute(0,3,1,2,4,5)
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # construct reduced density matrix by contracting lower and upper halfs
    # --->      --->
    # C2x2------C2x2   
    # |\23->01  |\45->23       
    # 0         1             
    # 1         0             
    # |/45->67  |/23->45     
    # C2x2------C2x2
    # <---      <---
    rdm = torch.tensordot(upper_half,upper_half,([0,1],[1,0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    #
    # --->      --->
    # C2x2------C2x2
    # |\01      | \23
    # |/67->45  | /45->67
    # C2x2------C2x2
    # <---      <---
    #
    # 01234567->02641375
    rdm = rdm.permute(0,2,6,4,1,3,7,5).contiguous()

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm
