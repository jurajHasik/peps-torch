import torch
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import ENV_C4V
import logging
log = logging.getLogger(__name__)

# ----- components -------------------------------------------------------------
def _log_cuda_mem(device, who="unknown",  uuid=""):
    log.info(f"{who} {uuid} GPU-MEM MAX_ALLOC {torch.cuda.max_memory_allocated(device)}"\
            + f" CURRENT_ALLOC {torch.cuda.memory_allocated(device)}")

def _sym_pos_def(rdm, verbosity=0, who="unknown"):
    rdm_asym= 0.5*(rdm-rdm.t())
    rdm= 0.5*(rdm+rdm.t())
    if verbosity>0: 
        log.info(f"{who} norm(rdm_sym) {rdm.norm()} norm(rdm_asym) {rdm_asym.norm()}")
    with torch.no_grad():
        D, U= torch.symeig(rdm, eigenvectors=True)
        if D.min() < 0:
            log.info(f"{who} max(diag(rdm)) {D.max()} min(diag(rdm)) {D.min()}")
            D= torch.clamp(D, min=0)
            rdm_posdef= U@torch.diag(D)@U.t()
            rdm.copy_(rdm_posdef)
    return rdm

def _get_open_C2x2_LU_sl(C, T, a, verbosity=0):
    who= "_get_open_C2x2_LU_sl"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CT")
        

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTT")

    # 4i) untangle the fused D^2 indices
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # C---------T----0
    # |         |\---2->1
    # |         1    
    # |         1 /0->4
    # T----4 2--a--4->6 
    # | |       3->5
    # |  --5->3
    # 3->2
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTa_init")
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTa_end")
    
    # 4iii) second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |/4->2  |
    # T----a----------6->4 
    # | |  |       1/0->5
    # |  -----3 2--a--4->7
    # |    |       3->6
    # |    |
    # 2->1 5->3
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who,"CTTaa_init")
    C2x2= torch.tensordot(C2x2, a,([1,3],[1,2]))
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
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size()[1]*(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2),a.size()[0],a.size()[0])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    return C2x2

def _get_open_C2x2_LU_dl(C, T, a, verbosity=0):
    who= "_get_open_C2x2_LU_dl"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    dimsa = a.size()
    A = torch.einsum('mefgh,nabcd->eafbgchdmn',a,a).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2, dimsa[0], dimsa[0])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
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
        T.size()[1]*A.size()[2],T.size()[1]*A.size()[3],dimsa[0],dimsa[0])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)
    
    return C2x2

def _get_aux_C2x2_LU(C, T, verbosity=0):
    who= "_get_aux_C2x2_LU"
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C------T--1->2
    # 0      2->3
    # 0
    # T--2->1
    # 1->0
    C2x2 = torch.tensordot(T, C2x2, ([0],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # |C2x2____|--2    |C2x2____|--1  
    # | |     3        | |     3 
    # |_|--1        => |_|--2
    #  0                0
    #
    C2x2= C2x2.permute(0,2,1,3).contiguous()
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    return C2x2

# ----- density matrices in physical space -------------------------------------
def rdm1x1(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor
    
    Computes 1-site reduced density matrix :math:`\rho_{1x1}` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--a^+a--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`a` 
    and its hermitian conjugate :math:`a^\dagger` are left uncontracted
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    CTC = torch.tensordot(C,T,([0],[0]))

    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))

    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(CTC,T,([2],[0]))

    # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
    #        Possibly reshape indices 1,2 of rdm, which are to be contracted with 
    #        on-site tensor and contract bra,ket in two steps instead of creating 
    #        double layer tensor
    #    /
    # --A--
    #  /|s
    #  
    # s'|/
    # --A--
    #  /
    #
    A= next(iter(state.sites.values()))
    dimsA = A.size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',A,A).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])
    # C--0
    # |
    # |       0->2
    # T--1 1--a--3
    # |       2\45(s,s')
    # |       3
    # C-------T--2->1
    rdm = torch.tensordot(rdm,a,([1,3],[1,2]))

    # C--0 0--T--1->0
    # |       2
    # |       2
    # T-------a--3->2
    # |       |\45->34(s,s')
    # |       |
    # C-------T--1
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))
    
    # C--T---0 0--------C
    # |  |              |
    # |  |              |
    # T--a---2 1--------T
    # |  |\34->01(s,s') |
    # |  |              |
    # C--T---1 2--------C
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[0,2,1]))

    # symmetrize
    rdm= 0.5*(rdm+rdm.t())
    # TODO make pos-def ?
    eps=0.0
    with torch.no_grad():
        D, U= torch.eig(rdm)
        # check only real part
        if D[:,0].min() < 0:
            eps= D[:,0].min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize
    if verbosity>0: log.info(f"Tr(rdm1x1): {torch.trace(rdm)}\n")
    rdm = rdm / torch.trace(rdm)

    return rdm

def rdm1x1_sl(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    CTC = torch.tensordot(C,T,([0],[0]))
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(CTC,T,([2],[0]))
    
    # 4i) untangle the fused D^2 indices
    #
    # C--0
    # |
    # T--1->1,2
    # |    3->4,5
    # C----T--2->3
    a= next(iter(state.sites.values()))
    rdm= rdm.view(rdm.size()[0],a.size()[2],a.size()[2],rdm.size()[2],a.size()[3],\
        a.size()[3])

    #    /
    # --a--
    #  /|s'
    #
    #  s|/
    # --a--
    #  /
    #

    # 4ii) first layer "bra" (in principle conjugate)
    # C--0    ->5
    # |       1/0->4
    # T--1 2--a--4->6
    # |\2->1  3
    # |       4 5->3
    # |       |/
    # C-------T--3->2
    rdm= torch.tensordot(rdm,a,([1,4],[2,3]))
    
    # 4iii) second layer "ket"
    # C--0
    # |   5->3     1->6
    # |  -|---1 2--a--4->7
    # | / |        3\0->5
    # |/  |/4->2   3
    # T---a-----------6->4
    # |   | ------/
    # |   |/
    # C---T-----------2->1
    rdm= torch.tensordot(rdm,a,([1,3],[2,3]))

    # 4iv) fuse pairs of aux indices    
    # C--0 (3 6)->2
    # |     | |/5
    # | ----|-a--7\
    # |/    | |    >3
    # T-----a----4/
    # |4<-2/| |
    # |     |/
    # C-----T----1
    rdm= rdm.permute(0,1,3,6,4,7,2,5).contiguous().view(rdm.size()[0],rdm.size()[1],\
        a.size()[1]**2,a.size()[4]**2,a.size()[0],a.size()[0])

    # C--0 0--T--1->0
    # |       2
    # |       2
    # T-------a--3->2
    # |       |\45->34(s,s')
    # |       |
    # C-------T--1
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))

    # C--T--0 0---------C
    # |  |              |
    # |  |              |
    # T--a--2 1---------T
    # |  |\34->01(s,s') |
    # |  |              |
    # C--T--1 2---------C
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[0,2,1]))

    # symmetrize
    rdm= 0.5*(rdm+rdm.t())
    # TODO make pos-def ?
    eps=0.0
    with torch.no_grad():
        D, U= torch.eig(rdm)
        # check only real part
        if D[:,0].min() < 0:
            eps= D[:,0].min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize
    if verbosity>0: log.info(f"Tr(rdm1x1): {torch.trace(rdm)}\n")
    rdm = rdm / torch.trace(rdm)

    return rdm

def rdm2x1(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
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
    who= "rdm2x1"
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
    A = torch.einsum('mefgh,nabcd->eafbgchdmn',a,a).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2, dimsa[0], dimsa[0])

    # C--1 0--T--1
    # 0       2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_init")
    C2x1 = torch.tensordot(C, T, ([1],[0]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CT_end")

    # C------T--1->0
    # 0      2->1
    # 0
    # T2--2->3
    # 1->2
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_init")
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTT_end")

    # C-------T--0
    # |       1
    # |       0
    # T2--3 1 A--3
    # 2->1    2\45
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTTA_init")
    C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTTA_end")

    # permute 012345->120345
    # reshape 12(03)45->01234
    # C2x2--2
    # | |\34
    # 0 1
    C2x2 = C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1],A.size()[3],T.size()[1]*A.size()[2],dimsa[0],dimsa[0])

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2--2->1
    # | |\34->23
    # 0 1
    # 0 2
    # C2x1--1->0
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTTATC_init")
    left_half = torch.tensordot(C2x1, C2x2, ([0,2],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"CTTATC_end")

    # construct reduced density matrix by contracting left and right halfs
    # C2x2--1 1----C2x2
    # |\23->01     |\23
    # |            |
    # C2x1--0 0----C2x1
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm_init")
    rdm = torch.tensordot(left_half,left_half,([0,1],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device,who,"rdm_end")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    
    if sym_pos_def: 
        rdm= _sym_pos_def(rdm, verbosity=verbosity, who="rdm2x1")

    # normalize and reshape and move to original device
    if verbosity>0: log.info(f"Tr(rdm2x1): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x1_sl(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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

    # C--1 0--T--1
    # 0       2
    C2x1 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1: {ten_size(C2x1)}")

    # see _get_open_C2x2_LU_sl
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))
    C2x2= torch.tensordot(C2x2, a,([1,3],[1,2]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1,C2x2: {ten_size(C2x1)+ten_size(C2x2)}")

    # 4iv) fuse (some) pairs of aux indices
    #
    # C------T----0\
    # | 3<-2\|\     \
    # T------a----4\ \ 
    # | \    | |    ->->2
    # |  ------a--7/
    # |      | |\5->4
    # 1->0  (3 6)->1
    # 
    # permute and reshape 01234567->1(36)(047)25->01234
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size()[1],(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2),a.size()[0],a.size()[0])
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x1,C2x2: {ten_size(C2x1)+ten_size(C2x2)}")

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2--2->1
    # | |\34->23
    # 0 1
    # 0 2
    # C2x1--1->0
    left_half = torch.tensordot(C2x1, C2x2, ([0,2],[0,1]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} left_half: {ten_size(left_half)}")

    # construct reduced density matrix by contracting left and right halfs
    # C2x2--1 1----C2x2
    # |\23->01     |\23
    # |            |
    # C2x1--0 0----C2x1
    rdm = torch.tensordot(left_half,left_half,([0,1],[0,1]))
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} rdm: {ten_size(rdm)}")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    if sym_pos_def: 
        rdm= _sym_pos_def(rdm, verbosity=verbosity, who="rdm2x1")

    # normalize and reshape and move to original device
    if verbosity>0: log.info(f"Tr(rdm2x1): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x2_NN_lowmem(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of 2 sites 
    that are nearest neighbours using strategy:

        1. compute upper left corner using double-layer on-site tensor
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2--C2x2c
        |  |     |     |   |     |
        T--A^+A--A^+A--T   C2x2--C2x2c
        |  |     |     |
        T--A^+A--A^+A--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        s1 c

    """
    return _rdm2x2_NN_lowmem(state,env,_get_open_C2x2_LU_dl,sym_pos_def=sym_pos_def,\
        force_cpu=force_cpu,verbosity=verbosity)

def rdm2x2_NN_lowmem_sl(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of 2 sites 
    that are nearest neighbours using strategy:

        1. compute upper left corner using layer-by-layer contraction of on-site tensor 
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2--C2x2c
        |  |     |     |   |     |
        T--a^+a--a^+a--T   C2x2--C2x2c
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        s1 c

    """
    return _rdm2x2_NN_lowmem(state,env,_get_open_C2x2_LU_sl,sym_pos_def=sym_pos_def,\
        force_cpu=force_cpu,verbosity=verbosity)

def _rdm2x2_NN_lowmem(state, env, f_c2x2, sym_pos_def=False, force_cpu=False, verbosity=0):
    who= "_rdm2x2_NN_lowmem"
    if force_cpu:
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

    # C2x2--1
    # |\23
    # 0
    C2x2= f_c2x2(C, T, a, verbosity=verbosity)

    # C2x2c--1
    # |
    # 0
    C2x2c= torch.einsum('abii->ab',C2x2)
    if not is_cpu and verbosity>1: 
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")

    #----- build upper part C2x2 -- C2x2c -----------------------------------
    # C2x2c--1 0--C2x2    C2x2c--C2x2
    # |           | \     |      | \
    # 0           1  23   0      1 (23)->2
    C2x2= C2x2.view(T.size()[1]*(a.size()[2]**2),T.size()[1]*(a.size()[3]**2),a.size()[0]**2)
    C2x2= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    # C2x2c= torch.matmul(C2x2c,C2x2)
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")

    # C2x2c----C2x2--2
    # |        |
    # 0        1
    # 0        1 
    # |        |
    # C2x2c----C2x2--2
    rdm= torch.einsum('abi,abj->ij',C2x2,C2x2)
    #C2x2= C2x2c.permute(2,1,0).contiguous().view(dimsA[0]*dimsA[0],(T.size()[1]*a.size()[3])**2)
    #C2x2c= C2x2c.view((T.size()[1]*a.size()[3])**2,dimsA[0]*dimsA[0])
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")

    rdm= rdm.view(tuple([a.size()[0] for i in range(4)]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    
    if sym_pos_def: 
        rdm= _sym_pos_def(rdm, verbosity=verbosity, who=who)

    # normalize and reshape and move to original device
    if verbosity>0: log.info(f"{who} Tr(rdm): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x2_NNN_lowmem(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}^{diag}` of 2 sites 
    that are next-nearest neighbours (along diagonal) using strategy:

        1. compute upper left corner using double-layer on-site tensor
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2---C2x2c
        |  |     |     |   |      |
        T--a^+a--a^+a--T   C2x2c--C2x2
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        c  s1

    """
    return _rdm2x2_NNN_lowmem(state,env,_get_open_C2x2_LU_dl,sym_pos_def=sym_pos_def,\
        force_cpu=force_cpu,verbosity=verbosity)

def rdm2x2_NNN_lowmem_sl(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x2}^{diag}` of 2 sites 
    that are next-nearest neighbours (along diagonal) using strategy:

        1. compute upper left corner using layer-by-layer contraction of on-site tensor 
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2---C2x2c
        |  |     |     |   |      |
        T--a^+a--a^+a--T   C2x2c--C2x2
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        c  s1

    """
    return _rdm2x2_NNN_lowmem(state,env,_get_open_C2x2_LU_sl,sym_pos_def=sym_pos_def, 
        force_cpu=force_cpu,verbosity=verbosity)

def _rdm2x2_NNN_lowmem(state, env, f_c2x2, sym_pos_def=False, force_cpu=False, verbosity=0):
    who= "_rdm2x2_NNN_lowmem"
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

    # C2x2--1
    # |\23
    # 0
    C2x2= f_c2x2(C, T, a, verbosity=verbosity)

    # C2x2c--1
    # |
    # 0
    C2x2c= torch.einsum('abii->ab',C2x2)
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")

    #----- build upper part C2x2 -- C2x2c -----------------------------------
    # C2x2c--1 0--C2x2    C2x2c--C2x2
    # |           | \     |      | \
    # 0           1  23   0      1 (23)->2
    C2x2= C2x2.view(T.size()[1]*(a.size()[2]**2),T.size()[1]*(a.size()[3]**2),a.size()[0]**2)
    # rdm= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    C2x2= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")

    #  ---C2x2----
    # |          /|
    # 0   2=s0s0' 1 
    # 1 2=s1s1'   0
    # |/          | 
    #  ---C2x2----
    rdm= torch.einsum('abi,baj->ij',C2x2,C2x2)
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)
        log.info(f"{who} C2x2,C2x2c: {ten_size(C2x2)+ten_size(C2x2c)}")


    # rdm= torch.einsum('abi,abj->ij',C2x2,rdm)
    rdm= rdm.view(tuple([a.size()[0] for i in range(4)]))
    if not is_cpu and verbosity>1:
        _log_cuda_mem(loc_device,who)

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)

    if sym_pos_def: 
        rdm= _sym_pos_def(rdm, verbosity=verbosity, who=who)

    # normalize and reshape and move to original device
    if verbosity>0: log.info(f"{who} Tr(rdm): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

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

    dimsa = a.size()
    A = torch.einsum('mefgh,nabcd->eafbgchdmn',a,a).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2, dimsa[0], dimsa[0])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
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
    C2x2 = C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1]*A.size()[2],T.size()[1]*A.size()[3],dimsa[0],dimsa[0])
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2--1 0--C2x2                 C2x2------C2x2
    # |\23->12   |\23->45   & permute |\12->23  |\45
    # 0          1->3                 0         3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    upper_half = torch.tensordot(C2x2, C2x2, ([1],[0]))
    upper_half = upper_half.permute(0,3,1,2,4,5)
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2------C2x2
    # |\23->01  |\45->23
    # 0         1
    # 0         1
    # |/23->45  |/45->67
    # C2x2------C2x2_RD
    rdm = torch.tensordot(upper_half,upper_half,([0,1],[0,1]))
    if not is_cpu and verbosity>1: _log_cuda_mem(loc_device,who)

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # and normalize
    rdm = rdm.permute(0,2,4,6,1,3,5,7).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**4,dimsRDM[0]**4)

    if sym_pos_def: 
        rdm= _sym_pos_def(rdm, verbosity=verbosity, who=who)
        
    # normalize and reshape
    if verbosity>0: log.info(f"{who} Tr(rdm): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM)

    return rdm

# ----- density matrices in auxiliary space ------------------------------------
def aux_rdm1x1(state, env, verbosity=0):
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a= next(iter(state.sites.values()))
    dimsa = a.size()
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    CTC = torch.tensordot(C,T,([0],[0]))
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(CTC,T,([2],[0]))
    # C--0 0--T--1->3
    # |       2->4
    # |       
    # T--1->0
    # |      
    # |       3->2
    # C-------T--2->1
    rdm = torch.tensordot(rdm,T,([0],[0]))
    
    # C----T--3 0--C
    # |    4->2    |    C----T----C
    # |            |    |    0    |
    # T--0  3<-1---T => T--1   3--T
    # |            |    |    2    |
    # |    2->1    |    C----T----C
    # C----T--1 2--C
    rdm = torch.tensordot(rdm,CTC,([1,3],[2,0]))

    rdm= rdm.permute(2,0,1,3).contiguous()
    rdm= rdm.view([dimsa[1]]*8).permute(0,2,4,6, 1,3,5,7).contiguous()

    return rdm

def aux_rdm2x2_NN(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site auxilliary reduced density matrix
    :rtype: torch.tensor

    Computes 2-site auxiliary reduced density matrix of nearest neighbours 
    within 2x2 subsystem without the on-site tensors (leaving corresponding 
    auxiliary indices open) using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           auxilliary density matrix

    :: 

          C----T----T----C    = aC2x2--aC2x2
          |    0    5    |      |       |
          T--1        4--T      C2x2----C2x2
          |    2    3    |
          C----T----T----C

    """
    #----- building pC2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a = next(iter(state.sites.values()))
    dimsa = a.size()
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    aC2x2= _get_aux_C2x2_LU(C, T, verbosity=verbosity)

    #----- build upper part aC2x2_LU--aC2x2_RU -----------------------------------
    # aC2x2----1 1----aC2x2
    # |_  \          /   _|
    # | |  3->2  5<-3   | |
    # 0 2               2 0
    #   V               V V
    #   1               4 3
    # aC2x2 = torch.tensordot(aC2x2, aC2x2, ([1],[1]))
    # =====================
    # aC2x2----1 0----aC2x2
    # |_  \          /   _|
    # | |  3->2  4<-2   | |
    # 0 2               3 1
    #   V               V V
    #   1               5 3
    aC2x2 = torch.tensordot(aC2x2, aC2x2, ([1],[0]))

    # C2x2--1 => C2x2--1 => |C2x2|--2
    # | \        |          | |
    # 0  23      0          0 1
    C2x2= _get_open_C2x2_LU_sl(C, T, a, verbosity=verbosity)
    C2x2= torch.einsum('abii->ab',C2x2)
    C2x2= C2x2.view(C.size()[0],dimsa[3]**2,C.size()[0]*(dimsa[4]**2))

    # C2x2----2 2----C2x2
    # |  |           |  |
    # 0  1           1  0
    #                V  V
    #                3  2
    C2x2= torch.tensordot(C2x2,C2x2,([2],[2]))

    # construct reduced density matrix by contracting lower and upper halfs
    #  __________________         __________________
    # |_______aC2x2______|       |______aC2x2_______|    C----T----T----C
    # | | \          / | |       | | \          / | |    |    1    3    |
    # 0 1  2        5  4 3       | 0  1        3  2 |    T--0        2--T
    # 0                  3  =>   |                  | => |              |
    # | 1  2        5  4 |       | 4  5        7  6 |    T--4        6--T
    # |_|_/__________\_|_|       |_|_/__________\_|_|    |    5    7    |
    # |_______C2x2_______|       |_______C2x2_______|    C----T----T----C
    # ===================================================================
    #  __________________         __________________
    # |_______aC2x2______|       |_______aC2x2______|    C----T----T----C
    # | | \          / | |       | | \          / | |    |    1    2    |
    # 0 1  2        4  5 3       | 0  1        2  3 |    T--0        3--T
    # 0                  2  =>   |                  | => |    4    5    |
    # | 1              3 |       | 4              5 |    C----T----T----C
    # |_|______________|_|       |_|______________|_|
    # |_______C2x2_______|       |________C2x2______|
    aC2x2 = torch.tensordot(aC2x2,C2x2,([0,3],[0,2]))

    # permute such, that aux-index increases from "up" in the anti-clockwise direction
    # aC2x2 = aC2x2.permute(1,0,4,5,7,6,2,3).contiguous()
    # ===================================================
    aC2x2 = aC2x2.permute(1,0,4,5,3,2).contiguous()
    # reshape and form bra and ket index
    aC2x2 = aC2x2.view([dimsa[1]]*12).permute(0,2,4,6,8,10, 1,3,5,7,9,11).contiguous()

    return aC2x2

def aux_rdm2x2(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 4-site auxilliary reduced density matrix
    :rtype: torch.tensor

    Computes 4-site auxiliary reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem 
    without the on-site tensors (leaving corresponding auxiliary indices open)
    using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           auxilliary density matrix
           
    TODO try single torch.einsum ?

    :: 

          C----T----T----C    = C2x2--C2x2
          |    0    7    |      |     |
          T--1        6--T      C2x2--C2x2
          |              |
          T--2        5--T
          |    3    4    |
          C----T----T----C

    """
    #----- building pC2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a = next(iter(state.sites.values()))
    dimsa = a.size()
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    aC2x2= _get_aux_C2x2_LU(C, T, verbosity=verbosity)

    #----- build upper part aC2x2_LU--aC2x2_RU -----------------------------------
    # aC2x2----1 1----aC2x2
    # |_  \          /   _|
    # | |  3->2  5<-3   | |
    # 0 2               2 0
    #   V               V V
    #   1               4 3
    # aC2x2 = torch.tensordot(aC2x2, aC2x2, ([1],[1]))
    # =====================
    # aC2x2----1 0----aC2x2
    # |_  \          /   _|
    # | |  3->2  4<-2   | |
    # 0 2               3 1
    #   V               V V
    #   1               5 3
    aC2x2 = torch.tensordot(aC2x2, aC2x2, ([1],[0]))

    # construct reduced density matrix by contracting lower and upper halfs
    #  __________________         __________________
    # |_______aC2x2______|       |__________________|    C----T----T----C
    # | | \          / | |       | | \          / | |    |    1    3    |
    # 0 1  2        5  4 3       | 0  1        3  2 |    T--0        2--T
    # 0                  3  =>   |                  | => |              |
    # | 1  2        5  4 |       | 4  5        7  6 |    T--4        6--T
    # |_|_/__________\_|_|       |_|_/__________\_|_|    |    5    7    |
    # |_______aC2x2______|       |_______aC2x2______|    C----T----T----C
    # ===================================================================
    #  __________________         __________________
    # |_______aC2x2______|       |__________________|    C----T----T----C
    # | | \          / | |       | | \          / | |    |    1    2    |
    # 0 1  2        4  5 3       | 0  1        2  3 |    T--0        3--T
    # 0                  3  =>   |                  | => |              |
    # | 1  2        4  5 |       | 4  5        6  7 |    T--4        7--T
    # |_|_/__________\_|_|       |_|_/__________\_|_|    |    5    6    |
    # |_______aC2x2______|       |_______aC2x2______|    C----T----T----C
    aC2x2 = torch.tensordot(aC2x2,aC2x2,([0,3],[0,3]))

    # permute such, that aux-index increases from "up" in the anti-clockwise direction
    # aC2x2 = aC2x2.permute(1,0,4,5,7,6,2,3).contiguous()
    # ===================================================
    aC2x2 = aC2x2.permute(1,0,4,5,6,7,3,2).contiguous()
    # reshape and form bra and ket index
    aC2x2 = aC2x2.view([dimsa[1]]*16).permute(0,2,4,6,8,10,12,14, 1,3,5,7,9,11,13,15).contiguous()

    return aC2x2

def test_symm_open_C2x2_LU(C, T, a):
    C2x2= _get_open_C2x2_LU_sl(C,T,a)
    tC2x2= C2x2.permute(1,0,2,3)
    print(f"open C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    C2x2= torch.einsum('ijaa->ij',C2x2)
    chi= C.size()[0]
    C2x2= C2x2.reshape(chi,a.size()[3]**2,chi,a.size()[4]**2)
    # |C2x2|--2    |C2x2|--1
    # |____|--3 => |____|--3
    # 0  1         0  2
    C2x2= C2x2.permute(0,2,1,3).contiguous()

    tC2x2= C2x2.permute(1,0,3,2)
    print(f"C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(1,0,2,3)
    print(f"C2x2-C2x2^t(env) {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(0,1,3,2)
    print(f"C2x2-C2x2^t(aux) {torch.norm(C2x2-tC2x2)}")

def test_symm_aux_C2x2_LU(env):
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    C2x2= _get_aux_C2x2_LU(C,T)

    tC2x2= C2x2.permute(1,0,3,2)
    print(f"C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(1,0,2,3)
    print(f"C2x2-C2x2^t(env) {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(0,1,3,2)
    print(f"C2x2-C2x2^t(aux) {torch.norm(C2x2-tC2x2)}")