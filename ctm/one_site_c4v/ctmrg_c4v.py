import time
from math import sqrt
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
from config import _torch_version_check
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v.ctm_components_c4v import *
from ctm.one_site_c4v.fpcm_c4v import fpcm_MOVE_sl
from linalg.custom_svd import *
from linalg.custom_eig import *
import logging
log = logging.getLogger(__name__)

def run(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args): 
    r"""
    :param state: wavefunction
    :param env: initial C4v symmetric environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type conv_check: function(IPEPS_C4V,ENV_C4V,Object,CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes specialized CTM algorithm for 1-site C4v symmetric iPEPS starting from the intial 
    environment ``env``. The is a single-layer version, which avoids explicitly building 
    double-layer on-site tensor.

    To establish the convergence of CTM before the maximal number of iterations 
    is reached  a ``conv_check`` function is invoked. Its expected signature is 
    ``conv_check(IPEPS_C4V,ENV_C4V,Object,CTMARGS)`` where ``Object`` is an arbitary argument. For 
    example it can be a list or dict used for storing CTM data from previous steps to   
    check convergence.

    If desired, CTM can be accelerated by fixed-point corner-matrix algorithm (FPCM) controlled 
    by settings in :py:class:`CTMARGS <config.CTMARGS>`.

    .. note::

        Currently, FPCM does not support reverse-mode differentiation.

    """

    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method=='SYMEIG':
        def truncated_eig(M, chi):
            return truncated_eig_sym(M, chi, keep_multiplets=True,\
                ad_decomp_reg=ctm_args.ad_decomp_reg, verbosity=ctm_args.verbosity_projectors)
    elif ctm_args.projector_svd_method == 'SYMARP':
        def truncated_eig(M, chi):
            return truncated_eig_symarnoldi(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    elif ctm_args.projector_svd_method == 'SYMLOBPCG':
        def truncated_eig(M, chi):
            return truncated_eig_symlobpcg(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    # elif ctm_args.projector_svd_method == 'GESDD':
    #     def truncated_eig(M, chi):
    #         return truncated_svd_gesdd(M, chi, verbosity=ctm_args.verbosity_projectors)
    # elif cfg.ctm_args.projector_svd_method == 'RSVD':
    #     truncated_svd= truncated_svd_rsvd
    else:
        raise Exception(f"Projector eig/svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    a= next(iter(state.sites.values()))

    # 1) perform CTMRG
    t_obs=t_ctm=t_fpcm=0.
    history=None
    past_steps_data=dict() # possibly store some data throughout the execution of CTM
    
    for i in range(ctm_args.ctm_max_iter):
        # FPCM acceleration
        if i>=ctm_args.fpcm_init_iter and ctm_args.fpcm_freq>0 and i%ctm_args.fpcm_freq==0:
            t0_fpcm= time.perf_counter()
            fpcm_MOVE_sl(a, env, ctm_args=ctm_args, global_args=global_args,
                past_steps_data=past_steps_data)
            t1_fpcm= time.perf_counter()
            t_fpcm+= t1_fpcm-t0_fpcm
            log.info(f"fpcm_MOVE_sl DONE t_fpcm {t1_fpcm-t0_fpcm} [s]")

        t0_ctm= time.perf_counter()
        ctm_MOVE_sl(a, env, truncated_eig, ctm_args=ctm_args, global_args=global_args,\
            past_steps_data=past_steps_data)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged, history= conv_check(state, env, history, ctm_args=ctm_args)
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG converged at iter= {i}")
                break
        t1_obs= time.perf_counter()
        
        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

def run_dl(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction
    :param env: initial C4v symmetric environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type conv_check: function(IPEPS,ENV_C4V,Object,CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    A double-layer variant (explicitly building double-layer tensor) of CTM algorithm.
    See :meth:`run`.
    """
    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method=='SYMEIG':
        def truncated_eig(M, chi):
            return truncated_eig_sym(M, chi, keep_multiplets=True,\
                verbosity=ctm_args.verbosity_projectors)
    elif ctm_args.projector_svd_method == 'SYMARP':
        def truncated_eig(M, chi):
            return truncated_eig_symarnoldi(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    elif ctm_args.projector_svd_method == 'SYMLOBPCG':
        def truncated_eig(M, chi):
            return truncated_eig_symlobpcg(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    else:
        raise Exception(f"Projector eig/svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    a= next(iter(state.sites.values()))

    # 1) perform CTMRG
    t_obs=t_ctm=t_fpcm=0.
    history=None
    past_steps_data=dict() # possibly store some data throughout the execution of CTM
    
    for i in range(ctm_args.ctm_max_iter):
        # FPCM acceleration
        if i>=ctm_args.fpcm_init_iter and ctm_args.fpcm_freq>0 and i%ctm_args.fpcm_freq==0:
            t0_fpcm= time.perf_counter()
            fpcm_MOVE_sl(a, env, ctm_args=ctm_args, global_args=global_args,
                past_steps_data=past_steps_data)
            t1_fpcm= time.perf_counter()
            t_fpcm+= t1_fpcm-t0_fpcm
            log.info(f"fpcm_MOVE_sl DONE t_fpcm {t1_fpcm-t0_fpcm} [s]")

        t0_ctm= time.perf_counter()
        ctm_MOVE_dl(a, env, truncated_eig, ctm_args=ctm_args, global_args=global_args)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged, history= conv_check(state, env, history, ctm_args=ctm_args)
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG converged at iter= {i}")
                break
        t1_obs= time.perf_counter()
        
        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

def _log_cuda_mem(device, who="unknown",  uuid=""):
    log.info(f"{who} {uuid} GPU-MEM MAX_ALLOC {torch.cuda.max_memory_allocated(device)}"\
            + f" CURRENT_ALLOC {torch.cuda.memory_allocated(device)}")

def _move_normalize_c(nC, nT, norm_type='inf', verbosity= 0):
        _ord=2
        if norm_type=='inf':
            _ord= float('inf')

        with torch.no_grad():
            scale_nC= torch.abs(nC[0,0])
            if _torch_version_check("1.9.0"):
                scale_nT= torch.linalg.vector_norm(nT,ord=_ord)
            else:
                scale_nT= nT.norm(p=_ord)
            if verbosity>0:
                print(f"nC {scale_nC} nT {scale_nT}")
        nC = nC/scale_nC
        nT = nT/scale_nT
        return nC, nT

# performs CTM move
def ctm_MOVE_dl(a, env, f_c2x2_decomp, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param a: on-site C4v symmetric tensor
    :param env: C4v symmetric environment
    :param f_c2x2_decomp: function performing the truncated spectral decomposition (eigenvalue/svd) 
                          of enlarged corner. The ``f_c2x2_decomp`` returns a tuple composed of
                          leading chi spectral values and projector on leading chi spectral values.
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type a: torch.Tensor
    :type env: ENV_C4V
    :type f_c2x2_decomp: function(torch.Tensor, int)->torch.Tensor, torch.Tensor
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes a single step of C4v symmetric CTM algorithm for 1-site C4v symmetric iPEPS.
    This variant of CTM step does explicitly build double-layer on-site tensor. 
    Hence, the leading cost in memory usage with bond dimension `D` is :math:`O(D^8)`.
    """
    who= "ctm_MOVE_dl"
    log_gpu_mem= False
    if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
        loc_gpu= torch.device(global_args.offload_to_gpu)
        log_gpu_mem= ctm_args.verbosity_ctm_move>0
    elif global_args.device != 'cpu':
        loc_gpu= a.device
        log_gpu_mem= ctm_args.verbosity_ctm_move>0

    # 0) extract raw tensors as tuple
    dimsa = a.size()
    if len(dimsa)==4:
        A= a
    else:
        A= torch.einsum('sefgh,sabcd->eafbgchd',a,a.conj()).contiguous()\
            .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)
        
    tensors= tuple([A,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_dl_c(*tensors):
        A, C, T= tensors
        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            A= A.to(global_args.offload_to_gpu) #A.cuda()
            C= C.to(global_args.offload_to_gpu) #C.cuda()
            T= T.to(global_args.offload_to_gpu) #T.cuda()

        # 1) build enlarged corner upper left corner
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="c2x2_dl_init")
        C2X2= c2x2_dl(A, C, T, verbosity=ctm_args.verbosity_projectors)
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="c2x2_dl_end")

        # 2) build projector
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="f_c2x2_decomp_init")
        #P, S, V = f_c2x2_decomp(C2X2, env.chi) # M = PSV^{T}
        D, P = f_c2x2_decomp(C2X2, env.chi) # M = PSV^{T}
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="f_c2x2_decomp_end")

        # 3) absorb and truncate
        #
        # C2X2--1 0--P--1
        # 0 
        # 0
        # P^t
        # 1->0
        # C2X2= P.t() @ C2X2 @ P
        C2X2= torch.diag((1.+0.j)*D) if C2X2.is_complex() else torch.diag(D)

        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="P-view_init")
        P= P.view(env.chi,T.size()[2],env.chi)
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="P-view_end")
        #      2->1
        #    __P__
        #   0     1->0
        # A 0
        # | T--2->3
        # | 1->2
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PT_init")
        nT = torch.tensordot(P, T,([0],[0]))
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PT_end")

        #    1->0
        #  __P____
        # |       0
        # |       0
        # T--3 1--A--3
        # 2->1    2 
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PTA_init")
        nT = torch.tensordot(nT, A,([0,3],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PTA_end")

        #    0
        #  __P____
        # |       |
        # |       |
        # T-------A--3->1
        # 1       2
        # 0       1
        # |___P___|
        #     2
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PTAP_init")
        nT = torch.tensordot(nT, P.conj(),([1,2],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PTAP_end")
        nT = nT.permute(0,2,1).contiguous()

        # 4) symmetrize, normalize and assign new C,T
        nT= 0.5*(nT + nT.permute(1,0,2).conj())
        C2X2, nT= _move_normalize_c(C2X2, nT, norm_type=ctm_args.ctm_absorb_normalization,\
            verbosity= ctm_args.verbosity_ctm_move)

        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            C2X2= C2X2.cpu()
            nT= nT.cpu()

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_dl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_dl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

# performs CTM move
def ctm_MOVE_sl(a, env, f_c2x2_decomp, ctm_args=cfg.ctm_args, global_args=cfg.global_args,
    past_steps_data=None):
    r"""
    :param a: on-site C4v symmetric tensor
    :param env: C4v symmetric environment
    :param f_c2x2_decomp: function performing the truncated spectral decomposition (eigenvalue/svd) 
                          of enlarged corner. The ``f_c2x2_decomp`` returns a tuple composed of
                          leading chi spectral values and projector on leading chi spectral values.
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :param past_steps_data: dictionary used for recording diagnostic information during CTM 
    :type a: torch.Tensor
    :type env: ENV_C4V
    :type f_c2x2_decomp: function(torch.Tensor, int)->torch.Tensor, torch.Tensor
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :type past_steps_data:

    Executes a single step of C4v symmetric CTM algorithm for 1-site C4v symmetric iPEPS.
    This variant of CTM step does not explicitly build double-layer on-site tensor.
    """

    # 0) extract raw tensors as tuple
    tensors= tuple([a,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_sl_c(*tensors):
        a, C, T= tensors
        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            #loc_gpu= torch.device(global_args.gpu)
            a= a.to(global_args.offload_to_gpu) #a.cuda()
            C= C.to(global_args.offload_to_gpu) #C.cuda()
            T= T.to(global_args.offload_to_gpu) #T.cuda()

        # 1) build enlarged corner upper left corner
        C2X2= c2x2_sl(a, C, T, verbosity=ctm_args.verbosity_projectors)

        # 2) build projector
        # P, S, V = f_c2x2_decomp(C2X2, env.chi) # M = PSV^T
        D, P= f_c2x2_decomp(C2X2, env.chi) # M = UDU^T
     
        # 3) absorb and truncate
        #
        # C2X2--1 0--P--1
        # 0
        # 0
        # P^t
        # 1->0
        # C2X2= P.t() @ C2X2 @ P
        C2X2= torch.diag((1.+0.j)*D) if C2X2.is_complex() else torch.diag(D)

        P= P.view(env.chi,T.size()[2],env.chi)
        #      2->1
        #    __P__
        #   0     1->0
        # A 0
        # | T--2->3
        # | 1->2
        nT= torch.tensordot(P, T,([0],[0]))

        # 4) double-layer tensor contraction - layer by layer
        # 4i) untangle the fused D^2 indices
        #    1->2
        #  __P__
        # |     0->0,1
        # |
        # T--3->4,5
        # 2->3
        nT= nT.view(a.size()[1],a.size()[1],nT.size()[1],nT.size()[2],\
            a.size()[2],a.size()[2])

        # 4ii) first layer "bra" (in principle conjugate)
        #    2->1
        #  __P___________
        # |         0    1->0
        # |         1 /0->4
        # T----4 2--a--4->6 
        # | |       3->5
        # |  --5->3
        # 3->2
        nT= torch.tensordot(nT, a,([0,4],[1,2]))

        # 4iii) second layer "ket"
        #    1->0
        #  __P__________
        # |    |       0
        # |    |/4 0\  | 
        # T----a---------6->3 
        # | |  |      \1
        # |  -----3 2--a--4->5
        # |    |       3->4
        # |    |
        # 2->1 5->2
        nT= torch.tensordot(nT, a.conj(),([0,3,4],[1,2,0]))

        # 4iv) fuse pairs of aux indices
        #    0
        #  __P_
        # |    | 
        # T----a----3\ 
        # | |  |\     ->3 
        # |  ----a--5/
        # |    | |
        # |    | |
        # 1   (2 4)->2
        nT= nT.permute(0,1,2,4,3,5).contiguous().view(nT.size()[0],nT.size()[1],\
            a.size()[3]**2,a.size()[4]**2)

        #    0
        #  __P____
        # |       |
        # |       |          A 0
        # T------aa--3->1 => | T--2
        # 1       2          | 1
        # 0       1
        # |___P___|
        #     2
        nT = torch.tensordot(nT,P.conj(),([1,2],[0,1]))
        nT = nT.permute(0,2,1).contiguous()

        # 4) symmetrize, normalize and assign new C,T
        nT= 0.5*(nT + nT.conj().permute(1,0,2))    
        C2X2, nT= _move_normalize_c(C2X2, nT, norm_type=ctm_args.ctm_absorb_normalization,\
            verbosity= ctm_args.verbosity_ctm_move)

        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            C2X2= C2X2.cpu()
            nT= nT.cpu()

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_sl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_sl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]