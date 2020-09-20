import time
from math import sqrt
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import *
from ctm.one_site_c4v.ctm_components_c4v import *
from ctm.one_site_c4v.fpcm_c4v import fpcm_MOVE_sl
from linalg.custom_svd import *
from linalg.custom_eig import *
import logging
log = logging.getLogger(__name__)

def _log_cuda_mem(device, who="unknown",  uuid=""):
    log.info(f"{who} {uuid} GPU-MEM MAX_ALLOC {torch.cuda.max_memory_allocated(device)}"\
            + f" CURRENT_ALLOC {torch.cuda.memory_allocated(device)}")

def run_dl_SYMARP(state, env, conv_check=None, ctm_args=cfg.ctm_args, \
    global_args=cfg.global_args):
    
    who= "ctm_MOVE_dl_SYMARP"
    log_gpu_mem= False
    if (global_args.device=='cpu' and ctm_args.step_core_gpu):
        loc_gpu= torch.device(global_args.gpu)
        log_gpu_mem= ctm_args.verbosity_ctm_move>0
    elif global_args.device != 'cpu':
        loc_gpu= a.device
        log_gpu_mem= ctm_args.verbosity_ctm_move>0

    # 0) extract raw tensors as tuple
    a= next(iter(state.sites.values()))
    dimsa = a.size()
    A = torch.einsum('sefgh,sabcd->eafbgchd',a,a).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_dl_c(A,C,T):
        if global_args.device=='cpu' and ctm_args.step_core_gpu:
            A= A.cuda()
            C= C.cuda()
            T= T.cuda()

        # 1) build enlarged corner upper left corner
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="c2x2_dl_init")
        C2X2= c2x2_dl(A, C, T, verbosity=ctm_args.verbosity_projectors)
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="c2x2_dl_end")

        # 2) build projector
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="f_c2x2_decomp_init")
        D, P = truncated_eig_symarnoldi(C2X2, env.chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="f_c2x2_decomp_end")

        # 3) absorb and truncate
        #
        # C2X2--1 0--P--1
        # 0 
        # 0
        # P^t
        # 1->0
        C2X2= torch.diag(D)

        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="P-view_init")
        P= P.view(env.chi,T.size()[2],env.chi)
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="P-view_end")
        #    2->1
        #  __P__
        # 0     1->0
        # 0
        # T--2->3
        # 1->2
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
        nT = torch.tensordot(nT, P,([1,2],[0,1]))
        if log_gpu_mem: _log_cuda_mem(loc_gpu, who=who, uuid="PTAP_end")
        nT = nT.permute(0,2,1).contiguous()

        # 4) symmetrize, normalize and assign new C,T
        C2X2= 0.5*(C2X2 + C2X2.t())
        nT= 0.5*(nT + nT.permute(1,0,2))
        C2X2= C2X2/torch.max(torch.abs(C2X2))
        nT= nT/torch.max(torch.abs(nT))

        if global_args.device=='cpu' and ctm_args.step_core_gpu:
            C2X2= C2X2.cpu()
            nT= nT.cpu()

        return C2X2, nT

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
        # Call the core function, allowing for checkpointing
        if ctm_args.fwd_checkpoint_move:
            new_tensors= checkpoint(ctm_MOVE_dl_c, A, env.C[env.keyC], env.T[env.keyT])
        else:
            new_tensors= ctm_MOVE_dl_c(A, env.C[env.keyC], env.T[env.keyT])
        env.C[env.keyC]= new_tensors[0]
        env.T[env.keyT]= new_tensors[1]
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