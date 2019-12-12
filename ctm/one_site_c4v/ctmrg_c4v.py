import time
from math import sqrt
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
import ipeps
from ipeps import IPEPS
from ctm.one_site_c4v.env_c4v import *
from custom_svd import *

def run(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args): 
    r"""
    :param state: wavefunction
    :param env: initial C4v symmetric environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type env: ENV_C4V
    :type conv_check: function(IPEPS,ENV_C4V,list[float],CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes specialized CTM algorithm for 1-site C4v symmetric iPEPS starting from the intial 
    environment ``env``. TODO add reference
    """

    if cfg.ctm_args.projector_svd_method == 'GESDD':
        truncated_svd= truncated_svd_gesdd
    elif cfg.ctm_args.projector_svd_method == 'GESDD_SU2':
        truncated_svd= truncated_svd_gesdd_su2
    elif cfg.ctm_args.projector_svd_method == 'SYM':
        def truncated_svd(M, chi):
            return truncated_svd_symeig(M, chi, env=env, verbosity=ctm_args.verbosity_projectors)
    elif cfg.ctm_args.projector_svd_method == 'RSVD':
        truncated_svd= truncated_svd_rsvd
    elif cfg.ctm_args.projector_svd_method == 'ARPACK':
        truncated_svd= truncated_svd_arnoldi

    # 0) Create double-layer (DL) tensors, preserving the same convenction
    # for order of indices 
    #
    #     /           /
    #  --A^dag-- = --a--
    #   /|          /
    #    |/
    #  --A--
    #   /
    #
    sitesDL=dict()
    for coord,A in state.sites.items():
        dimsA = A.size()
        a = torch.einsum('mefgh,mabcd->eafbgchd',(A,A)).contiguous().view(dimsA[1]**2,\
            dimsA[2]**2, dimsA[3]**2, dimsA[4]**2)
        sitesDL[coord]=a
    stateDL = IPEPS(sitesDL,state.vertexToSite)

    # 1) perform CTMRG
    t_obs=0.
    t0= time.perf_counter()
    history=[]
    for i in range(ctm_args.ctm_max_iter):
        ctm_MOVE(stateDL, env, truncated_svd, ctm_args=ctm_args, global_args=global_args)

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged = conv_check(state, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history[-1])
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG  converged at iter= {i}, history= {history[-1]}")
                break
            elif len(history)>4:
                # we are not converged, but perhaps we have entered stationary
                # oscilatory regime x_i=x_{i+2}, x_{i+1}=x_{i+3} but |x_i-x_{i+1}|=const > small_eps

                # 1) check whether the last value of loss is the on upper or lower branch
                lower= history[-1] < history[-2] 

                # 2) check if the upper branch decreases and the lower branch increases
                del0= history[-1]-history[-3]
                del1= history[-2]-history[-4]
                dec_upper= del1 < 0 if lower else del0 < 0
                inc_lower= del0 > 0 if lower else del1 > 0

                # 3) if both lower and upper branch is converged within eps of ctm, but
                # their difference is larger than large_eps
                if abs(del0) < ctm_args.ctm_conv_tol and abs(del1) < ctm_args.ctm_conv_tol \
                    and abs(history[-1] - history[-2]) > sqrt(ctm_args.ctm_conv_tol):
                    if not lower:
                        break
                    # if we are on lower branch, make an extra ctm step
                    else:
                        ctm_MOVE(stateDL, env, truncated_svd, ctm_args=ctm_args, global_args=global_args)
                        history.append(history[-2])
                        break

        t1_obs= time.perf_counter()
        t_obs+= t1_obs-t0_obs


    t1= time.perf_counter()

    return env, history, t1-t0, t_obs

# performs CTM move
def ctm_MOVE(state, env, svd_method, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # 0) extract raw tensors as tuple
    tensors= tuple([next(iter(state.sites.values())),env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_c(*tensors):
        A, C, T= tensors
        # 1) build enlarged corner upper left corner
        C2X2= c2x2(A, C, T, verbosity=ctm_args.verbosity_projectors)

        # 2) build projector
        P, S, V = svd_method(C2X2, env.chi) # M = PSV^{T}

        # 3) absorb and truncate
        #
        # C2X2--1 0--P--1
        # 0 
        # 0
        # P^t
        # 1->0
        C2X2= P.t() @ C2X2 @ P

        P= P.view(env.chi,T.size()[2],env.chi)
        #    2->1
        #  __P__
        # 0     1->0
        # 0
        # T--2->3
        # 1->2
        nT = torch.tensordot(P, T,([0],[0]))

        #    1->0
        #  __P____
        # |       0
        # |       0
        # T--3 1--A--3
        # 2->1    2
        nT = torch.tensordot(nT, A,([0,3],[0,1]))

        #    0
        #  __P____
        # |       |
        # |       |
        # T-------A--3->1
        # 1       2
        # 0       1
        # |___P___|
        #     2
        nT = torch.tensordot(nT, P,([1,2],[0,1]))
        nT = nT.permute(0,2,1).contiguous()

        # 4) symmetrize, normalize and assign new C,T
        C2X2= 0.5*(C2X2 + C2X2.t())
        nT= 0.5*(nT + nT.permute(1,0,2))
        C2X2= C2X2/torch.max(torch.abs(C2X2))
        nT= nT/torch.max(torch.abs(nT))

        ret_list= tuple([C2X2, nT])
        return ret_list

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_c,*tensors)
    else:
        new_tensors= ctm_MOVE_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

def c2x2(A, C, T, verbosity=0):
    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3 
    # 2->1    2
    C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T.size()[1]*A.size()[3],T.size()[1]*A.size()[2])

    # C2x2--1
    # |
    # 0
    if verbosity>1: print(C2x2)

    return C2x2
