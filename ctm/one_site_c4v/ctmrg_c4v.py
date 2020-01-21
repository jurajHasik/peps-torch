import time
from math import sqrt
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
import ipeps
from ipeps import IPEPS
from ctm.one_site_c4v.env_c4v import *
from linalg.custom_svd import *

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
        def truncated_svd(M, chi):
            return truncated_svd_gesdd(M, chi, verbosity=ctm_args.verbosity_projectors)
    elif cfg.ctm_args.projector_svd_method == 'SYMEIG':
        def truncated_svd(M, chi):
            return truncated_svd_symeig(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    elif cfg.ctm_args.projector_svd_method == 'SYMARP':
        def truncated_svd(M, chi):
            return truncated_svd_symarnoldi(M, chi, keep_multiplets=True, \
                verbosity=ctm_args.verbosity_projectors)
    elif cfg.ctm_args.projector_svd_method == 'RSVD':
        truncated_svd= truncated_svd_rsvd
    else:
        raise(f"Projector svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    a= next(iter(state.sites.values()))
    #A = torch.einsum('mefgh,mabcd->eafbgchd',(a,a)).contiguous().view(a.shape[1]**2,\
    #        a.shape[2]**2, a.shape[3]**2, a.shape[4]**2)

    # 1) perform CTMRG
    t_obs=t_ctm=0.
    history=[]
    for i in range(ctm_args.ctm_max_iter):
        t0_ctm= time.perf_counter()
        # ctm_MOVE_dl(A, env, truncated_svd, ctm_args=ctm_args, global_args=global_args)
        ctm_MOVE_sl(a, env, truncated_svd, ctm_args=ctm_args, global_args=global_args)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged = conv_check(state, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history[-1])
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG converged at iter= {i}, history= {history[-1]}")
                break
        t1_obs= time.perf_counter()
        
        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

# performs CTM move
def ctm_MOVE_dl(A, env, svd_method, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # 0) extract raw tensors as tuple
    tensors= tuple([A,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_dl_c(*tensors):
        A, C, T= tensors
        # 1) build enlarged corner upper left corner
        C2X2= c2x2_dl(A, C, T, verbosity=ctm_args.verbosity_projectors)

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
        # C2X2= torch.diag(S)

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

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_dl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_dl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

def c2x2_dl(A, C, T, verbosity=0):
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

# performs CTM move
def ctm_MOVE_sl(a, env, svd_method, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # 0) extract raw tensors as tuple
    tensors= tuple([a,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_sl_c(*tensors):
        a, C, T= tensors
        # 1) build enlarged corner upper left corner
        C2X2= c2x2_sl(a, C, T, verbosity=ctm_args.verbosity_projectors)

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
        # TODO allow for symdiag instead of SVD
        # C2X2= torch.diag(S)

        P= P.view(env.chi,T.size()[2],env.chi)
        #    2->1
        #  __P__
        # 0     1->0
        # 0
        # T--2->3
        # 1->2
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
        nT= torch.tensordot(nT, a,([0,3,4],[1,2,0]))

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
        # |       |
        # T------aa--3->1
        # 1       2
        # 0       1
        # |___P___|
        #     2
        nT = torch.tensordot(nT,P,([1,2],[0,1]))
        nT = nT.permute(0,2,1).contiguous()

        # 4) symmetrize, normalize and assign new C,T
        C2X2= 0.5*(C2X2 + C2X2.t())
        nT= 0.5*(nT + nT.permute(1,0,2))
        C2X2= C2X2/torch.max(torch.abs(C2X2))
        nT= nT/torch.max(torch.abs(nT))

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_sl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_sl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

def c2x2_sl(a, C, T, verbosity=0):
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
    # C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))

    # 4) double-layer tensor contraction - layer by layer
    # 4i) untangle the fused D^2 indices
    # 
    # C-----T--0
    # |     1->1,2
    # |  
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],\
        a.size()[2],a.size()[2])

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
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))

    # 4iii) second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |/4 0\  |
    # T----a----------6->3 
    # | |  |      \1
    # |  -----3 2--a--4->5
    # |    |       3->4
    # |    |
    # 2->1 5->2
    C2x2= torch.tensordot(C2x2, a,([1,3,4],[1,2,0]))

    # 4iv) fuse pairs of aux indices
    #
    # C----T----0
    # |    |\
    # T----a----3\ 
    # | |  |\|    ->3
    # |  ----a--5/
    # |    | |
    # 1   (2 4)->2 
    # 
    # and simultaneously
    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2= C2x2.permute(1,2,4,0,3,5).contiguous().view(C2x2.size()[1]*(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2))

    # C2x2--1
    # |
    # 0
    if verbosity>1: print(C2x2)

    return C2x2