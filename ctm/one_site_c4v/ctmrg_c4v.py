import time
import torch
import config as cfg
import ipeps
from ipeps import IPEPS
from ctm.one_site_c4v.env_c4v import *
from custom_svd import *

if cfg.ctm_args.projector_svd_method == 'GESDD':
    truncated_svd= truncated_svd_gesdd
elif cfg.ctm_args.projector_svd_method == 'RSVD':
    truncated_svd= truncated_svd_rsvd

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
    t0= time.perf_counter()
    history=[]
    for i in range(ctm_args.ctm_max_iter):
        ctm_MOVE(stateDL, env, ctm_args=ctm_args, global_args=global_args)

        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged = conv_check(state, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history[-1])
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG  converged at iter= {i}, history= {history[-1]}")
                break
    t1= time.perf_counter()

    return env, history, t1-t0

# performs CTM move
def ctm_MOVE(state, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # 1) build enlarged corner upper left corner
    C2X2= c2x2(state, env, verbosity=ctm_args.verbosity_projectors)

    # 2) build projector
    P, S, V = truncated_svd(C2X2, env.chi) # M = PSV^{T}
    # No inversion, thus no relative tolerance
    # S = S[S/S[0] > ctm_args.projector_svd_reltol]
    # S_zeros = torch.zeros((chi-S.size()[0]), dtype=global_args.dtype, device=global_args.device)
    # S_sqrt = torch.rsqrt(S)
    # S_sqrt = torch.cat((S_sqrt, S_zeros))
    if ctm_args.verbosity_projectors>0: print(S)

    # 3) absorb and truncate
    #
    # C2X2--1 0--P--1
    # 0 
    # 0
    # P^t
    # 1->0
    C2X2= P.t() @ C2X2 @ P

    T= env.T[env.keyT]
    A= next(iter(state.sites.values()))
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
    env.C[env.keyC]= C2X2/torch.max(torch.abs(C2X2))
    env.T[env.keyT]= nT/torch.max(torch.abs(nT))

def c2x2(state, env, verbosity=0):
    C= env.C[env.keyC]
    T= env.T[env.keyT]
    A= next(iter(state.sites.values()))

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
