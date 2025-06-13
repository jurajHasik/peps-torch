import time
from typing import Callable
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs.fixed_pt import NoFixedPointError
from ctm.generic.env import ENV

YASTN_ENV_INIT={"CTMRG": "dl", "PROD": "eye"}
YASTN_PROJ_METHOD={"DEFAULT": "fullrank", "GESDD": "fullrank", "QR": "qr",
                   "RSVD": "randomized",
                   "ARP": "block_arnoldi", "PROPACK": "block_propack" }


def from_yastn_env_generic(env_yastn: EnvCTM, vertexToSite: Callable= None) -> ENV:
    assert env_yastn.geometry.boundary=='infinite'

    _map_site= lambda c: (vertexToSite(c) if vertexToSite else c)
    # TODO: determine dtype and device from yastn.config
    # TODO: determine chi
    pt_env= ENV(chi=0)

    # run over non-equivalent sites in the primitive unit cell
    for site in env_yastn.sites():
        assert type(site)==tuple and len(site)==2 and type(site[0])==int and type(site[0])==int

        # corners
        for rel_c,rel_label in zip([(-1,-1),(1,-1),(1,1),(-1,1)], ('tl', 'tr', 'br', 'bl')):
            if rel_label=='bl':
                # ENV   <=> YASTN
                # 0         1
                # C--1      C--0
                pt_env.C[(_map_site(site),rel_c)]= env_yastn[site].__getattribute__(rel_label).transpose(axes=(1,0)).to_dense()
            else:
                pt_env.C[(_map_site(site),rel_c)]= env_yastn[site].__getattribute__(rel_label).to_dense()
            if pt_env.C[(_map_site(site),rel_c)]._is_view():
                pt_env.C[(_map_site(site),rel_c)]= pt_env.C[(_map_site(site),rel_c)].clone()

        # half-row/-column
        for rel_c,rel_label in zip([(-1,0),(0,-1),(1,0),(0,1)], ('l', 't', 'r', 'b')):
            if rel_label=='l':
                # ENV   <=> YASTN
                # 0         2
                # T--2      T--1
                # 1         0
                pt_env.T[(_map_site(site),rel_c)]= env_yastn[site].__getattribute__(rel_label).transpose(axes=(2,0,1)).to_dense()
            elif rel_label=='b':
                # ENV      <=> YASTN
                #    0            1
                # 1--T--2      2--T--0
                pt_env.T[(_map_site(site),rel_c)]= env_yastn[site].__getattribute__(rel_label).transpose(axes=(1,2,0)).to_dense()
            else:
                pt_env.T[(_map_site(site),rel_c)]= env_yastn[site].__getattribute__(rel_label).to_dense()
            if pt_env.T[(_map_site(site),rel_c)]._is_view():
                pt_env.T[(_map_site(site),rel_c)]= pt_env.T[(_map_site(site),rel_c)].clone()
    pt_env.chi= pt_env.min_chi()

    return pt_env


def ctmrg(env: EnvCTM, ctm_conv_check_f : Callable, options_svd : dict, **kwargs):
    # t_ctm, t_check = 0.0, 0.0
    # t_ctm_prev = time.perf_counter()
    # converged,conv_history=False,[]
    # checkpoint_move= kwargs.get("checkpoint_move",False)
    # if checkpoint_move is True: # default
    #     checkpoint_move= "nonreentrant"

    # for sweep in range(kwargs.get("max_sweeps",0)):
    #     env.update_(options_svd,
    #                 method=kwargs.get("method","2site"),
    #                 use_qr=kwargs.get("use_qr",False), \
    #                 checkpoint_move=checkpoint_move)
    #     t_ctm_after = time.perf_counter()
    #     t_ctm += t_ctm_after - t_ctm_prev
    #     t_ctm_prev = t_ctm_after

    # converged, conv_history= ctm_conv_check_f(env, conv_history)
    # if converged:
    #     break
    max_sweeps= kwargs.pop("max_sweeps", 0)
    method=kwargs.pop("method","2site")
    ctm_itr= env.ctmrg_(iterator_step=1, method=method,  max_sweeps=max_sweeps, opts_svd=options_svd, corner_tol=None, **kwargs)

    t_ctm, t_check = 0.0, 0.0
    converged,conv_history=False,[]
    for sweep in range(max_sweeps):
        t0 = time.perf_counter()
        ctm_out_info= next(ctm_itr)
        t1 = time.perf_counter()
        t_ctm += t1-t0

        t2 = time.perf_counter()
        converged, conv_history = ctm_conv_check_f(env, conv_history)
        t_check += time.perf_counter()-t2
        if converged:
            break

    return env, converged, conv_history, t_ctm, t_check