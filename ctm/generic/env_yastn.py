import time
import config as cfg
from typing import Callable
import yastn.yastn as yastn
from yastn.yastn.backend import backend_torch
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs.fixed_pt import NoFixedPointError
from ctm.generic.env import ENV
from ipeps.ipeps import IPEPS
from ipeps.integration_yastn import PepsAD

YASTN_ENV_INIT={"CTMRG": "dl", "PROD": "eye"}
YASTN_PROJ_METHOD={"DEFAULT": "fullrank", "GESDD": "fullrank", "QR": "qr",
                   "RSVD": "randomized",
                   "ARP": "block_arnoldi", "PROPACK": "block_propack" }


def from_yastn_env_generic(env_yastn: EnvCTM, vertexToSite: Callable= None) -> ENV:
    assert env_yastn.geometry.boundary=='infinite'

    # x and y axes are interchanged
    _map_site= lambda c: (vertexToSite((c[1], c[0])) if vertexToSite else c)
    # TODO: determine dtype and device from yastn.config
    # TODO: determine chi
    pt_env= ENV(chi=0)

    # run over non-equivalent sites in the primitive unit cell
    for site in env_yastn.sites():
        assert len(site)==2 and type(site[0])==int and type(site[0])==int

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


def from_env_generic_dense_to_yastn(env: ENV, state: IPEPS, global_args=cfg.global_args) -> EnvCTM:
    """
    """
    # site convention  (-)0 /4(+)
    #               (+)1--A--3(-)
    #                  (+)2
    state_yastn= PepsAD.from_pt(state,global_args=global_args).to_Peps()

    yastn_cfg_nosym= yastn.make_config(
        backend= backend_torch,
        default_dtype= dict(zip(backend_torch.DTYPE.values(), backend_torch.DTYPE.keys()))[state.dtype],
        default_device= global_args.device
    )
    def _wrap_t(t,s):
        res= yastn.Tensor(config=yastn_cfg_nosym, s=s)
        res.set_block(val=t, Ds=t.shape)
        return res

    env_yastn= EnvCTM(psi=state_yastn, init=None)
    for (site, d), t in env.C.items():
        yastn_site = (site[1], site[0])
        if d == (-1, -1):
            setattr(env_yastn[yastn_site], 'tl', _wrap_t(t,s=[1,-1]))
        elif d == (1, -1):
            setattr(env_yastn[yastn_site], 'tr', _wrap_t(t,s=[1,1]))
        elif d == (1, 1):
            setattr(env_yastn[yastn_site], 'br', _wrap_t(t,s=[-1,1]))
        else:
            setattr(env_yastn[yastn_site], 'bl', _wrap_t(t.permute(1,0),s=[-1,-1]))

    for (site, d), t in env.T.items():
        yastn_site = (site[1], site[0])
        if d == (-1, 0):
            # 0       2
            # T--2 -> T--1
            # 1       0
            setattr(env_yastn[yastn_site], 'l', _wrap_t(t.permute(1,2,0).reshape(
                [t.shape[1]]+[state.site(site).shape[2],]*2+[t.shape[0]]),s=[1,-1,1,-1]).fuse_legs(axes=(0,(1,2),3)) )
        elif d == (0, -1):
            # 0-- T --2
            #     1
            setattr(env_yastn[yastn_site], 't', _wrap_t(t.reshape([t.shape[0]]+[state.site(site).shape[1],]*2+[t.shape[2]]),
                                                   s=[1,1,-1,-1]).fuse_legs(axes=(0,(1,2),3)) )
        elif d == (1, 0):
            #    0
            # 1--T
            #    2
            setattr(env_yastn[yastn_site], 'r', _wrap_t(t.reshape([t.shape[0]]+[state.site(site).shape[4],]*2+[t.shape[2]]),
                                                   s=[-1,1,-1,1]).fuse_legs(axes=(0,(1,2),3)) )
        else:
            #    0          1
            # 1--T--2 -> 2--T--0
            setattr(env_yastn[yastn_site], 'b', _wrap_t(t.permute(2,0,1).reshape(
                [t.shape[2]]+[state.site(site).shape[3],]*2+[t.shape[1]]),s=[-1,-1,1,1]).fuse_legs(axes=(0,(1,2),3)) )

    return env_yastn


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