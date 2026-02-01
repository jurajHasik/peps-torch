import time
from typing import Callable

import config as cfg
from yastn.yastn.tn.fpeps import EnvCTM, EnvCTM_c4v, RectangularUnitcell, Site

from ipeps.integration_yastn import PepsAD
from ctm.generic_abelian.env_abelian import ENV_ABELIAN
from ctm.one_site_c4v_abelian.env_c4v_abelian import ENV_C4V_ABELIAN
from ctm.generic.env_yastn import ctmrg


def _set_signature(t,expected_s):
    """ Change signature of a tensor to expected_s."""
    from yastn.yastn.tensor._merging import _unfuse_Fusion

    if t.s==expected_s:
        return t
    else:

        def _unravel_fusion_1l(i, f_curr, depth, hf):
            if len(hf.tree) <= 1 or depth>0:
                f_curr.append(i)
                i += 1
                # print(f"{i} {depth}")
            else:
                f_curr.append([])
                for lower_hf in _unfuse_Fusion(hf)[-1]:
                    i = _unravel_fusion_1l(i, f_curr[-1], depth+1, lower_hf)
            return i

        # Unravel fusion tree as sequence of fusion layers, i.e. nested axes indices
        fusion_layers=[]
        def _unravel_layer(T):
            fusions=[]
            i=0
            for l in T.get_legs():
                i= _unravel_fusion_1l(i,fusions,0,l.hf)
            return fusions

        # pre-compute expected signature. This can be done without unfusing
        exp_s_unfused= []
        for s_e,l in zip(expected_s,t.get_legs()):
            _s= l.s*s_e
            exp_s_unfused.append([_s * s for s in l.hf.s[1:]] if l.is_fused() else [_s*l.hf.s[0]])
        exp_s_unfused= sum(exp_s_unfused,[])

        fusion_layers.append(_unravel_layer(t))
        while any( isinstance(g,list) for g in fusion_layers[-1]):
            t= t.unfuse_legs(axes=[j for j,l in enumerate(t.get_legs()) if l.is_fused()])
            fusion_layers.append(_unravel_layer(t))
        # print(fusion_layers)

        t= t.switch_signature(axes=[i for i,s_e in enumerate(exp_s_unfused) if s_e!=t.s[i]])

        # fuse back to original
        for l in fusion_layers[::-1]:
            t= t.fuse_legs(axes=l)
        return t

        # return t.switch_signature(axes=[i for i,s_e in enumerate(expected_s) if s_e!=t.s[i]])

def from_yastn_env_generic(env_yastn: EnvCTM, vertexToSite: Callable = None,\
                           ctm_args=cfg.ctm_args, global_args=cfg.global_args) -> ENV_ABELIAN:
    assert env_yastn.geometry.boundary=='infinite'

    _map_site= lambda c: (vertexToSite((c[1], c[0])) if vertexToSite else c)
    config= env_yastn.config
    # TODO: determine chi
    pt_env= ENV_ABELIAN(chi=0, settings=config, ctm_args=ctm_args, global_args=global_args)

    # run over non-equivalent sites in the primitive unit cell
    for site in env_yastn.sites():
        assert len(site)==2 and type(site[0])==int and type(site[0])==int

        # corners
        for rel_c,rel_label in zip([(-1,-1),(1,-1),(1,1),(-1,1)], ('tl', 'tr', 'br', 'bl')):
            if rel_label=='tl':
                # C--1(+)
                # 0(+)
                pt_env.C[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label), (1,1))
            elif rel_label=='tr':
                # (-)0--C
                #    (+)1
                pt_env.C[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label), (-1,1))
            elif rel_label=='br':
                #    (-)0
                # (-)1--C
                pt_env.C[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label), (-1,-1))
            elif rel_label=='bl':
                # ENV      <=> YASTN
                # 0(-)         1
                # C--1(+)      C--0
                pt_env.C[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label).transpose(axes=(1,0)), (-1,1))

        # half-row/-column
        for rel_c,rel_label in zip([(-1,0),(0,-1),(1,0),(0,1)], ('l', 't', 'r', 'b')):
            if rel_label=='l':
                # ENV   <=> YASTN
                # 0(-)         2
                # T--2(+)      T--1
                # 1(+)         0
                pt_env.T[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label).transpose(axes=(2,0,1)), (-1,1,1))
            elif rel_label=='b':
                # ENV           <=> YASTN
                #       0(-)               1
                # (-)1--T--2(+)         2--T--0
                pt_env.T[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label).transpose(axes=(1,2,0)), (-1,-1,1))
            elif rel_label=='r':
                # ENV
                #    (-)0
                # (-)1--T
                #    (+)2
                pt_env.T[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label), (-1,-1,1))
            elif rel_label=='t':
                # ENV
                # (-)0--T--2(+)
                #       1(+)
                pt_env.T[(_map_site(site),rel_c)]= _set_signature(env_yastn[site].__getattribute__(rel_label), (-1,1,1))

    return pt_env


def from_yastn_c4v_env_generic(env_yastn: EnvCTM_c4v, \
                           ctm_args=cfg.ctm_args, global_args=cfg.global_args) -> ENV_ABELIAN:
    assert env_yastn.geometry.boundary=='infinite'

    env_yastn[0,0].tr= env_yastn[0,0].br= env_yastn[0,0].bl= env_yastn[0,0].tl
    env_yastn[0,0].l= env_yastn[0,0].b= env_yastn[0,0].r= env_yastn[0,0].t

    env_bp= from_yastn_env_generic(env_yastn, vertexToSite=None, ctm_args=ctm_args, global_args=global_args)
    for rel_dir in (-1,-1),(1,-1),(1,1),(-1,1):
        env_bp.C[((1,0),rel_dir)]= env_bp.C[((0,0),rel_dir)].conj().switch_signature(axes='all')
    for rel_dir in (-1,0),(1,0),(0,1),(0,-1):
        env_bp.T[((1,0),rel_dir)]= env_bp.T[((0,0),rel_dir)].conj().switch_signature(axes='all')

    return env_bp


def from_yastn_c4v_env_c4v(env_yastn: EnvCTM_c4v, \
                           ctm_args=cfg.ctm_args, global_args=cfg.global_args) -> ENV_ABELIAN:
    assert env_yastn.geometry.boundary=='infinite'

    env_c4v= ENV_C4V_ABELIAN(settings=env_yastn.config, chi=0, ctm_args=ctm_args, global_args=global_args)
    env_c4v.C[env_c4v.keyC]= _set_signature(env_yastn[0,0].tl,(-1,-1))
    env_c4v.T[env_c4v.keyT]= _set_signature(env_yastn[0,0].t.transpose(axes=(0,2,1)),(1,1,-1)).unfuse_legs(axes=2)

    return env_c4v
