import time
from typing import Callable

import config as cfg
from yastn.yastn.tn.fpeps import EnvCTM
from ctm.generic_abelian.env_abelian import ENV_ABELIAN

from ctm.generic.env_yastn import ctmrg


def _set_signature(t,expected_s):
    """ Change signature of a tensor to expected_s."""
    return t if t.s==expected_s else t.switch_signature(axes=[i for i,s_e in enumerate(expected_s) if s_e!=t.s[i]])


def from_yastn_env_generic(env_yastn: EnvCTM, vertexToSite: Callable = None,\
                           ctm_args=cfg.ctm_args, global_args=cfg.global_args) -> ENV_ABELIAN:
    assert env_yastn.geometry.boundary=='infinite'

    _map_site= lambda c: (vertexToSite(c) if vertexToSite else c)
    config= env_yastn.config
    # TODO: determine chi
    pt_env= ENV_ABELIAN(chi=0, settings=config, ctm_args=ctm_args, global_args=global_args)

    # run over non-equivalent sites in the primitive unit cell
    for site in env_yastn.sites():
        assert type(site)==tuple and len(site)==2 and type(site[0])==int and type(site[0])==int

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
