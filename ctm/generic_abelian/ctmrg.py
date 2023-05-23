import time
import warnings
import copy
from typing import NamedTuple
import config as cfg
from yastn.yastn import decompress_from_1d
from ipeps.ipeps_abelian import IPEPS_ABELIAN, _fused_dl_site
from ctm.generic_abelian.env_abelian import ENV_ABELIAN
from ctm.generic_abelian.ctm_components import *
from ctm.generic_abelian.ctm_projectors import *
from tn_interface_abelian import contract
try:
    import torch
    from torch.utils.checkpoint import checkpoint
except ImportError as e:
    warnings.warn("torch not available", Warning)

def run(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args): 
    r"""
    :param state: wavefunction
    :param env: environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type conv_check: function(IPEPS_ABELIAN,ENV_ABELIAN,list[float],CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes directional CTM algorithm for generic iPEPS starting from the intial environment ``env``.

    To establish the convergence of CTM before the maximal number of iterations 
    is reached  a ``conv_check`` function is invoked. Its expected signature is 
    ``conv_check(IPEPS_ABELIAN,ENV_ABELIAN,Object,CTMARGS)`` where ``Object`` is an arbitary argument. For 
    example it can be a list or dict used for storing CTM data from previous steps to   
    check convergence.
    """
    #TODO add reference

    # 0) Create double-layer (DL) tensors, preserving the same convention
    #    for order of indices 
    #
    #     /                /(+1)
    #  --a*---   =  (+1)--A--(-1)
    #   /|               /
    #    |/            (-1)
    #  --a--
    #   /
    #
    sitesDL=dict()
    if not state.sites_dl is None:
        sitesDL= state.sites_dl
    else:
        for coord,a in state.sites.items():
            sitesDL[coord]= _fused_dl_site(a)
    dl_peps_args= copy.deepcopy(cfg.peps_args)
    dl_peps_args.build_dl= dl_peps_args.build_dl_open= False
    stateDL = IPEPS_ABELIAN(state.engine, sitesDL, vertexToSite=state.vertexToSite,\
        peps_args=dl_peps_args)

    # 1) perform CTMRG
    t_obs=t_ctm=0.
    history=None
    for i in range(ctm_args.ctm_max_iter):
        t0_ctm= time.perf_counter()
        for direction in ctm_args.ctm_move_sequence:
            diagnostics= {"ctm_i": i, "ctm_d": direction} if ctm_args.verbosity_projectors>0 else None
            num_rows_or_cols= stateDL.lX if direction in [(-1,0),(1,0)] else stateDL.lY
            for row_or_col in range(num_rows_or_cols):
                ctm_MOVE(direction, stateDL, env, ctm_args=ctm_args, global_args=global_args, \
                    verbosity=ctm_args.verbosity_ctm_move, diagnostics=diagnostics)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged, history = conv_check(state, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history[-1])
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG  converged at iter= {i}, history= {history[-1]}")
                break
        t1_obs= time.perf_counter()

        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

# performs CTM move in one of the directions 
# [Up=(0,-1), Left=(-1,0), Down=(0,1), Right=(1,0)]
def ctm_MOVE(direction, state, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args, \
    verbosity=0, diagnostics=None):
    r"""
    :param direction: one of Up=(0,-1), Left=(-1,0), Down=(0,1), Right=(1,0)
    :type direction: tuple(int,int)
    :param state: wavefunction
    :param env: environment
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes a single directional CTM move in one of the directions. First, build  
    projectors for each non-equivalent bond (to be truncated) in the unit cell of iPEPS.
    Second, construct enlarged environment tensors and then truncate them 
    to obtain updated environment tensors.
    """
    # select projector function
    if ctm_args.projector_method=='4X4':
        ctm_get_projectors=ctm_get_projectors_4x4
    elif ctm_args.projector_method=='4X2':
        ctm_get_projectors=ctm_get_projectors_4x2
    else:
        raise ValueError("Invalid Projector method: "+str(ctm_args.projector_method))

    # prepare custom for peps_args for double-layer iPEPS
    dl_peps_args= copy.deepcopy(cfg.peps_args)
    dl_peps_args.build_dl= dl_peps_args.build_dl_open= False

    # 0) compress tensors into 1D representation
    #
    # keep inputs for autograd stored on cpu, move to gpu for the core 
    # of the computation if desired
    metadata_store= {}
    tmp= tuple(state.sites[key].compress_to_1d() for key in state.sites.keys()) \
        + tuple(env.C[key].compress_to_1d() for key in env.C.keys()) \
        + tuple(env.T[key].compress_to_1d() for key in env.T.keys())
    # 1) split into raw 1D tensors and metadata
    tensors, metadata_store["in"]= list(zip(*tmp))

    def move_normalize_c(nC1, nC2, nT, norm_type=ctm_args.ctm_absorb_normalization,\
        verbosity= ctm_args.verbosity_ctm_move):
        assert nC1.size > 0 and nC2.size > 0 and nT.size > 0,"Ill-defined environment"
        if any([nC1.requires_grad, nC2.requires_grad, nT.requires_grad]):
            with torch.no_grad():
                scale_nC1= nC1.norm(p=norm_type)
                scale_nC2= nC2.norm(p=norm_type)
                scale_nT= nT.norm(p=norm_type)
        else:
            scale_nC1= nC1.norm(p=norm_type)
            scale_nC2= nC2.norm(p=norm_type)
            scale_nT= nT.norm(p=norm_type)
        if verbosity>0:
            print(f"nC1 {scale_nC1} nC2 {scale_nC2} nT {scale_nT}")
        nC1 = nC1/scale_nC1
        nC2 = nC2/scale_nC2
        nT = nT/scale_nT
        return nC1, nC2, nT

    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_c(*tensors):
        # 0) reconstruct the tensors from their 1D representation
        _loc_engine= env.engine
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors= tuple(r1d.to(global_args.offload_to_gpu) for r1d in tensors)
            for meta in metadata_store["in"]: 
                meta['config']= meta['config']._replace(default_device=global_args.offload_to_gpu)
            # TODO we assume that configs of all tensors are identical
            _loc_engine= metadata_store["in"][0]["config"]
        
        tensors= tuple(decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(tensors,metadata_store["in"]))

        # 1) wrap raw tensors back into IPEPS and ENV classes 
        sites_loc= dict(zip(state.sites.keys(),tensors[0:len(state.sites)]))
        state_loc= IPEPS_ABELIAN(_loc_engine, sites_loc, state.vertexToSite, peps_args=dl_peps_args)
        env_loc= ENV_ABELIAN(env.chi, settings=_loc_engine)
        env_loc.C= dict(zip(env.C.keys(),tensors[len(state.sites):len(state.sites)+len(env.C)]))
        env_loc.T= dict(zip(env.T.keys(),tensors[len(state.sites)+len(env.C):]))
        # Loop over all non-equivalent sites of ipeps
        # and compute projectors P(coord), P^tilde(coord)

        P = dict()
        Pt = dict()
        for coord,site in state_loc.sites.items():
            if not (diagnostics is None): diagnostics["coord"]= coord 
            P[coord], Pt[coord] = ctm_get_projectors(direction, coord, state_loc, env_loc, \
                ctm_args, global_args, diagnostics=diagnostics)
            if verbosity>0:
                print(f"P,Pt {coord} {direction}") #P: {P[coord].size()} Pt: {Pt[coord].size()}")
            if verbosity>1:
                print(P[coord])
                print(Pt[coord])

        # Loop over all non-equivalent sites of ipeps
        # and perform absorption and truncation
        nC1 = dict()
        nC2 = dict()
        nT = dict()
        for c in state_loc.sites.keys():
            if direction==(0,-1):
                nC1[c], nC2[c], nT[c]= absorb_truncate_CTM_MOVE_UP(c, state_loc, env_loc, P, Pt)
            elif direction==(-1,0):
                nC1[c], nC2[c], nT[c]= absorb_truncate_CTM_MOVE_LEFT(c, state_loc, env_loc, P, Pt)
            elif direction==(0,1):
                nC1[c], nC2[c], nT[c]= absorb_truncate_CTM_MOVE_DOWN(c, state_loc, env_loc, P, Pt)
            elif direction==(1,0):
                nC1[c], nC2[c], nT[c]= absorb_truncate_CTM_MOVE_RIGHT(c, state_loc, env_loc, P, Pt)
            else:
                raise ValueError("Invalid direction: "+str(direction))
            nC1[c], nC2[c], nT[c]= move_normalize_c(nC1[c], nC2[c], nT[c])

        # 2) Return raw new tensors
        tmp_loc= tuple(nC1[key].compress_to_1d() for key in nC1.keys()) \
            + tuple(nC2[key].compress_to_1d() for key in nC2.keys()) \
            + tuple(nT[key].compress_to_1d() for key in nT.keys())
        tensors_loc, metadata_store["out"]= list(zip(*tmp_loc))
        
        # move back to (default) cpu if offload_to_gpu is specified
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors_loc= tuple(r1d.to(global_args.device) for r1d in tensors_loc)
            for meta in metadata_store["out"]:
                meta['config']= meta['config']._replace(default_device=global_args.device)

        return tensors_loc

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_c,*tensors)
    else:
        new_tensors= ctm_MOVE_c(*tensors)
    
    new_tensors= tuple(decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(new_tensors,metadata_store["out"]))

    # 3) warp the returned raw tensor in dictionary
    tmp_coords= state.sites.keys()
    count_coord= len(tmp_coords)
    nC1 = dict(zip(tmp_coords, new_tensors[0:count_coord]))
    nC2 = dict(zip(tmp_coords, new_tensors[count_coord:2*count_coord]))
    nT = dict(zip(tmp_coords, new_tensors[2*count_coord:]))

    # Assign new nC1,nT,nC2 to appropriate environment tensors
    rel_CandT_vecs = dict()
    # specify relative vectors identifying the environment tensors
    # with respect to the direction 
    if direction==(0,-1):
        rel_CandT_vecs = {"nC1": (1,-1), "nC2": (-1,-1), "nT": direction}
    elif direction==(-1,0):
        rel_CandT_vecs = {"nC1": (-1,-1), "nC2": (-1,1), "nT": direction}
    elif direction==(0,1):
        rel_CandT_vecs = {"nC1": (-1,1), "nC2": (1,1), "nT": direction}
    elif direction==(1,0):
        rel_CandT_vecs = {"nC1": (1,1), "nC2": (1,-1), "nT": direction}
    else:
        raise ValueError("Invalid direction: "+str(direction))

    for coord,site in state.sites.items():
        new_coord = state.vertexToSite((coord[0]-direction[0], coord[1]-direction[1]))
        # print("coord: "+str(coord)+" + "+str(direction)+" -> "+str(new_coord))

        env.C[(new_coord,rel_CandT_vecs["nC1"])] = nC1[coord]
        env.C[(new_coord,rel_CandT_vecs["nC2"])] = nC2[coord]
        env.T[(new_coord,rel_CandT_vecs["nT"])] = nT[coord]

#####################################################################
# functions performing absorption and truncation step
#####################################################################
def absorb_truncate_CTM_MOVE_UP(coord, state, env, P, Pt, verbosity=0):
    vec = (1,0)
    coord_shift_left= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    coord_shift_right = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    tensors= env.C[(coord,(1,-1))], env.T[(coord,(1,0))], env.T[(coord,(0,-1))], \
        env.T[(coord,(-1,0))], env.C[(coord,(-1,-1))], state.site(coord), \
        P[coord], Pt[coord], P[coord_shift_right], Pt[coord_shift_right]

    if cfg.ctm_args.fwd_checkpoint_absorb:
        # return checkpoint(absorb_truncate_CTM_MOVE_UP_c,*tensors)
        raise RuntimeError("Checkpointing not implemented")
    else:
        return absorb_truncate_CTM_MOVE_UP_c(*tensors)

def absorb_truncate_CTM_MOVE_UP_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1= tensors
    # Move: UP    (+1)0--P--1(-1)
    #             (-1)0--Pt--1(+1)

    # (-1)0--C1 => (+1)(0<-01)--C1T1
    #        1                  |  
    #        0           (-1)1<-2
    # (+1)1--T1 
    #    (-1)2 
    nC1= contract(C1,T1,([1],[0]))
    nC1= nC1.fuse_legs( axes=((0,1),2) )

    #        --0 0--C1    <=>  (?)1--Pt1--0(-1) (+1)0--C1T1
    #       |       |                                  |
    # 0<-2--Pt1     |                              (-1)1
    #       |       | 
    #        --1 1--T1
    #               2->1
    nC1 = contract(Pt1, nC1,([0],[0]))

    # C2--1->0(-1) => C2T2--(02->0)(-1)
    # 0               |
    # 0               1(-1)
    # T2--2(-1)
    # 1(-1)
    nC2= contract(C2, T2,([0],[0]))
    nC2= nC2.fuse_legs( axes=((0,2),1) )

    # C2--0 0--         <=> C2T2--0(-1) (+1)0--P2--1(-?)
    # |        |            |
    # |        P2--2->1     1(-1)
    # |        |
    # T2--2 1--
    # 1->0
    nC2 = contract(nC2, P2,([0],[0]))
    
    #                                        --0(-1) (+1)0--T--2->3(-1)
    #                       0(-1)           |               1->2(-1)
    # (+1)2<-1--Pt2--0(-1)->|      => 1<-2--Pt2
    #                       1(-1)           |
    #                                        --1->0(-1)
    Pt2= Pt2.unfuse_legs(axes=0)
    nT = contract(Pt2, T, ([0],[0]))

    #            -------T--3->1(-1) =>         ----T-----
    #           |       2                     |    |    |--1(-1)
    # (+1)0<-1--Pt2     |              (+1)0--Pt2--A-----
    #           |       0                          2(-1)
    #            --0 1--A--3(-1)
    #                   2(-1) 
    nT = contract(nT, A,([0,2],[1,0]))
    nT = nT.fuse_legs( axes=(0,(1,3),2) )

    #         -------T---
    #        |       |  | 
    # (+1)0--Pt2     |  |--1(-1)(+1)0--P1--1->2(-1)
    #        |       |  |
    #         -------A---
    #                2->1(-1)
    nT = contract(nT, P1,([1],[0]))

    # Assign new C,T 
    #
    # C(coord,(-1,-1))--                --T(coord,(0,-1))--             --C(coord,(1,-1))
    # |                  P2--       --Pt2 |                P1--     -Pt1  |
    # T(coord,(-1,0))---                --A(coord)---------             --T(coord,(1,0))
    # |                                   |                               |
    #
    # =>                            
    #
    # C^new(coord+(0,1),(-1,-1))--      --T^new(coord+(0,1),(0,-1))--   --C^new(coord+(0,1),(1,-1))
    # |                                   |                               |
    return nC1, nC2, nT


def absorb_truncate_CTM_MOVE_LEFT(coord, state, env, P, Pt, verbosity=0):
    vec = (0,-1)
    coord_shift_up= state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    coord_shift_down= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    tensors = env.C[(coord,(-1,-1))], env.T[(coord,(0,-1))], env.T[(coord,(-1,0))], \
        env.T[(coord,(0,1))], env.C[(coord,(-1,1))], state.site(coord), \
        P[coord], Pt[coord], P[coord_shift_up], Pt[coord_shift_up]

    if cfg.ctm_args.fwd_checkpoint_absorb:
        # return checkpoint(absorb_truncate_CTM_MOVE_LEFT_c,*tensors)
        raise RuntimeError("Checkpointing not implemented")
    else:
        return absorb_truncate_CTM_MOVE_LEFT_c(*tensors)

def absorb_truncate_CTM_MOVE_LEFT_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1= tensors
    # Move: LEFT  (-1)0--P--1(+1)
    #             (+1)0--Pt--1(-1)

    # C1--1 0--T1--2(-1) <=> C1T1--2->1(-1)
    # |        |             (01->0)(-1)
    # 0(-1)    1(-1)
    nC1= contract(C1,T1,([1],[0]))
    nC1= nC1.fuse_legs( axes=((0,1),2) )

    # C1--1 0--T1--2->1 <=> C1T1--1(-1)
    # |        |            0(-1)
    # 0        1            0(+1)
    # 0        1            Pt1
    # |___Pt1__|            1->0(-1) 
    #     2->0
    nC1= contract(Pt1, nC1,([0],[0]))

    # 0        0->1  <=> (01->0)(+1)
    # C2--1 1--T2--2     C2T2--2->1(-1)
    nC2= contract(C2, T2,([1],[1]))
    nC2= nC2.fuse_legs( axes=((0,1),2) )

    #    2->0         <=> 1(+1)
    # ___P2___            P2
    # 0      1            0(-1)
    # 0      1            0(+1)
    # C2-----T2--2->1     C2T2--1(-1)
    nC2 = contract(P2, nC2,([0],[0]))

    #    2->1    <=>  1(+1)    2->1(+1)
    # ___P1__         P1    => P1------
    # 0     1->0      0(-1)    0(-1)  1->0(-1)
    # 0                        0(+1)
    # T--2->3                  T--2->3(-1)
    # 1->2                     1->2(-1)
    P1= P1.unfuse_legs(axes=0)
    nT = contract(P1, T,([0],[0]))

    #    1->0(+1)        =>     0(+1)
    # ___P1______            ___P1____
    # |         0           |         |
    # |         0           T---------A--3->2(-1)
    # T--3 1----A--3(-1)     \-------/
    # 2->1(-1)  2(-1)           1(-1)
    nT= contract(nT, A,([0,3],[0,1]))
    nT= nT.fuse_legs( axes=(0,(1,2),3) )

    #    0            <=>     0(+1)
    # ___P1___             ___P1____
    # |       |           |         |
    # |       |           |         |             0
    # T-------A--3->1     T---------A--2->1(-1) & T--1->2
    # 1       2               1(-1)               2->1
    # 0       1               0(+1) 
    # |___Pt2_|               Pt2
    #     2                   1->2(-1)
    nT = contract(nT, Pt2,([1],[0]))    
    nT = nT.transpose((0,2,1))
    

    # Assign new C,T 
    #
    # C(coord,(-1,-1))--T(coord,(0,-1))-- => C^new(coord+(1,0),(-1,-1))--
    # |________   ______|                    |
    #          Pt1
    #          |
    #
    #          |
    # _________P1______
    # |                |                     |
    # T(coord,(-1,0))--A(coord)--            T^new(coord+(1,0),(-1,0))--
    # |________   _____|                     |
    #          Pt2
    #          |                     
    #          
    #          |
    #  ________P2_______
    # |                 |                    |
    # C(coord,(-1,1))--T(coord,(0,1))--      C^new(coord+(1,0),(-1,1))
    return nC1, nC2, nT


def absorb_truncate_CTM_MOVE_DOWN(coord, state, env, P, Pt, verbosity=0):
    vec = (-1,0)
    coord_shift_right= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    coord_shift_left = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    tensors= env.C[(coord,(-1,1))], env.T[(coord,(-1,0))], env.T[(coord,(0,1))], \
        env.T[(coord,(1,0))], env.C[(coord,(1,1))], state.site(coord), \
        P[coord], Pt[coord], P[coord_shift_left], Pt[coord_shift_left]

    if cfg.ctm_args.fwd_checkpoint_absorb:
        # return checkpoint(absorb_truncate_CTM_MOVE_DOWN_c,*tensors)
        raise RuntimeError("Checkpointing not implemented")
    else:
        return absorb_truncate_CTM_MOVE_DOWN_c(*tensors)

def absorb_truncate_CTM_MOVE_DOWN_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1= tensors
    # Move: DOWN  (-1)0--P--1(+1)
    #             (+1)0--Pt--1(-1)

    # 0->1     <=> 1(+1)
    # T1--2->2     C1T1--(02->0)(-1)
    # 1
    # 0
    # C1--1->0
    nC1 = contract(C1,T1,([0],[1]))
    nC1 = nC1.fuse_legs( axes=((0,2),1) )

    # 1->0               <=> 1->0(+1)
    # T1--2 1--              C1T1--0(-1)(+1)0--Pt1--1(-1)
    # |        |        
    # |        Pt1--2->1
    # |        |
    # C1--0 0--
    nC1 = contract(nC1, Pt1, ([0],[0]))

    #    1<-0  <=>          (+1)1  
    # 2<-1--T2     (+1)(0<-02)--C2T2
    #       2
    #       0
    # 0<-1--C2     
    nC2 = contract(C2, T2,([0],[2]))
    nC2 = nC2.fuse_legs( axes=((0,2),1) )

    #            0<-1  <=>                 (+1)0<-1
    #        --1 2--T2     (+1)1--P2--0(-1)(+1)0--C2T2
    #       |       |
    # 1<-2--P2      |
    #       |       | 
    #        --0 0--C2
    nC2 = contract(nC2, P2, ([0],[0]))

    #        --1->0         <=>                                    --1(-1)
    #       |                                                     |
    # 1<-2--P1                  (+1)1--P1--0(-1) => (+1)2<-1--P1--|  
    #       |       0->2                                          |              0->2(+1)
    #        --0 1--T--2->3                                        --0(-1)(+1)1--T--2->3(-1)
    P1= P1.unfuse_legs(axes=0)
    nT = contract(P1, T, ([0],[1]))

    #                   0->2(+1)    =>           2(+1)
    #            --0 1--A--3(-1)               --A--
    #           |       2                     |  |  |
    # (+1)0<-1--P1      |              (+1)0--P1 |  |--1(-1)
    #           |       2                     |  |  |
    #            -------T--3->1(-1)            --T--
    nT = contract(nT, A,([0,2],[1,2]))
    nT = nT.fuse_legs( axes=(0,(1,3),2) )

    #                2->1(+1)
    #         -------A-- 
    #        |       |  |
    # (+1)0--P1      |  |--1(-1)(+1)0--Pt2--1->2(-1)
    #        |       |  |
    #         -------T--
    nT = contract(nT, Pt2,([1],[0]))
    nT = nT.transpose((1,0,2))
    

    # Assign new C,T
    # 
    # |                                 |                              |
    # T(coord,(-1,0))--               --A(coord)--------             --T(coord,(1,0))
    # |                Pt1--      --P1  |               Pt2--    --P2  |
    # C(coord,(-1,1))--               --T(coord,(0,1))--             --C(coord,(1,1))
    #
    # =>                            
    #
    # |                                 |                              |
    # C^new(coord+(0,-1),(-1,1))--    --T^new(coord+(0,-1),(0,1))--  --C^new(coord+(0,-1),(1,1))
    return nC1, nC2, nT


def absorb_truncate_CTM_MOVE_RIGHT(coord, state, env, P, Pt, verbosity=0):
    vec = (0,1)
    coord_shift_down = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    coord_shift_up = state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    tensors= env.C[(coord,(1,1))], env.T[(coord,(0,1))], env.T[(coord,(1,0))], \
        env.T[(coord,(0,-1))], env.C[(coord,(1,-1))], state.site(coord), \
        P[coord], Pt[coord], P[coord_shift_down], Pt[coord_shift_down]
 
    if cfg.ctm_args.fwd_checkpoint_absorb:
        # return checkpoint(absorb_truncate_CTM_MOVE_RIGHT_c,*tensors)
        raise RuntimeError("Checkpointing not implemented")
    else:
        return absorb_truncate_CTM_MOVE_RIGHT_c(*tensors)

def absorb_truncate_CTM_MOVE_RIGHT_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1= tensors
    # Move: RIGHT  (+1)0--P--1(-1)
    #              (-1)0--Pt--1(+1)

    #       0->1     0        (+1)(0<-01)    
    # 2<-1--T1--2 1--C1 <=> (+1)1<-2--C1T1
    nC1= contract(C1, T1,([1],[2]))
    nC1= nC1.fuse_legs( axes=((0,1),2) ) 

    #          2->0      (+1)0<-1
    #        __Pt1_             Pt1
    #       1     0         (-1)0
    #       1     0         (+1)0  
    # 1<-2--T1----C1 <=> (+1)1--C1T1
    nC1= contract(Pt1, nC1,([0],[0]))

    # 1<-0--T2--2 0--C2 <=>  (+1)1--C2T2
    #    2<-1     0<-1      (-1)(0<-02) 
    nC2= contract(C2,T2,([0],[2]))
    nC2= nC2.fuse_legs( axes=((0,2),1) )

    # 0<-1--T2----C2 <=> (+1)0<-1--C2T2
    #       2     0            (-1)0
    #       1     0            (+1)0
    #       |__P2_|                P2
    #          2->1            (-1)1
    nC2= contract(nC2, P2,([0],[0]))

    #    1<-2     <=> (+1)1   =>    (+1)1<-2
    #    ___Pt2__         Pt2            __Pt2__
    # 0<-1      0     (-1)0      (-1)0<-1   (-1)0
    #           0                           (+1)0
    #     2<-1--T                     (+1)2<-1--T
    #        3<-2                        (-1)3<-2
    Pt2= Pt2.unfuse_legs(axes=0)
    nT= contract(Pt2, T,([0],[0]))

    #       (+1)0<-1        =>        (+1)0  
    #           ___Pt2____             ___Pt2___
    #           0         |    (+1)2--A        T
    #           0         |            \______/ 
    # (+1)2<-1--A--3 2----T           (-1)1
    #    (+1)3<-2  (-1)1<-3 
    nT= contract(nT, A,([0,2],[0,3]))
    nT= nT.fuse_legs( axes=(0,(1,3),2) )

    #          0        <=>          (+1)0
    #       ___Pt2__                  ___Pt2___
    #       |       |       (+1)1<-2--A       T
    #       |       |                  \_____/
    # 1<-2--A-------T                (-1)1 
    #       3       1                (+1)0        
    #       1       0                    P1
    #       |___P1__|             (-1)2<-1
    #           2 
    nT= contract(nT, P1,([1],[0]))
    
    
    # Assign new C,T 
    #
    # --T(coord,(0,-1))--C(coord,(1,-1)) =>--C^new(coord+(-1,0),(1,-1))
    #   |______  ________|                   |
    #          P2
    #          |
    #
    #          |
    #    ______Pt2
    #   |         |                          |
    # --A(coord)--T(coord,(1,0))           --T^new(coord+(-1,0),(1,0))
    #   |______  _|                          |
    #          P1
    #          |                     
    #          
    #          |
    #    ______Pt1______
    #   |               |                    |
    # --T(coord,(0,1))--C(coord,(1,1))     --C^new(coord+(-1,0),(1,1))
    return nC1, nC2, nT
