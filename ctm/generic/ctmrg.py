import time
import copy
import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
from config import _torch_version_check
from ipeps.ipeps import IPEPS
from ctm.generic.env import *
from ctm.generic.ctm_components import *
from ctm.generic.ctm_projectors import *
from tn_interface import contract, einsum
from tn_interface import conj
from tn_interface import contiguous, view, permute
import logging
log = logging.getLogger(__name__)

def run(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args): 
    r"""
    :param state: wavefunction
    :param env: environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type env: ENV
    :type conv_check: function(IPEPS,ENV,list[float],CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes directional CTM algorithm for generic iPEPS starting from the intial environment ``env``.
    
    To establish the convergence of CTM before the maximal number of iterations 
    is reached  a ``conv_check`` function is invoked. Its expected signature is 
    ``conv_check(IPEPS,ENV,Object,CTMARGS)`` where ``Object`` is an arbitary argument. For 
    example it can be a list or dict used for storing CTM data from previous steps to   
    check convergence.
    """

    # 0) Create double-layer (DL) tensors, preserving the same convention
    # for order of indices 
    #
    #     /           /
    #  --A^dag-- = --a--
    #   /|          /
    #    |/
    #  --A--
    #   /
    #
    if not ctm_args.ctm_force_dl or len(next(iter(state.sites.values())).size())==4:
        stateDL= state
    elif len(next(iter(state.sites.values())).size())==5:
        sitesDL=dict()
        for coord,A in state.sites.items():
            dimsA = A.size()
            a = contiguous(einsum('mefgh,mabcd->eafbgchd',A,conj(A)))
            a= view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
            sitesDL[coord]=a
        stateDL = IPEPS(sites=sitesDL,vertexToSite=state.vertexToSite,lX=state.lX,lY=state.lY,\
            global_args=global_args)

    # 1) perform CTMRG
    t_obs=t_ctm=0.
    history=None
    for i in range(ctm_args.ctm_max_iter):
        t0_ctm= time.perf_counter()
        for direction in ctm_args.ctm_move_sequence:
            diagnostics={"ctm_i": i, "ctm_d": direction} if ctm_args.verbosity_projectors>0 else None
            num_rows_or_cols= stateDL.lX if direction in [(-1,0),(1,0)] else stateDL.lY
            for row_or_col in range(num_rows_or_cols):
                ctm_MOVE(direction, stateDL, env, ctm_args=ctm_args, global_args=global_args, \
                    verbosity=ctm_args.verbosity_ctm_move,diagnostics=diagnostics)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged, history = conv_check(state, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history)
            if converged:
                if ctm_args.verbosity_ctm_convergence>0: 
                    print(f"CTMRG  converged at iter= {i}, history= {history[-1]}")
                break
        t1_obs= time.perf_counter()

        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

def run_overlap(state1, state2, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # r"""
    # :param state: wavefunction
    # :param env: environment
    # :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
    #                    the algorithm performs ``ctm_args.ctm_max_iter`` iterations.
    # :param ctm_args: CTM algorithm configuration
    # :param global_args: global configuration
    # :type state: IPEPS
    # :type env: ENV
    # :type conv_check: function(IPEPS,ENV,list[float],CTMARGS)->bool
    # :type ctm_args: CTMARGS
    # :type global_args: GLOBALARGS

    # Executes directional CTM algorithm for generic iPEPS starting from the intial environment ``env``.
    # TODO add reference
    # """

    assert ctm_args.ctm_force_dl,'ctmrg for wavefunction overlap requires use of double-layer routines'
    # 0) Create double-layer (DL) tensors, preserving the same convention
    # for order of indices
    #
    #     /           /
    #  --A1^dag-- = --a--
    #   /|          /
    #    |/
    #  --A2--
    #   /
    #
    sitesDL=dict()
    for coord,site in state1.sites.items():
        A1 = state1.site((coord[0], coord[1]))
        A2 = state2.site((coord[0], coord[1]))
        dimsA = A1.size()
        a = contiguous(einsum('mefgh,mabcd->eafbgchd',A1,conj(A2)))
        a = view(a, (dimsA[1]**2,dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
        sitesDL[coord]=a
    stateDL = IPEPS(sitesDL,state1.vertexToSite)

    # 1) perform CTMRG
    t_obs=t_ctm=0.
    history=None
    for i in range(ctm_args.ctm_max_iter):
        t0_ctm= time.perf_counter()
        for direction in ctm_args.ctm_move_sequence:
            ctm_MOVE(direction, stateDL, env, ctm_args=ctm_args, global_args=global_args, \
                verbosity=ctm_args.verbosity_ctm_move)
        t1_ctm= time.perf_counter()

        t0_obs= time.perf_counter()
        if conv_check is not None:
            # evaluate convergence of the CTMRG procedure
            converged, history = conv_check(state1, state2, env, history, ctm_args=ctm_args)
            if ctm_args.verbosity_ctm_convergence>1: print(history)
            if converged:
                if ctm_args.verbosity_ctm_convergence>0:
                    print(f"CTMRG  converged at iter= {i}, history= {history[-1]}")
                break
        t1_obs= time.perf_counter()

        t_ctm+= t1_ctm-t0_ctm
        t_obs+= t1_obs-t0_obs

    return env, history, t_ctm, t_obs

# performs 
# 
def ctm_MOVE(direction, state, env, ctm_args=cfg.ctm_args, global_args=cfg.global_args, \
    verbosity=0, diagnostics=None):
    r"""
    :param direction: one of Up=(0,-1), Left=(-1,0), Down=(0,1), Right=(1,0)
    :type direction: tuple(int,int)
    :param state: wavefunction
    :param env: environment
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS
    :type env: ENV
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

    # 0) extract raw tensors as tuple
    tensors= tuple(state.sites[key] for key in state.sites.keys()) \
        + tuple(env.C[key] for key in env.C.keys()) + tuple(env.T[key] for key in env.T.keys())

    def move_normalize_c(nC1, nC2, nT, norm_type=ctm_args.ctm_absorb_normalization,\
        verbosity= ctm_args.verbosity_ctm_move):
        _ord=2
        if norm_type=='inf':
            _ord= float('inf')

        with torch.no_grad():
            if _torch_version_check("1.9.0"):
                scale_nC1= torch.linalg.vector_norm(nC1,ord=_ord)
                scale_nC2= torch.linalg.vector_norm(nC2,ord=_ord)
                scale_nT= torch.linalg.vector_norm(nT,ord=_ord)
            else:
                scale_nC1= nC1.norm(p=_ord)
                scale_nC2= nC2.norm(p=_ord)
                scale_nT= nT.norm(p=_ord)
        if verbosity>0:
            print(f"nC1 {scale_nC1} nC2 {scale_nC2} nT {scale_nT}")
        nC1 = nC1/scale_nC1
        nC2 = nC2/scale_nC2
        nT = nT/scale_nT
        return nC1, nC2, nT

    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_c(*tensors):
        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            tensors= tuple( t.to(global_args.offload_to_gpu) for t in tensors )

        # 1) wrap raw tensors back into IPEPS and ENV classes
        sites_loc= dict(zip(state.sites.keys(),tensors[0:len(state.sites)]))
        state_loc= IPEPS(sites_loc, vertexToSite=state.vertexToSite, lX=state.lX, lY=state.lY)
        env_loc= ENV(env.chi)
        env_loc.C= dict(zip(env.C.keys(),tensors[len(state.sites):len(state.sites)+len(env.C)]))
        env_loc.T= dict(zip(env.T.keys(),tensors[len(state.sites)+len(env.C):]))
        # Loop over all non-equivalent sites of ipeps
        # and compute projectors P(coord), P^tilde(coord)

        P = dict()
        Pt = dict()
        for coord,site in state_loc.sites.items():
            # TODO compute isometries
            if not (diagnostics is None): diagnostics["coord"]= coord
            P[coord], Pt[coord] = ctm_get_projectors(direction, coord, state_loc, env_loc,\
                ctm_args, global_args, diagnostics=diagnostics)
            if verbosity>0:
                log.info("P,Pt RIGHT "+str(coord)+" P: "+str(P[coord].size())+" Pt: "+str(Pt[coord].size()))
            if verbosity>1:
                print(P[coord])
                print(Pt[coord])

        # Loop over all non-equivalent sites of ipeps
        # and perform absorption and truncation
        nC1 = dict()
        nC2 = dict()
        nT = dict()
        for coord in state_loc.sites.keys():
            if direction==(0,-1):
                nC1[coord], nC2[coord], nT[coord] = absorb_truncate_CTM_MOVE_UP(coord, state_loc, env_loc, P, Pt, ctm_args)
            elif direction==(-1,0):
                nC1[coord], nC2[coord], nT[coord] = absorb_truncate_CTM_MOVE_LEFT(coord, state_loc, env_loc, P, Pt, ctm_args)
            elif direction==(0,1):
                nC1[coord], nC2[coord], nT[coord] = absorb_truncate_CTM_MOVE_DOWN(coord, state_loc, env_loc, P, Pt, ctm_args)
            elif direction==(1,0):
                nC1[coord], nC2[coord], nT[coord] = absorb_truncate_CTM_MOVE_RIGHT(coord, state_loc, env_loc, P, Pt, ctm_args)
            else:
                raise ValueError("Invalid direction: "+str(direction))
            nC1[coord], nC2[coord], nT[coord]= move_normalize_c(nC1[coord], nC2[coord], nT[coord])

        # 2) Return raw new tensors
        ret_list= tuple(nC1[key] for key in nC1.keys()) + tuple(nC2[key] for key in nC2.keys()) \
            + tuple(nT[key] for key in nT.keys())
        if global_args.device=='cpu' and global_args.offload_to_gpu != 'None':
            ret_list= tuple( t.to(global_args.device) for t in ret_list )

        return ret_list

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_c,*tensors)
    else:
        new_tensors= ctm_MOVE_c(*tensors)
    
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
def absorb_truncate_CTM_MOVE_UP(coord, state, env, P, Pt, ctm_args=cfg.ctm_args):
    mode= not ctm_args.ctm_force_dl
    vec = (1,0)
    coord_shift_left= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    coord_shift_right = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    tensors= env.C[(coord,(1,-1))], env.T[(coord,(1,0))], env.T[(coord,(0,-1))], \
        env.T[(coord,(-1,0))], env.C[(coord,(-1,-1))], state.site(coord), \
        view(P[coord], (env.chi,state.site(coord_shift_left).size(3+mode)**(mode+1),env.chi)), \
        view(Pt[coord], (env.chi,state.site(coord).size(1+mode)**(mode+1),env.chi)), \
        view(P[coord_shift_right], (env.chi,state.site(coord).size(3+mode)**(mode+1),env.chi)), \
        view(Pt[coord_shift_right], (env.chi,state.site(coord_shift_right).size(1+mode)**(mode+1),env.chi))
    if mode:
        tensors += (torch.ones(1,dtype=torch.bool),)

    if cfg.ctm_args.fwd_checkpoint_absorb:
        return checkpoint(absorb_truncate_CTM_MOVE_UP_c,*tensors)
    else:
        return absorb_truncate_CTM_MOVE_UP_c(*tensors)

def absorb_truncate_CTM_MOVE_UP_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1, mode= tensors if len(tensors)==11 else tensors+(None,)

    # 0--C1
    #    1
    #    0
    # 1--T1
    #    2 
    nC1 = contract(C1,T1,([1],[0]))

    #        --0 0--C1
    #       |       |
    # 0<-2--Pt1     |
    #       |       | 
    #        --1 1--T1
    #               2->1
    nC1 = contract(Pt1, nC1,([0,1],[0,1]))

    # C2--1->0
    # 0
    # 0
    # T2--2
    # 1
    nC2 = contract(C2, T2,([0],[0])) 

    # C2--0 0--
    # |        |        
    # |        P2--2->1
    # |        |
    # T2--2 1--
    # 1->0
    nC2 = contract(nC2, P2,([0,2],[0,1]))

    if not mode:
        #        --0 0--T--2->3
        #       |       1->2
        # 1<-2--Pt2
        #       |
        #        --1->0 
        nT = contract(Pt2, T, ([0],[0]))

        #        -------T--3->1
        #       |       2
        # 0<-1--Pt2     | 
        #       |       0
        #        --0 1--A--3
        #               2 
        nT = contract(nT, A,([0,2],[1,0]))

        #     -------T--1 0--
        #    |       |       |
        # 0--Pt2     |       P1--2
        #    |       |       |
        #     -------A--3 1--
        #            2->1 
        nT = contract(nT, P1,([1,3],[0,1]))
        nT = contiguous(nT)
    else:
        # unfuse aux inds connecting to on-site tensor
        #
        # 0--T--2
        #    1
        T= T.view([T.size(0)]+[A.size(1)]*2+[T.size(2)])
        #
        #      /--0
        # 2--Pt2--1
        Pt2= Pt2.view([Pt2.size(0)]+[A.size(2)]*2+[Pt2.size(2)])
        #      0--\
        #      1--P1--2
        P1= P1.view([P1.size(0)]+[A.size(4)]*2+[P1.size(2)])

        #        --(0)0 0--T----3 3(0)------
        #       |          1, 2             |
        # (3)4--Pt2        1  2             P1--7(3)
        #       |          |  |             |
        #        --(1)8 8--a--|---10 10(1)--
        #       |           12|             |
        #        --(2)9 9-----a*--11 11(2)-- 
        #                  |  | 
        #                  5  6
        nT= torch.einsum(T,[0,1,2,3],Pt2,[0,8,9,4],A,[12,1,8,5,10],A.conj(),[12,2,9,6,11],\
            P1,[3,10,11,7],[4,5,6,7])
        nT= nT.view(nT.size(0),nT.size(1)*nT.size(2),nT.size(3))

    # Assign new C,T 
    #
    # C(coord,(-1,-1))--                --T(coord,(0,-1))--             --C(coord,(1,-1))
    # |                  P2--       --Pt2 |                P1--      -Pt1  |
    # T(coord,(-1,0))---                --A(coord)---------             --T(coord,(1,0))
    # |                                   |                               |
    #
    # =>                            
    #
    # C^new(coord+(0,1),(-1,-1))--      --T^new(coord+(0,1),(0,-1))--   --C^new(coord+(0,1),(1,-1))
    # |                                   |                               |
    return nC1, nC2, nT

def absorb_truncate_CTM_MOVE_LEFT(coord, state, env, P, Pt, ctm_args=cfg.ctm_args):
    mode= not ctm_args.ctm_force_dl
    vec = (0,-1)
    coord_shift_up= state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    coord_shift_down= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    tensors = env.C[(coord,(-1,-1))], env.T[(coord,(0,-1))], env.T[(coord,(-1,0))], \
        env.T[(coord,(0,1))], env.C[(coord,(-1,1))], state.site(coord), \
        view(P[coord], (env.chi,state.site(coord_shift_down).size(0+mode)**(mode+1),env.chi)), \
        view(Pt[coord], (env.chi,state.site(coord).size(2+mode)**(mode+1),env.chi)), \
        view(P[coord_shift_up], (env.chi,state.site(coord).size(0+mode)**(mode+1),env.chi)), \
        view(Pt[coord_shift_up], (env.chi,state.site(coord_shift_up).size(2+mode)**(mode+1),env.chi))
    if mode:
        tensors += (torch.ones(1,dtype=torch.bool),)

    if cfg.ctm_args.fwd_checkpoint_absorb:
        return checkpoint(absorb_truncate_CTM_MOVE_LEFT_c,*tensors)
    else:
        return absorb_truncate_CTM_MOVE_LEFT_c(*tensors)

def absorb_truncate_CTM_MOVE_LEFT_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1, mode= tensors if len(tensors)==11 else tensors+(None,)

    # C1--1 0--T1--2
    # |        |
    # 0        1
    nC1 = contract(C1,T1,([1],[0]))

    # C1--1 0--T1--2->1
    # |        |
    # 0        1
    # 0        1
    # |___Pt1__|
    #     2->0
    nC1 = contract(Pt1, nC1,([0,1],[0,1]))

    # 0        0->1
    # C2--1 1--T2--2
    nC2 = contract(C2, T2,([1],[1])) 

    #    2->0
    # ___P2___
    # 0      1
    # 0      1  
    # C2-----T2--2->1
    nC2 = contract(P2, nC2,([0,1],[0,1]))

    if not mode:
        #    2->1
        # ___P1__
        # 0     1->0
        # 0
        # T--2->3
        # 1->2
        nT = contract(P1, T,([0],[0]))

        #    1->0
        # ___P1____
        # |       0
        # |       0
        # T--3 1--A--3
        # 2->1    2
        nT = contract(nT, A,([0,3],[0,1]))

        #    0
        # ___P1___
        # |       |
        # |       |
        # T-------A--3->1=>2
        # 1       2
        # 0       1
        # |___Pt2_|
        #     2=>1
        nT = contract(nT, Pt2,([1,2],[0,1]))
        nT = contiguous(permute(nT, (0,2,1)))
    else:
        # unfuse aux inds connecting to on-site tensor
        # 0
        # T--2
        # 1
        T= T.view(list(T.size()[:2])+[A.size(2)]*2)
        #
        #   2
        # --P1--
        # 0    1
        P1= P1.view([P1.size(0)]+[A.size(1)]*2+[P1.size(2)])
        # 0     1
        # --Pt2--
        #   2
        Pt2= Pt2.view([Pt2.size(0)]+[A.size(3)]*2+[Pt2.size(2)])

        #   9(3)
        # --P1-------------
        # 0           4(1) 5(2)
        # 0           4    5
        # T---2 2-----a----|-----10
        # |  \        | 8  |
        # 1   3 3----------a*----11
        # 1(0)        6    7
        # |           6(1) 7(2)     
        # --Pt2-------------
        #   12(3) 
        nT= torch.einsum(T,[0,1,2,3],Pt2,[1,6,7,12],A,[8,4,2,6,10],A.conj(),[8,5,3,7,11],\
            P1,[0,4,5,9],[9,12,10,11])
        nT= nT.view(nT.size(0), nT.size(1), nT.size(2)*nT.size(3))

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

def absorb_truncate_CTM_MOVE_DOWN(coord, state, env, P, Pt, ctm_args=cfg.ctm_args):
    mode= not ctm_args.ctm_force_dl
    vec = (-1,0)
    coord_shift_right= state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    coord_shift_left = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    tensors= env.C[(coord,(-1,1))], env.T[(coord,(-1,0))], env.T[(coord,(0,1))], \
        env.T[(coord,(1,0))], env.C[(coord,(1,1))], state.site(coord), \
        view(P[coord], (env.chi,state.site(coord_shift_right).size(1+mode)**(mode+1),env.chi)), \
        view(Pt[coord], (env.chi,state.site(coord).size(3+mode)**(mode+1),env.chi)), \
        view(P[coord_shift_left], (env.chi,state.site(coord).size(1+mode)**(mode+1),env.chi)), \
        view(Pt[coord_shift_left], (env.chi,state.site(coord_shift_left).size(3+mode)**(mode+1),env.chi))
    if mode:
        tensors += (torch.ones(1,dtype=torch.bool),)

    if cfg.ctm_args.fwd_checkpoint_absorb:
        return checkpoint(absorb_truncate_CTM_MOVE_DOWN_c,*tensors)
    else:
        return absorb_truncate_CTM_MOVE_DOWN_c(*tensors)

def absorb_truncate_CTM_MOVE_DOWN_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1, mode= tensors if len(tensors)==11 else tensors+(None,)

    # 0->1
    # T1--2->2
    # 1
    # 0
    # C1--1->0
    nC1 = contract(C1,T1,([0],[1]))

    # 1->0
    # T1--2 1--
    # |        |        
    # |        Pt1--2->1
    # |        |
    # C1--0 0--   
    nC1 = contract(nC1, Pt1, ([0,2],[0,1]))

    #    1<-0
    # 2<-1--T2
    #       2
    #       0
    # 0<-1--C2
    nC2 = contract(C2, T2,([0],[2])) 

    #            0<-1
    #        --1 2--T2
    #       |       |
    # 1<-2--P2      |
    #       |       | 
    #        --0 0--C2
    nC2 = contract(nC2, P2, ([0,2],[0,1]))

    if not mode:
        #        --1->0
        #       |
        # 1<-2--P1
        #       |       0->2
        #        --0 1--T--2->3 
        nT = contract(P1, T, ([0],[1]))

        #               0->2
        #        --0 1--A--3 
        #       |       2 
        # 0<-1--P1      |
        #       |       2
        #        -------T--3->1
        nT = contract(nT, A,([0,2],[1,2]))

        #               2->1=>0
        #        -------A--3 1--
        #       |       |       |
        # 1<=0--P1      |       Pt2--2
        #       |       |       |
        #        -------T--1 0--
        nT = contract(nT, Pt2,([1,3],[0,1]))
        nT = contiguous(permute(nT, (1,0,2)))
    else:
        # unfuse aux inds connecting to on-site tensor
        #  
        #    0
        # 1--T--2
        T= T.view([A.size(3)]*2+list(T.size()[1:]))
        #
        #     /--1
        # 2--P1--0
        P1= P1.view([P1.size(0)]+[A.size(2)]*2+[P1.size(2)])
        #      1--\
        #      0--Pt2--2
        Pt2= Pt2.view([Pt2.size(0)]+[A.size(4)]*2+[Pt2.size(2)])

        #                  5  6
        #        --(1)8 8--a--|---10 10(1)--
        #       |           12|             |
        #        --(2)9 9-----a*--11 11(2)-- 
        #       |          |  |             |
        #       |          0  1             |
        # (3)4--P1         0  1             Pt2--7(3)
        #       |          |, |             |
        #        --(0)2 2--T----3 3(0)------
        nT= torch.einsum(T,[0,1,2,3],Pt2,[3,10,11,7],A,[12,5,8,0,10],A.conj(),[12,6,9,1,11],\
            P1,[2,8,9,4],[5,6,4,7])
        nT= nT.view(nT.size(0)*nT.size(1),nT.size(2),nT.size(3))

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

def absorb_truncate_CTM_MOVE_RIGHT(coord, state, env, P, Pt, ctm_args=cfg.ctm_args):
    mode= not ctm_args.ctm_force_dl
    vec = (0,1)
    coord_shift_down = state.vertexToSite((coord[0]+vec[0], coord[1]+vec[1]))
    coord_shift_up = state.vertexToSite((coord[0]-vec[0], coord[1]-vec[1]))
    tensors= env.C[(coord,(1,1))], env.T[(coord,(0,1))], env.T[(coord,(1,0))], \
        env.T[(coord,(0,-1))], env.C[(coord,(1,-1))], state.site(coord), \
        view(P[coord], (env.chi,state.site(coord_shift_up).size(2+mode)**(mode+1),env.chi)), \
        view(Pt[coord], (env.chi,state.site(coord).size(0+mode)**(mode+1),env.chi)), \
        view(P[coord_shift_down], (env.chi,state.site(coord).size(2+mode)**(mode+1),env.chi)), \
        view(Pt[coord_shift_down], (env.chi,state.site(coord_shift_down).size(0+mode)**(mode+1),env.chi))
    if mode:
        tensors += (torch.ones(1,dtype=torch.bool),)

    if cfg.ctm_args.fwd_checkpoint_absorb:
        return checkpoint(absorb_truncate_CTM_MOVE_RIGHT_c,*tensors)
    else:
        return absorb_truncate_CTM_MOVE_RIGHT_c(*tensors)

def absorb_truncate_CTM_MOVE_RIGHT_c(*tensors):
    C1, T1, T, T2, C2, A, P2, Pt2, P1, Pt1, mode= tensors if len(tensors)==11 else tensors+(None,)

    #       0->1     0
    # 2<-1--T1--2 1--C1
    nC1 = contract(C1, T1,([1],[2])) 

    #          2->0
    #        __Pt1_
    #       1     0
    #       1     0
    # 1<-2--T1----C1
    nC1 = contract(Pt1, nC1,([0,1],[0,1]))

    # 1<-0--T2--2 0--C2
    #    2<-1     0<-1
    nC2 = contract(C2,T2,([0],[2]))

    # 0<-1--T2----C2
    #       2     0
    #       1     0
    #       |__P2_|
    #          2->1
    nC2 = contract(nC2, P2,([0,2],[0,1]))

    if not mode:
        #    1<-2
        #    ___Pt2__
        # 0<-1      0
        #           0
        #     2<-1--T
        #        3<-2
        nT = contract(Pt2, T,([0],[0]))

        #       0<-1 
        #       ___Pt2__
        #       0       |
        #       0       |
        # 2<-1--A--3 2--T
        #    3<-2    1<-3
        nT = contract(nT, A,([0,2],[0,3]))

        #          0
        #       ___Pt2__
        #       |       |
        #       |       |
        # 1<-2--A-------T
        #       3       1
        #       1       0
        #       |___P1__|
        #           2 
        nT = contract(nT, P1,([1,3],[0,1]))
        nT = contiguous(nT)
    else:
        # unfuse aux inds connecting to on-site tensor
        #    0
        # 1--T
        #    2
        T= T.view([T.size(0)]+[A.size(2)]*2+[T.size(2)])
        #
        #   2
        # --Pt2--
        # 1     0
        Pt2= Pt2.view([Pt2.size(0)]+[A.size(1)]*2+[Pt2.size(2)])
        # 1     0
        # --P1--
        #   2
        P1= P1.view([P1.size(0)]+[A.size(3)]*2+[P1.size(2)])

        #                        9(3)
        #         ---------------Pt2----
        #         4(1) 5(2)            0
        #         4    5               0
        #     10--a----|---------1 1---T
        #         | 8  |             / |
        #     11------a*---------2 2   3
        #         6    7               3(0)
        #         6(1) 7(2)            |
        #         ---------------P1-----
        #                        12(3) 
        nT= torch.einsum(T,[0,1,2,3],Pt2,[0,4,5,9],A,[8,4,10,6,1],A.conj(),[8,5,11,7,2],\
            P1,[3,6,7,12],[9,10,11,12])
        nT= nT.view(nT.size(0), nT.size(1)*nT.size(2), nT.size(3))
    
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
