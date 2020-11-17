import time
import warnings
import config as cfg
from yamps.tensor import decompress_from_1d
from tn_interface_abelian import contract, permute
from ctm.one_site_c4v_abelian.ctm_components_c4v import *
try:
    from torch.utils.checkpoint import checkpoint
except ImportError as e:
    warnings.warn("torch not available", Warning)
import logging
log = logging.getLogger(__name__)
import pdb

def run(state, env, conv_check=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args): 
    r"""
    :param state: wavefunction
    :param env: initial C4v symmetric environment
    :param conv_check: function which determines the convergence of CTM algorithm. If ``None``,
                       the algorithm performs ``ctm_args.ctm_max_iter`` iterations. 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type state: IPEPS_C4V_ABELIAN
    :type env: ENV_C4V_ABELIAN
    :type conv_check: function(IPEPS_C4V_ABELIAN,ENV_C4V_ABELIAN,Object,CTMARGS)->bool
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes specialized CTM algorithm for 1-site C4v symmetric iPEPS starting from the intial 
    environment ``env``. To establish the convergence of CTM before the maximal number of iterations 
    is reached  a ``conv_check`` function is invoked. Its expected signature is 
    ``conv_check(IPEPS,ENV_C4V,Object,CTMARGS)`` where ``Object`` is an arbitary argument. For 
    example it can be a list or dict used for storing CTM data from previous steps to   
    check convergence.
    """

    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method=='GESDD':
        def truncated_decomp(M, chi, sU=1):
            # return truncated_svd_gesdd(M, chi, verbosity=ctm_args.verbosity_projectors)
            return M.split_svd((0,1), tol=ctm_args.projector_svd_reltol, D_total=chi, \
                sU=sU, keep_multiplets=True)
    else:
        raise Exception(f"Projector eig/svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    a= next(iter(state.sites.values()))
    a_dl= contract(a,a, ((0),(0)), conj=(0,1)) # mefgh,mabcd->efghabcd
    a_dl= a_dl.transpose((0,4,1,5,2,6,3,7)) # efghabcd->eafbgchd
    a_dl, lo3= a_dl.group_legs((6,7), new_s=1) # eafbgc(hd->H)->eafbgcH
    a_dl, lo2= a_dl.group_legs((4,5), new_s=1) # eafb(gc->G)H->eafbGH
    a_dl, lo1= a_dl.group_legs((2,3), new_s=1) # ea(fb->F)GH->eaFGH
    a_dl, lo0= a_dl.group_legs((0,1), new_s=1) # (ea->E)F->EFGH
    a_dl._leg_fusion_data= {k: v for k,v in enumerate([lo0, lo1, lo2, lo3])}

    # 1) perform CTMRG
    t_obs=t_ctm=t_fpcm=0.
    history=None
    past_steps_data=dict() # possibly store some data throughout the execution of CTM
    
    for i in range(ctm_args.ctm_max_iter):

        t0_ctm= time.perf_counter()
        ctm_MOVE_dl(a_dl, env, truncated_decomp, ctm_args=ctm_args, global_args=global_args)
        # ctm_MOVE_sl(a, env, truncated_decomp, ctm_args=ctm_args, global_args=global_args,\
        #     past_steps_data=past_steps_data)
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

# performs CTM move
def ctm_MOVE_dl(a_dl, env, f_c2x2_decomp, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    # 0) extract raw tensors as tuple
    metadata_store= {}
    tmp= tuple([a_dl.compress_to_1d(), \
        env.C[env.keyC].compress_to_1d(), env.T[env.keyT].compress_to_1d()])
    metadata_store["in"], tensors= list(zip(*tmp))
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_dl_c(*tensors):
        tensors= tuple(decompress_from_1d(r1d, settings=env.engine, d=meta) \
            for r1d,meta in zip(tensors,metadata_store["in"]))
        A, C, T= tensors

        pdb.set_trace()
        # 1) build enlarged corner upper left corner
        C2X2= c2x2_dl(A, C, T, verbosity=ctm_args.verbosity_projectors)

        # 2) build projector
        # (-1)--M--(-1) = (-1)0--P--1(+1)(-1)--S--(+1)--(-1)0--Vh--1(-1)
        P, S, Vh = f_c2x2_decomp(C2X2, env.chi)

        # (-1)0--P--1(-1) => (+1)0---P--1->2(+1)
        #                    (+1)1--/
        P= P.ungroup_leg(0, C2X2._leg_fusion_data[0])

        # (-1)0--Vh--1(-1) => (-1)0--Vh--1(+1) => (+1)0--Vh--1(-1)
        #                             \--2(+1)            \--2(-1)
        Vh= Vh.ungroup_leg(1, C2X2._leg_fusion_data[1])
 
        # C2X2--1(-1) => C2X2---1(+1) => C2X2   |--1->2(+1)
        # |              |   \--2(+1)    |______|--2->3(+1)  
        # |              |               |     |
        # 0(-1)          0(-1)           0(+1) 1(+1)
        lo0, lo1= C2X2._leg_fusion_data[0], C2X2._leg_fusion_data[1]
        C2X2= C2X2.ungroup_leg(1, lo1)
        C2X2= C2X2.ungroup_leg(0, lo0)

        # 3) absorb and truncate
        #     __
        # C2X2  |--2->1(+1) =>  C2X2---1(+1) (1->-1)0---P--1(1->-1) => C2X2--1(-1)
        # |_____|--3->2(+1)     |   \--2(+1) (1->-1)1--/               |
        # |     |               0(-1)                                  0(-1)
        # 0(+1) 1(+1)
        # 0(-1) 1(-1)
        # |    /         
        # P^h--
        # 2->0(-1)
        C2X2= contract(P.conj(), C2X2, ([0,1],[0,1]))
        C2X2= contract(C2X2, P.change_signature((-1,-1,-1)), ([1,2],[0,1]))
        # C2X2= contract(C2X2, Vh.change_signature((1,-1)).conj(), ([1],[1]))
        
        #        2->1(+1)
        #  ______P___
        # 0(+1->-1)  1->0(+1->-1)
        # 0(+1)
        # T--2->3(-1)
        # 1->2(+1)
        P= P.change_signature((-1,-1,1))
        nT= contract(P, T,([0],[0]))

        #    1->0(+1)
        #  __P____________
        # |               0(-1)
        # |               0(+1)
        # T--3(-1) (+1)1--A--3(+1)
        # 2->1(+1)        2(+1)
        nT= contract(nT, A,([0,3],[0,1]))

        
        # Vh= Vh.change_signature((1,-1,-1))

        #    0(+1)
        #  __P____
        # |       |
        # |       |              0(+1)  
        # T-------A--3->1(+1) => T--1(+1)->2(-1)
        # 1(+1)   2(+1)          2->1(+1)
        # 1(-1)   2(-1)
        # |___Vh__|
        #     0->2(+1)
        # nT= contract(nT, Vh, ([1,2],[1,2]))

        # 0(-1)   1(-1)
        # |___P___|
        #     2(+1)
        nT= contract(nT, P,([1,2],[0,1]))
        
        nT= nT.transpose((0,2,1))
        nT= nT.change_signature((1,1,-1))
        nT._leg_fusion_data[2]= A._leg_fusion_data[3]

        # 4) symmetrize, normalize and assign new C,T
        # C2X2= 0.5*(C2X2 + C2X2.transpose())
        C2X2= 0.5*( C2X2 + C2X2.transpose().conj().change_signature((-1,-1)) )
        # C2X2= C2X2.conj().change_signature((-1,-1))
        nT= 0.5*(nT + nT.transpose((1,0,2)) ) # TODO missing hermitian conjugate
        C2X2= C2X2/S.max_abs()
        nT= nT/nT.max_abs()
        nT._leg_fusion_data[2]= A._leg_fusion_data[3]

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_dl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_dl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

# performs CTM move
def ctm_MOVE_sl(a, env, f_c2x2_decomp, ctm_args=cfg.ctm_args, global_args=cfg.global_args,
    past_steps_data=None):
    r"""
    :param a: on-site C4v symmetric tensor
    :param env: C4v symmetric environment
    :param f_c2x2_decomp: function performing the truncated spectral decomposition (eigenvalue/svd) 
                          of enlarged corner. The ``f_c2x2_decomp`` returns a tuple composed of
                          leading chi spectral values and projector on leading chi spectral values.
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :param past_steps_data: dictionary used for recording diagnostic information during CTM 
    :type a: torch.Tensor
    :type env: ENV_C4V
    :type f_c2x2_decomp: function(torch.Tensor, int)->torch.Tensor, torch.Tensor
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :type past_steps_data:

    Executes a single step of C4v symmetric CTM algorithm for 1-site C4v symmetric iPEPS.
    This variant of CTM step does not explicitly build double-layer on-site tensor.
    """

    # 0) extract raw tensors as tuple
    tensors= tuple([a,env.C[env.keyC],env.T[env.keyT]])
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_sl_c(*tensors):
        a, C, T= tensors
        if global_args.device=='cpu' and ctm_args.step_core_gpu:
            #loc_gpu= torch.device(global_args.gpu)
            a= a.cuda()
            C= C.cuda()
            T= T.cuda()

        # 1) build enlarged corner upper left corner
        C2X2= c2x2_sl(a, C, T, verbosity=ctm_args.verbosity_projectors)

        # 2) build projector
        # P, S, V = f_c2x2_decomp(C2X2, env.chi) # M = PSV^T
        D, P= f_c2x2_decomp(C2X2, env.chi) # M = UDU^T

        # 3) absorb and truncate
        #
        # C2X2--1 0--P--1
        # 0
        # 0
        # P^t
        # 1->0
        # C2X2= P.t() @ C2X2 @ P
        C2X2= torch.diag(D)

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
        # C2X2= C2X2/torch.sum(torch.abs(D))
        nT= nT/torch.max(torch.abs(nT))
        # nT= ((nT.size()[0]*nT.size()[1]*nT.size()[2])/nT.norm())*nT
        # print(f"{nT.norm()}")

        if global_args.device=='cpu' and ctm_args.step_core_gpu:
            C2X2= C2X2.cpu()
            nT= nT.cpu()

        return C2X2, nT

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_sl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_sl_c(*tensors)

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]