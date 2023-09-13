import time
import warnings
from types import SimpleNamespace
import config as cfg
# from yamps.tensor import decompress_from_1d
import yastn.yastn as yastn
from tn_interface_abelian import contract, permute
from ctm.one_site_c4v_abelian.ctm_components_c4v import *
try:
    from torch.utils.checkpoint import checkpoint
except ImportError as e:
    warnings.warn("torch not available", Warning)
import logging
log = logging.getLogger(__name__)


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

    Executes specialized CTM algorithm for abelian-symmetric 1-site C4v symmetric iPEPS starting 
    from the intial environment ``env``. 

    To establish the convergence of CTM before 
    the maximal number of iterations is reached  a ``conv_check`` function is invoked. 
    Its expected signature is ``conv_check(IPEPS_ABELIAN_C4V,ENV_ABELIAN_C4V,Object,CTMARGS)``
    where ``Object`` is an arbitary argument. For example it can be a list or dict used 
    for storing CTM data from previous steps to check convergence.

    If :attr:`config.CTMARGS.ctm_force_dl`, then double-layer tensor is precomputed
    and used in the CTMRG. Otherwise, `ket` and `bra` layer of on-site tensor are contracted
    sequentially when building enlarged corner. 
    """

    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method=='GESDD':
        def truncation_f(S):
            return yastn.linalg.truncation_mask_multiplets(S,keep_multiplets=True, D_total=env.chi,\
                tol=ctm_args.projector_svd_reltol, tol_block=ctm_args.projector_svd_reltol_block, \
                eps_multiplet=ctm_args.projector_eps_multiplet)
        def truncated_decomp(M, chi, legs=(0,1), sU=1, diagnostics=None):
            return yastn.linalg.svd_with_truncation(M, legs, sU=sU, mask_f=truncation_f, diagnostics=diagnostics)
    # elif ctm_args.projector_svd_method=='SYMARP':
    #     def truncated_decomp(M, chi, legs=(0,1), sU=1):
    #         return M.split_svd_eigsh_2C(axes=legs, tol=ctm_args.projector_svd_reltol, D_total=chi, \
    #             sU=sU, keep_multiplets=True)
    else:
        raise Exception(f"Projector eig/svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    a= state.site()
    if ctm_args.ctm_force_dl:
        # 0) Create double-layer (DL) tensors, preserving the same convention
        #    for order of indices 
        #
        #     /               /(+1)
        #  --a---   =  (+1)--A--(+1)
        #   /|              /
        #    |/           (+1)
        #  --a*--
        #   /
        #
        a_dl= contract(a,a, ((0),(0)), conj=(0,1)) # mefgh,mabcd->efghabcd; efghabcd->eafbgchd
        a_dl= a_dl.fuse_legs( axes=((0,4),(1,5),(2,6),(3,7)) )

    # 1) perform CTMRG
    t_obs=t_ctm=t_fpcm=0.
    history=None
    past_steps_data=dict() # possibly store some data throughout the execution of CTM
    
    for i in range(ctm_args.ctm_max_iter):

        t0_ctm= time.perf_counter()
        if ctm_args.ctm_force_dl:
            ctm_MOVE_dl(a_dl, env, truncated_decomp, ctm_args=ctm_args, global_args=global_args)
        else:
            ctm_MOVE_sl(a, env, truncated_decomp, ctm_args=ctm_args, global_args=global_args)
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
    r"""
    :param a_dl: double-layer on-site C4v symmetric tensor
    :param env: C4v symmetric environment
    :param f_c2x2_decomp: function performing the truncated spectral decomposition (eigenvalue/svd) 
                          of enlarged corner. The ``f_c2x2_decomp`` returns a tuple composed of
                          leading chi spectral values and projector on leading chi spectral values.
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type a_dl: yastn.Tensor
    :type env: ENV_C4V_ABELIAN
    :type f_c2x2_decomp: function(yastn.Tensor, int)->yastn.Tensor, yastn.Tensor, yastn.Tensor
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes a single step of C4v symmetric CTM algorithm for 1-site C4v symmetric iPEPS.
    This variant of CTM step uses pre-built double-layer on-site tensor. 
    """

    # 0) compress abelian symmetric tensor into 1D representation for the purposes of 
    #    checkpointing
    metadata_store= {}
    tmp= tuple([a_dl.compress_to_1d(), env.C[env.keyC].compress_to_1d(), \
        env.T[env.keyT].compress_to_1d()])
    tensors, metadata_store["in"]= list(zip(*tmp))
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_dl_c(*tensors):
        #
        # keep inputs for autograd stored on cpu, move to gpu for the core 
        # of the computation if desired
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors= tuple(r1d.to(global_args.offload_to_gpu) for r1d in tensors)
            for meta in metadata_store["in"]: 
                meta['config']= meta['config']._replace(default_device=global_args.offload_to_gpu)

        tensors= tuple(yastn.decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(tensors,metadata_store["in"]))
        A, C, T= tensors
        
        # Fuse on-site tensor indices into single index  
        # 0(+)
        # T--2(-),3(+)->2(-)
        # 1(+)
        T= T.fuse_legs(axes=(0,1,(2,3)))

        # 1) build enlarged corner upper left corner
        C2X2= c2x2_dl(A, C, T, verbosity=ctm_args.verbosity_projectors)
        # import pdb; pdb.set_trace()

        # 2) build projector
        # (+1)--M--(+1) = (+1)0--P--1(+1)(-1)--S--(+1)--(-1)0--Vh--1(+1)
        P, S, Vh = f_c2x2_decomp(C2X2, env.chi)

        # 3) absorb and truncate
        #     __
        # C2X2  |--1(+1) =>  C2X2---1(+1) (1->-1)0---P--1(1->-1) => C2X2--1(-1)
        # |_____|            |                                      |
        # |                  0(-1)                                  0(-1)
        # 0(+1)
        # 0(-1)
        # |         
        # P*--
        # 1->0(-1)
        # NOTE change signature <=> hermitian-conj since C2X2= UDU^\dag where U=U^\dag ?
        C2X2= contract(P.conj(), C2X2, ([0],[0]))
        C2X2= contract(C2X2, P.flip_signature(), ([1],[0]))

        #        2->1(+1->-1)
        #  ______P___
        # 0(+1->-1)  1->0(+1->-1)
        # 0(+1)
        # T--2->3(-1)
        # 1->2(+1)
        P= P.unfuse_legs(axes=0)
        nT= contract(P.flip_signature(), T,([0],[0]))

        #    1->0(-1)
        #  __P____________
        # |               0(-1)
        # |               0(+1)
        # T--3(-1) (+1)1--A--3(+1)
        # 2->1(+1)        2(+1)
        # TODO is it neccessary to "ungroup" the index connecting T to the site ?
        nT= contract(nT, A,([0,3],[0,1]))

        #    0(-1)
        #  __P____
        # |       |
        # |       |              0(-1)  
        # T-------A--3->1(+1) => T--1(+1)
        # 1(+1)   2(+1)          2->1(-1)
        # 0(-1)   1(-1)
        # |___P*__|
        #     2(-1)
        # TODO do we conjugate here ?
        nT= contract(nT, P.conj(),([1,2],[0,1]))
        nT= nT.transpose((0,2,1)).flip_signature()

        # 4) symmetrize, normalize and assign new C,T
        C2X2= 0.5*( C2X2 + C2X2.transpose((1,0)).conj_blocks() )
        nT= 0.5*(nT + nT.transpose((1,0,2)).conj_blocks() )
        C2X2= C2X2/S.norm(p='inf')
        nT= nT/nT.norm(p='inf')

        # 0(-)
        # T--2(+)->2(-),3(+)
        # 1(-)
        nT= nT.unfuse_legs(axes=2)

        # 2) Return raw new tensors
        tmp_loc= tuple([C2X2.compress_to_1d(), nT.compress_to_1d()])
        tensors_loc, metadata_store["out"]= list(zip(*tmp_loc))

        # move back to (default) cpu if offload_to_gpu is specified
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors_loc= tuple(r1d.to(global_args.device) for r1d in tensors_loc)
            for meta in metadata_store["out"]: 
                meta['config']= meta['config']._replace(default_device=global_args.device)
        
        return tensors_loc

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_dl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_dl_c(*tensors)

    new_tensors= tuple(yastn.decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(new_tensors,metadata_store["out"]))

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]

# performs CTM move
def ctm_MOVE_sl(a, env, f_c2x2_decomp, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
    r"""
    :param a: on-site C4v symmetric tensor
    :param env: C4v symmetric environment
    :param f_c2x2_decomp: function performing the truncated spectral decomposition (eigenvalue/svd) 
                          of enlarged corner. The ``f_c2x2_decomp`` returns a tuple composed of
                          leading chi spectral values and projector on leading chi spectral values.
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type a: yastn.Tensor
    :type env: ENV_ABELIAN_C4V
    :type f_c2x2_decomp: function(yastn.Tensor, int)->yastn.Tensor, yastn.Tensor, yastn.Tensor
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS

    Executes a single step of C4v symmetric CTM algorithm for 1-site C4v symmetric iPEPS.
    This variant of CTM step does not explicitly build double-layer on-site tensor.
    """

    # 0) compress abelian symmetric tensor into 1D representation for the purposes of 
    #    checkpointing
    #
    # keep inputs for autograd stored on cpu, move to gpu for the core 
    # of the computation if desired
    metadata_store= {}
    tmp= tuple([a.compress_to_1d(), \
            env.C[env.keyC].compress_to_1d(), env.T[env.keyT].compress_to_1d()])
    tensors, metadata_store["in"] = list(zip(*tmp))
    
    # function wrapping up the core of the CTM MOVE segment of CTM algorithm
    def ctm_MOVE_sl_c(*tensors):
        #
        # keep inputs for autograd stored on cpu, move to gpu for the core 
        # of the computation if desired
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors= tuple(r1d.to(global_args.offload_to_gpu) for r1d in tensors)
            for meta in metadata_store["in"]: 
                meta['config']= meta['config']._replace(default_device=global_args.offload_to_gpu)
                
        a,C,T= tuple(yastn.decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(tensors,metadata_store["in"]))

        # 1) build enlarged corner upper left corner
        # C----T---3(+)
        # |    |
        # T---a*a--4(+),5(-)
        # |    |
        # 0(+) 1(+),2(-)
        C2X2= c2x2_sl(a, C, T, verbosity=ctm_args.verbosity_projectors)

        # 2) build projector
        # (+)0--C2X2--(+)3  = (+)0--P--3(+1)(-1)--S--(+1)--(-1)0--Vh--3->1(+)
        # (+)1--    --(+)4    (+)1--                                --4->2(+)
        # (-)2--    --(-)5    (-)2--                                --5->3(-)
        P, S, Vh = f_c2x2_decomp(C2X2, env.chi, legs=([0,1,2],[3,4,5]))

        # 3) absorb and truncate
        # C2X2  |--3->1(+1) =>  C2X2_|--1(+) (+1->-)0---P--3->1(+1->-) => C2X2--1(-1)
        # |     |--4->2(+)      |    \--2(+) (+1->-)1--/                  |
        # |_____|--5->3(-)     0(-)   --3(-) (-1->+)2--                   0(-1)
        # |     |
        # 0(+1) 1(+),2(-)
        # 0(-)  1(-),2(+)
        # |    /         
        # P*--
        # 2->0(-1)
        C2X2= contract(P.conj(), C2X2, ([0,1,2],[0,1,2]))
        C2X2= contract(C2X2, P.flip_signature(), ([1,2,3],[0,1,2]))

        # The absorption step for C is done with T placed at B-sublattice
        # and hence the half-row/column tensor absorption step is done with T'
        # corresponding to A-sublattice
        # 
        # C--T--C => C---T--T'--T--C => C--T-- & --T'--
        # T--A--T    T---A--B---A--T    T--A--   --B--- 
        # C--T--C    T'--B--A---B--T    |  |       |
        #            T---A--B---A--T
        #            C---T--T'--T--C
        #
        # 0(+1->-1)
        # T--2(-)->2(-),3(+)->2(-1->+1),3(+1->-1)
        # 1(+1->-1)
        T= T.flip_signature()

        #       3(+)->2(+)                        0 
        # ______P___                         _____P___
        # 0(+)     1(+),2(-)->0(+),1(-)  => |        2(+),3(-)
        # 0(-)                              |
        # T--2(+),3(-)->4(+),5(-)           T--4(+),5(-)
        # 1(-)->3(-)                        1(-)
        nT= contract(P, T,([0],[0]))
        nT= permute(nT,(2,3,0,1,4,5))

        # Half-row/column tensor absorbs B-sublattice version of on-site tensor
        #
        #         0(+)                            0(+)
        #  _______P__________           =>   _____P_________
        # |             (+)2 \3->2(-)       |             | \
        # |             (-)1                T-------------a-------6->3(-)
        # T----4(+)(-)2----a--4->6(-)       |\       (-)4/|   2(-)
        # |\5->3(-)        |\0->4(-)        | \      (+)0-|--\1(+) 
        # |                3->5(-)          |  \(-)3(+)2-----a*---4->5(+)
        # 1(-)                              |             |  | 
        #                                   1(-)    (-)2<-5  3->4(+)
        #
        _a= a.flip_signature()
        nT= contract(nT, _a, ([2,4], [1,2]))
        nT= contract(nT, _a.conj(), ([2,3,4], [1,2,0]))

        #    0(+)                              0(+)
        #  __P__                              _P__
        # |     |                        =>  |    |
        # T'---a*a--3(-),5(+)->4(-),5(+)     T'--a*a--4(-),5(+)
        # |     |                            |    |
        # 1(-1) 2(-),4(+)->2(-),3(+)         1(-) 2(-),3(+)
        nT= permute(nT, (0,1,2,4,3,5))

        #    0(+)
        #  __P____
        # |       |
        # |       |                           0(+)  
        # T'-----a*a--4(-),5(+)->1(-),2(+) => T--1(-),2(+)->2(-),3(+)
        # 1(-1)   2(-),3(+)                   2->1(+)
        # 0(+1)   1(+),2(-)
        # |___P__|
        #     3(+)
        # TODO do we conjugate here ?
        nT= contract(nT, P,([1,2,3],[0,1,2]))
        
        nT= nT.transpose((0,3,1,2))

        # 4) symmetrize, normalize and assign new C,T
        C2X2= 0.5*( C2X2 + C2X2.transpose((1,0)).conj_blocks() )
        nT= 0.5*(nT + nT.transpose((1,0,2,3)).conj_blocks() )
        C2X2= C2X2/S.norm(p='inf')
        nT= nT/nT.norm(p='inf')

        # 2) Return raw new tensors
        tmp_loc= tuple([C2X2.compress_to_1d(), nT.compress_to_1d()])
        tensors_loc, metadata_store["out"]= list(zip(*tmp_loc))

        # move back to (default) cpu if offload_to_gpu is specified
        if global_args.offload_to_gpu != 'None' and global_args.device=='cpu':
            tensors_loc= tuple(r1d.to(global_args.device) for r1d in tensors_loc)
            for meta in metadata_store["out"]: 
                meta['config']= meta['config']._replace(default_device=global_args.device)

        return tensors_loc

    # Call the core function, allowing for checkpointing
    if ctm_args.fwd_checkpoint_move:
        new_tensors= checkpoint(ctm_MOVE_sl_c,*tensors)
    else:
        new_tensors= ctm_MOVE_sl_c(*tensors)

    new_tensors= tuple(yastn.decompress_from_1d(r1d, meta) \
            for r1d,meta in zip(new_tensors,metadata_store["out"]))

    env.C[env.keyC]= new_tensors[0]
    env.T[env.keyT]= new_tensors[1]