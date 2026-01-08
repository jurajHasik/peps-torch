import config as cfg
import yastn.yastn as yastn
from ctm.generic_abelian.ctm_components import *
from tn_interface_abelian import mm
from tn_interface_abelian import transpose
import logging
# TODO checkpointing for projector construction
# from torch.utils.checkpoint import checkpoint
log = logging.getLogger(__name__)

def ctm_get_projectors_4x4(direction, coord, state, env, ctm_args=cfg.ctm_args, \
    global_args=cfg.global_args, diagnostics=None):
    r"""
    :param direction: direction of the CTM move for which the projectors are to be computed
    :param coord: vertex (x,y) specifying (together with ``direction``) 4x4 tensor network 
                  used to build projectors
    :param state: wavefunction
    :param env: environment corresponding to ``state`` 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type direction: tuple(int,int) 
    :type coord: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: yastn.Tensor, yastn.Tensor


    Compute a pair of projectors from two halfs of 4x4 tensor network given 
    by ``direction`` and ``coord``::

        Case of LEFT move <=> direction=(-1,0)
                                                _____________
        C--T---------------T---------------C = |_____R_______|
        T--A(coord)--------A(coord+(1,0))--T    |__|_____|__| 
        |  |               |               |   |_____Rt______|
        T--A(coord+(0,1))--A(coord+(1,1))--T    
        C--T---------------T---------------C

    This function constructs two halfs of a 4x4 network and then calls 
    :py:func:`ctm_get_projectors_from_matrices` for projector construction 
    """
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1):
        # C2x2--1->0 0--C2x2(coord) =     _0(-1) (+1)0_
        # |0           1|                |             |
        # |0           0|                Rt            R
        # C2x2--1    1--C2x2             |_1(-1) (+1)1_|
        #
        R, Rt = halves_of_4x4_CTM_MOVE_UP(coord, state, env, verbosity=verbosity)
    elif direction==(-1,0): 
        # C2x2(coord)--1 0--C2x2 = ----R----
        # |0               1|      |0(-1)  |1(-1)
        # 
        # |0            1<-0|      |0(+1)  |1(+1)
        # C2x2--1 1---------C2x2   ----Rt---
        #
        R, Rt = halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, verbosity=verbosity)
    elif direction==(0,1):
        # C2x2---------1    1<-0--C2x2 =     _1(-1) (+1)1_
        # |0                      |1        |             |
        # |0                      |0        R             Rt
        # C2x2(coord)--1->0 0<-1--C2x2      |_0(-1) (+1)0_|
        #
        R, Rt = halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, verbosity=verbosity)
    elif direction==(1,0):
        # C2x2--1 0--C2x2        = ----Rt---
        # |0->1      |1->0         |1(-1)  |0(-1)
        # 
        # |0->1      |0            |1(+1)  |0(+1)
        # C2x2--1 1--C2x2(coord)   ----R----
        #
        R, Rt = halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, verbosity=verbosity)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, direction, \
        ctm_args, global_args, diagnostics=diagnostics)

def ctm_get_projectors_4x2(direction, coord, state, env, ctm_args=cfg.ctm_args, \
    global_args=cfg.global_args, diagnostics=None):
    r"""
    :param direction: direction of the CTM move for which the projectors are to be computed
    :param coord: vertex (x,y) specifying (together with ``direction``) 4x2 (vertical) or 
                  2x4 (horizontal) tensor network used to build projectors
    :param state: wavefunction
    :param env: environment corresponding to ``state`` 
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type direction: tuple(int,int) 
    :type coord: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: yastn.Tensor, yastn.Tensor


    Compute a pair of projectors from two enlarged corners making up 4x2 (2x4) tensor network 
    given by ``direction`` and ``coord``::

        Case of LEFT move <=> direction=(-1,0)
                                ____
        C--T---------------\ = |_R__|=\\
        T--A(coord)--------\\   |__|  ||
        |  |               ||  |_Rt_|=//
        T--A(coord+(0,1))--//    
        C--T---------------/

        Case of UP move <=> direction=(0,-1)
                                           ____    ___
        C--T---------T----------------C = |_Rt_|==|_R_|
        T--A(coord+(-1,0))--A(coord)--T    |  |    | |
        |  |         |                |     \_\===/_/
        \__\========/________________/

    This function constructs two enlarged corners of a 4x2 (2x4) network and then calls 
    :py:func:`ctm_get_projectors_from_matrices` for projector construction 
    """

    # function ctm_get_projectors_from_matrices expects first dimension of R, Rt
    # to be truncated. Instead c2x2 family of functions returns corners with 
    # index-position convention following the definition in env module 
    # (anti-clockwise from "up")
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1): # UP
        R= c2x2_RU(coord, state, env, verbosity=verbosity)
        Rt= c2x2_LU((coord[0]-1,coord[1]), state, env, verbosity=verbosity)
        Rt= transpose(Rt)
    elif direction==(-1,0): # LEFT
        R= c2x2_LU(coord, state, env, verbosity=verbosity)
        Rt= c2x2_LD((coord[0],coord[1]+1), state, env, verbosity=verbosity)
    elif direction==(0,1): # DOWN
        R= c2x2_LD(coord, state, env, verbosity=verbosity)
        R= transpose(R)
        Rt= c2x2_RD((coord[0]+1,coord[1]), state, env, verbosity=verbosity)
        Rt= transpose(Rt)
    elif direction==(1,0): # RIGHT
        R= c2x2_RD(coord, state, env, verbosity=verbosity)
        Rt= c2x2_RU((coord[0],coord[1]-1), state, env, verbosity=verbosity)
        Rt= transpose(Rt)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args,\
        diagnostics=diagnostics)

#####################################################################
# direction-independent function performing bi-diagonalization
#####################################################################

def ctm_get_projectors_from_matrices(R, Rt, chi, direction, \
    ctm_args=cfg.ctm_args, global_args=cfg.global_args, diagnostics=None):
    r"""
    :param R: rank-2 tensor
    :param Rt: rank-2 tensor
    :param chi: environment bond dimension
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type R: yastn.Tensor 
    :type Rt: yastn.Tensor
    :type chi: int
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: yastn.Tensor, yastn.Tensor

    Given the two tensors R and Rt (R tilde) compute the projectors P, Pt (P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf). The R, Rt are expected
    to have compatible virtual spaces on first index.
        
        1. Perform SVD over :math:`R\widetilde{R}` contracted through index which is going to
           be truncated::
           
                       _______          ______
                dim1--|___R___|--dim0--|__Rt__|--dim1  ==SVD==> dim1(R)--U--S--V^+--dim1(Rt) 

           Hence, for the inverse :math:`(R\widetilde{R})^{-1}`::
              
                       ________          ________
                dim1--|__Rt^-1_|--dim0--|__R^-1__|--dim1 = dim1(Rt)--V--S^-1--U^+--dim1(R) 

        2. Approximate an identity :math:`RR^{-1}\widetilde{R}^{-1}\widetilde{R}` by truncating
           the result of :math:`SVD(R\widetilde{R}^{-1})`::

                           ____          ______          _______          ____
                I = dim0--|_R__|--dim1--|_R^-1_|--dim0--|_Rt^-1_|--dim1--|_Rt_|--dim0
                           ____          _____                            ____          ____
                I ~ dim0--|_R__|--dim1--|_U^+_|--St^-1/2--\chi--St^-1/2--|_V__|--dim1--|_Rt_|--dim0
        
           where :math:`\widetilde{S}` has been truncated to the leading :math:`\chi` singular values    
        
        3. Finally construct the projectors :math:`P, \widetilde{P}`::
                
                           ____          _____
                P = dim0--|_R__|--dim1--|_U^+_|--St^-1/2--\chi
                                     ____          ____
                Pt = \chi--St^-1/2--|_V__|--dim1--|_Rt_|--dim0

        The projectors :math:`P, \widetilde{P}` approximate contraction of the original
        matrices :math:`R, \widetilde{R}`::
                        
             _______     _________
            |___R___| ~ |___R_____|
             _|___|_      |     |
            |___Rt__|    dim0  dim1
                        __|___  |                                         
                        \_P__/  |
                          |     |
                         chi    |
                         _|__   |
                        /_Pt_\  |
                          |     |    
                         dim0  dim1
                         _|_____|_
                        |____Rt___|
    """
    assert R.ndim == Rt.ndim and R.ndim == 2
    verbosity = ctm_args.verbosity_projectors

    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method=='GESDD':
        def truncation_f(S):
            if ctm_args.projector_eps_multiplet>0:
                return yastn.linalg.truncation_mask_multiplets(S,keep_multiplets=True, D_total=chi,\
                    tol=ctm_args.projector_svd_reltol, tol_block=ctm_args.projector_svd_reltol_block, \
                    eps_multiplet=ctm_args.projector_eps_multiplet)
            return yastn.linalg.truncation_mask(S, D_total=chi,\
                tol=ctm_args.projector_svd_reltol, tol_block=ctm_args.projector_svd_reltol_block)
        def truncated_svd(M, chi, sU=1):
            return yastn.linalg.svd_with_truncation(M, (0,1), sU=sU, mask_f=truncation_f, diagnostics=diagnostics)
    # elif ctm_args.projector_svd_method == 'ARP':
    #     def truncated_svd(M, chi):
    #         return truncated_svd_arnoldi(M, chi, verbosity=ctm_args.verbosity_projectors)
    else:
        raise(f"Projector svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    # 0)
    # Move: UP    (+1)0--R--1(+1) =>T=> (+1)1--R--0(+1)(-1)0--Rt--1(-1) =>SVD=> (+1)U(?)S(-?)Vh(-1)
    #       LEFT  (-1)0--R--1(-1) =>T=> (-1)1--R--0(-1)(+1)0--Rt--1(+1) =>SVD=> (-1)U(?)S(-?)Vh(+1)
    #       DOWN  (-1)0--R--1(-1) =>T=> (-1)1--R--0(-1)(+1)0--Rt--1(+1) =>SVD=> (-1)U(?)S(-?)Vh(+1)
    #       RIGHT (+1)0--R--1(+1) =>T=> (+1)1--R--0(+1)(-1)0--Rt--1(-1) =>SVD=> (+1)U(?)S(-?)Vh(-1)
    #
    if ctm_args.fwd_checkpoint_projectors:
        raise RuntimeError("Checkpointing projectors not implemented")
        M = checkpoint(mm, transpose(R), Rt)
    else:
        M = R.tensordot(Rt,([0],[0]))
    if ctm_args.verbosity_projectors > 0:
        print(f"{diagnostics}")
        M.print_blocks_shape()
    if ctm_args.verbosity_projectors > 1:
        M_unfused= M.unfuse_legs(axes=(0,1))
        print(f"{M_unfused}")
        M_unfused.print_blocks_shape()

    # 1) SVD decomposition and Truncation
    signature_U={(0,-1): 1, (-1,0): -1, (0,1): -1, (1,0): 1}
    U, S, Vh = truncated_svd(M, chi, signature_U[direction]) # M = USV^{+}
    S_sqrt= S.rsqrt()
    
    if verbosity>0: 
        print(f"{diagnostics} {S.data.max()} {S_sqrt.data.max()}")

    # 3) Construct projectors
    expr='ij,j->ij'
    def P_Pt_c(*tensors):
        R, Rt, U, Vh, S_sqrt= tensors
        # Move: UP    (+1)0--U--1(?)=>C=> (+1)0--R--1(+1)(-1)0--U--1(-?)S_sqrt = (+1)P(-1)
        #       ?=1   (-?)0--Vh--1(-1)=>CT=> (-1)0--Rt--1(-1)(+1)1--Vh--0(?)   = (-1)Pt(+1)
        #       LEFT  (-1)0--U--1(?)=>C=> (-1)0--R--1(-1)(+1)0--U--1(-?)S_sqrt = (-1)P(+1)
        #       ?=-1  (-?)0--Vh--1(+1)=>CT=> (+1)0--Rt--1(+1)(-1)1--Vh--0(?)   = (+1)Pt(-1)
        #       DOWN  (-1)0--U--1(?)=>C=> (-1)0--R--1(-1)(+1)0--U--1(-?)S_sqrt = (-1)P(+1)
        #       ?=-1  (-?)0--Vh--1(+1)=>CT=> (+1)0--Rt--1(+1)(-1)1--Vh--0(?)   = (+1)Pt(-1)
        #       RIGHT (+1)0--U--1(?)=>C=> (+1)0--R--1(+1)(-1)0--U--1(-?)S_sqrt = (+1)P(-1)
        #             (-?)0--Vh--1(-1)=>CT=> (-1)0--Rt--1(-1)(+1)1--Vh--0(?)   = (-1)Pt(+1)
        P= ( R.tensordot(U.conj(), ([1],[0])) ).tensordot(S_sqrt,([1],[1]))
        Pt= ( Rt.tensordot(Vh.conj(),([1],[1])) ).tensordot(S_sqrt,([1],[0]))
        return P, Pt

    tensors= R, Rt, U, Vh, S_sqrt
    if ctm_args.fwd_checkpoint_projectors:
        raise RuntimeError("Checkpointing projectors not implemented")
        return checkpoint(P_Pt_c, *tensors)
    else:
        return P_Pt_c(*tensors)
