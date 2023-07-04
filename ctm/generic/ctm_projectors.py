import torch
from torch.utils.checkpoint import checkpoint
import config as cfg
#from ipeps.ipeps import IPEPS
#from ctm.generic.env import ENV
from ctm.generic.ctm_components import *
from linalg.custom_svd import *
from tn_interface import mm
from tn_interface import conj, transpose
import logging
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
    :type state: IPEPS
    :type env: ENV
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor


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
    mode = 'dl' if ctm_args.ctm_force_dl else 'sl'
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1):
        R, Rt = halves_of_4x4_CTM_MOVE_UP(coord, state, env, mode=mode, verbosity=verbosity)
    elif direction==(-1,0): 
        R, Rt = halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, mode=mode, verbosity=verbosity)
    elif direction==(0,1):
        R, Rt = halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, mode=mode, verbosity=verbosity)
    elif direction==(1,0):
        R, Rt = halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, mode=mode, verbosity=verbosity)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args,\
        diagnostics=diagnostics)

def ctm_get_projectors_4x2(direction, coord, state, env, ctm_args=cfg.ctm_args, \
    global_args=cfg.global_args,diagnostics=None):
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
    :type state: IPEPS
    :type env: ENV
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor


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
    mode = 'dl' if ctm_args.ctm_force_dl else 'sl'
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1): # UP
        R= c2x2_RU(coord, state, env, mode=mode, verbosity=verbosity)
        Rt= c2x2_LU((coord[0]-1,coord[1]), state, env, mode=mode, verbosity=verbosity)
        Rt= transpose(Rt)
    elif direction==(-1,0): # LEFT
        R= c2x2_LU(coord, state, env, mode=mode, verbosity=verbosity)
        Rt= c2x2_LD((coord[0],coord[1]+1), state, env, mode=mode, verbosity=verbosity)
    elif direction==(0,1): # DOWN
        R= c2x2_LD(coord, state, env, mode=mode, verbosity=verbosity)
        R= transpose(R)
        Rt= c2x2_RD((coord[0]+1,coord[1]), state, env, mode=mode, verbosity=verbosity)
        Rt= transpose(Rt)
    elif direction==(1,0): # RIGHT
        R= c2x2_RD(coord, state, env, mode=mode, verbosity=verbosity)
        Rt= c2x2_RU((coord[0],coord[1]-1), state, env, mode=mode, verbosity=verbosity)
        Rt= transpose(Rt)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args,\
        diagnostics=diagnostics)

#####################################################################
# direction-independent function performing bi-diagonalization
#####################################################################

def ctm_get_projectors_from_matrices(R, Rt, chi, ctm_args=cfg.ctm_args, \
    global_args=cfg.global_args, diagnostics=None):
    r"""
    :param R: tensor of shape (dim0, dim1)
    :param Rt: tensor of shape (dim0, dim1)
    :param chi: environment bond dimension
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type R: torch.tensor 
    :type Rt: torch.tensor
    :type chi: int
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :return: pair of projectors P, Pt, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor

    Given the two tensors R and Rt (R tilde) compute the projectors P, Pt (P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf)
        
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
                        \_Pt__/ |
                          |     |
                         chi    |
                         _|__   |
                        /_P _\  |
                          |     |    
                         dim0  dim1
                         _|_____|_
                        |____Rt___|
    """
    assert R.shape == Rt.shape
    assert len(R.shape) == 2
    verbosity = ctm_args.verbosity_projectors

    if ctm_args.projector_svd_method=='DEFAULT' or ctm_args.projector_svd_method in ['GESDD','GESDD_CPU']:
        # returns U, S, V of M= USV^\dag
        if ctm_args.projector_svd_method=="GESDD_CPU":
            def truncated_svd(M, chi):
                _M= M.cpu()
                _USV= truncated_svd_gesdd(_M, chi, keep_multiplets=True, \
                    abs_tol=ctm_args.projector_multiplet_abstol,\
                    eps_multiplet=ctm_args.projector_eps_multiplet, verbosity=ctm_args.verbosity_projectors,\
                    diagnostics=diagnostics)
                return (x.to(device=M.device) for x in _USV)
        else:
            def truncated_svd(M, chi):
                return truncated_svd_gesdd(M, chi, keep_multiplets=True, \
                    abs_tol=ctm_args.projector_multiplet_abstol,\
                    eps_multiplet=ctm_args.projector_eps_multiplet, verbosity=ctm_args.verbosity_projectors,\
                    diagnostics=diagnostics)
    elif ctm_args.projector_svd_method=='AF':
        def truncated_svd(M, chi):
            return truncated_svd_af(M, chi, keep_multiplets=True, \
                abs_tol=ctm_args.projector_multiplet_abstol,\
                eps_multiplet=ctm_args.projector_eps_multiplet, verbosity=ctm_args.verbosity_projectors,\
                diagnostics=diagnostics)
    elif ctm_args.projector_svd_method == 'ARP':
        def truncated_svd(M, chi):
            return truncated_svd_arnoldi(M, chi, keep_multiplets=True, \
                abs_tol=ctm_args.projector_multiplet_abstol, \
                eps_multiplet=ctm_args.projector_eps_multiplet, verbosity=ctm_args.verbosity_projectors)
    else:
        raise(f"Projector svd method \"{cfg.ctm_args.projector_svd_method}\" not implemented")

    #  SVD decomposition
    if ctm_args.fwd_checkpoint_projectors:
        M = checkpoint(mm, transpose(R), Rt)
    else:
        M = mm(transpose(R), Rt)
    U, S, V = truncated_svd(M, chi) # M = USV^{T}

    S_nz= S[S/S[0] > ctm_args.projector_svd_reltol]
    S_sqrt= S*0
    S_sqrt[:S_nz.size(0)]= torch.rsqrt(S_nz)
    
    if verbosity>0:
        log.info(f"{diagnostics}")
    if verbosity>1: print(S_sqrt)

    # Construct projectors
    expr='ij,j->ij'
    def P_Pt_c(*tensors):
        R, Rt, U, V, S_sqrt= tensors
        return mm(R, conj(U))*S_sqrt[None,:], mm(Rt,V)*S_sqrt[None,:]

    tensors= R, Rt, U, V, S_sqrt
    if ctm_args.fwd_checkpoint_projectors:
        return checkpoint(P_Pt_c, *tensors)
    else:
        return P_Pt_c(*tensors)
