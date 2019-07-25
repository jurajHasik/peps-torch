import torch
from ipeps import IPEPS
from env import ENV
from args import CTMARGS, GLOBALARGS
from ctm_components import *
from custom_svd import *

def ctm_get_projectors(direction, coord, state, env, ctm_args=CTMARGS(), global_args=GLOBALARGS()):
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
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1):
        R, Rt = halves_of_4x4_CTM_MOVE_UP(coord, state, env, verbosity=verbosity)
    elif direction==(-1,0): 
        R, Rt = halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, verbosity=verbosity)
    elif direction==(0,1):
        R, Rt = halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, verbosity=verbosity)
    elif direction==(1,0):
        R, Rt = halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, verbosity=verbosity)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args)

#####################################################################
# direction-independent function performing bi-diagonalization
#####################################################################

def ctm_get_projectors_from_matrices(R, Rt, chi, ctm_args=CTMARGS(), global_args=GLOBALARGS()):

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
    :return: pair of projectors, tensors of dimension :math:`\chi \times \chi \times D^2`. 
             The D might vary depending on the auxiliary bond dimension of related on-site
             tensor.
    :rtype: torch.tensor, torch.tensor

    Given the two tensor R and Rt (R tilde) compute the projectors P, Pt (P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf)::
        
         _______________
        |______R________|
          |         |
         dim0      dim1
        __|______   | 
        \___P___/   
            |
           chi
            |

            |
           chi
         ___|___
        /_ _Pt__\
          |         |    
         dim0      dim1
         _|_________|___
        |______Rt_______|
    """
    assert R.shape == Rt.shape
    assert len(R.shape) == 2
    verbosity = ctm_args.verbosity_projectors

    #  SVD decomposition
    M = torch.mm(R.transpose(1, 0), Rt)
    U, S, V = truncated_svd_gesdd(M, chi) # M = USV^{T}

    # if abs_tol is not None: St = St[St > abs_tol]
    # if abs_tol is not None: St = torch.where(St > abs_tol, St, Stzeros)
    # if rel_tol is not None: St = St[St/St[0] > rel_tol]
    S = S[S/S[0] > ctm_args.projector_svd_reltol]
    S_zeros = torch.zeros((chi-S.size()[0]), dtype=global_args.dtype, device=global_args.device)
    S_sqrt = torch.rsqrt(S)
    S_sqrt = torch.cat((S_sqrt, S_zeros))
    if verbosity>0: print(S_sqrt)

    # Construct projectors
    # P = torch.einsum('i,ij->ij', S_sqrt, torch.mm(U.transpose(1, 0), R.transpose(1, 0)))
    P = torch.einsum('ij,j->ij', torch.mm(R, U), S_sqrt)
    # Pt = torch.einsum('i,ij->ij', S_sqrt, torch.mm(V.transpose(1, 0), Rt.transpose(1, 0)))
    Pt = torch.einsum('ij,j->ij', torch.mm(Rt, V), S_sqrt)

    return P, Pt