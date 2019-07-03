import torch
from ipeps import IPEPS
from env import ENV
from args import CTMARGS, GLOBALARGS
from ctm_components import *
from custom_svd import *

#####################################################################
# compute the projectors from 4x4 TN given by coord
#####################################################################

def ctm_get_projectors(direction, coord, ipeps, env, ctm_args=CTMARGS(), global_args=GLOBALARGS()):
    verbosity = ctm_args.verbosity_projectors
    if direction==(0,-1):
        R, Rt = halves_of_4x4_CTM_MOVE_UP(coord, ipeps, env, verbosity=verbosity)
    elif direction==(-1,0): 
        R, Rt = halves_of_4x4_CTM_MOVE_LEFT(coord, ipeps, env, verbosity=verbosity)  
    elif direction==(0,1):
        R, Rt = halves_of_4x4_CTM_MOVE_DOWN(coord, ipeps, env, verbosity=verbosity)
    elif direction==(1,0):
        R, Rt = halves_of_4x4_CTM_MOVE_RIGHT(coord, ipeps, env, verbosity=verbosity)
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return ctm_get_projectors_from_matrices(R, Rt, env.chi, ctm_args, global_args)

#####################################################################
# direction-independent function performing bi-diagonalization
#####################################################################

def ctm_get_projectors_from_matrices(R, Rt, chi, ctm_args=CTMARGS(), global_args=GLOBALARGS()):
    """
    Given the two tensor T and Tt (T tilde) this computes the projectors
    Computes The projectors (P, P tilde)
    (PRB 94, 075143 (2016) https://arxiv.org/pdf/1402.2859.pdf)
    The indices of the input R, Rt are

        R (torch.Tensor):
            tensor of shape (dim0, dim1)
        Rt (torch.Tensor):
            tensor of shape (dim0, dim1)
        chi (int):
            auxiliary bond dimension  

    --------------------
    |        T         |
    --------------------
      |         |
     dim0      dim1
      |         |
    ---------  
     \\ P //   
     -------
        |
       chi
        |


        |
       chi
        |
     -------   
    // Pt  \\
    ---------
      |         |    
     dim0      dim1
      |         |    
    --------------------
    |        Rt        |
    --------------------
    """
    assert R.shape == Rt.shape
    assert len(R.shape) == 2

    #  SVD decomposition
    M = torch.mm(R.transpose(1, 0), Rt)
    # embed()
    U, S, V = truncated_svd_gesdd(M, chi) # M = USV^{T}

    # if abs_tol is not None: St = St[St > abs_tol]
    # if abs_tol is not None: St = torch.where(St > abs_tol, St, Stzeros)
    # if rel_tol is not None: St = St[St/St[0] > rel_tol]
    S = S[S/S[0] > ctm_args.projector_svd_reltol]
    S_zeros = torch.zeros((chi-S.size()[0]), dtype=global_args.dtype, device=global_args.device)
    S_sqrt = torch.rsqrt(S)
    S_sqrt = torch.cat((S_sqrt, S_zeros))
    # print(S_sqrt)

    # Construct projectors
    # P = torch.einsum('i,ij->ij', S_sqrt, torch.mm(U.transpose(1, 0), R.transpose(1, 0)))
    P = torch.einsum('ij,j->ij', torch.mm(R, U), S_sqrt)
    # Pt = torch.einsum('i,ij->ij', S_sqrt, torch.mm(V.transpose(1, 0), Rt.transpose(1, 0)))
    Pt = torch.einsum('ij,j->ij', torch.mm(Rt, V), S_sqrt)

    return P, Pt