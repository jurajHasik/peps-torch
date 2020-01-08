import torch

def make_c4v_symm(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: c4v symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Perform left-right, up-down, diagonal symmetrization
    """
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right symmetry
    A= 0.5*(A + A.permute(0,3,2,1,4))   # up-down symmetry
    A= 0.5*(A + A.permute(0,4,3,2,1))   # skew-diagonal symmetry
    A= 0.5*(A + A.permute(0,2,1,4,3))   # diagonal symmetry
    return A