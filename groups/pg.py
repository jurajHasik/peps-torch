import torch

def make_d2_symm(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: d2 symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Perform left-right symmetrization
    """
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right symmetry
    return A

def make_d2_antisymm(A):
    r"""
    :param A: on-site tensor
    :type A: torch.tensor
    :return: d2 anti-symmetrized tensor ``A``
    :rtype: torch.tensor

    ::
           u s 
           |/ 
        l--A--r  <=> A[s,u,l,d,r]
           |
           d
    
    Perform left-right symmetrization
    """
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right symmetry
    return A

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
    
    Project and sum real C4v irreps A1, A2, B1, B2
    """
    # A_symm= make_c4v_symm_A1(A) + make_c4v_symm_A2(A) \
    #   + make_c4v_symm_B2(A) + make_c4v_symm_B2(A)
    # A_symm= make_c4v_symm_A1(A) + make_c4v_symm_A2(A)
    A_symm = make_c4v_symm_A1(A)
    # A_symm = make_c4v_symm_B1(A)
    # A_symm = make_c4v_symm_A2(A) 
    # A_symm = make_c4v_symm_B2(A)
    return A_symm

def make_c4v_symm_A1(A):
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
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right reflection
    A= 0.5*(A + A.permute(0,3,2,1,4))   # up-down reflection
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise
    A= 0.5*(A + A.permute(0,2,3,4,1))   # pi/2 clockwise

    return A

def make_c4v_symm_A2(A):
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
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A - A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A + A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def make_c4v_symm_B1(A):
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
    A= 0.5*(A + A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A - A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A - A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A + A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def make_c4v_symm_B2(A):
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
    A= 0.5*(A - A.permute(0,1,4,3,2))   # left-right reflection (\sigma)
    A= 0.5*(A + A.permute(0,4,3,2,1))   # skew reflection (\sigma R^-1) 
    A= 0.5*(A + A.permute(0,4,1,2,3))   # pi/2 anti-clockwise (R)
    A= 0.5*(A - A.permute(0,3,4,1,2))   # pi anti-clockwise (R^2)
    return A

def verify_c4v_symm_A1(A):
    with torch.no_grad():
        symm= True
        max_d=0.
        for p in [(0,1,4,3,2), (0,3,2,1,4), (0,4,1,2,3), (0,2,3,4,1)]:
            d= torch.dist(A,A.permute(p))
            symm= symm * (d<1.0e-14)
            max_d= max(max_d,d)
        return symm, max_d
