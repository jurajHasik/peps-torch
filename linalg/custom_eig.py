import torch
import config as cfg
from linalg.eig_sym import SYMEIG
from linalg.eig_arnoldi import SYMARNOLDI

def truncated_eig_sym(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: symmetric matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal(in magnitude) eigenvalue 
    :param rel_tol: relative tolerance on minimal(in magnitude) eigenvalue
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.tensor
    :type chi: int
    :type abs_tol: float
    :type rel_tol: float
    :type keep_multiplets: bool
    :type eps_multiplet: float
    :type verbosity: int
    :return: leading :math:`\chi` eigenvalues D and eigenvectors U
    :rtype: torch.tensor, torch.tensor

    Returns leading :math:`\chi` eigenpairs of a matrix M, where M is a symmetric 
    matrix :math:`M=M^T`, by computing the full symmetric decomposition :math:`M= UDU^T`. 
    Returned tensors have dimensions

    .. math:: dim(D)=(\chi),\ dim(U)=(N,\chi)
    """
    D, U= SYMEIG.apply(M)

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets and chi<D.shape[0]:
        # regularize by discarding small values
        gaps=torch.abs(D[:chi+1].clone().detach())
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:chi]-torch.abs(D[1:chi+1]))/(gaps[:chi]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        Dt = D[:chi].clone()
        Dt[chi_new+1:]=0.

        Ut = U[:, :Dt.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        return Dt, Ut

    Dt = D[:min(chi,D.shape[0])]
    Ut = U[:, :Dt.shape[0]]

    return Dt, Ut

def truncated_eig_symarnoldi(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: symmetric matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal(in magnitude) eigenvalue 
    :param rel_tol: relative tolerance on minimal(in magnitude) eigenvalue
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.tensor
    :type chi: int
    :type abs_tol: float
    :type rel_tol: float
    :type keep_multiplets: bool
    :type eps_multiplet: float
    :type verbosity: int
    :return: leading :math:`\chi` eigenvalues D and eigenvectors U
    :rtype: torch.tensor, torch.tensor

    **Note:** `depends on scipy`

    Returns leading :math:`\chi` eigenpairs of a matrix M, where M is a symmetric matrix :math:`M=M^T`,
    by computing the partial symmetric decomposition :math:`M= UDU^T` up to rank :math:`\chi`. 
    Returned tensors have dimensions 

    .. math:: dim(D)=(\chi),\ dim(U)=(N,\chi)
    """
    D, U= SYMARNOLDI.apply(M, chi+int(keep_multiplets))

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets:
        # regularize by discarding small values
        gaps=torch.abs(D.clone().detach())
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:len(D)-1]-torch.abs(D[1:len(D)]))/(gaps[:len(D)-1]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        Dt = D[:chi].clone()
        Dt[chi_new+1:]=0.

        Ut = U[:, :Dt.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        return Dt, Ut

    return D, U

