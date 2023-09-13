import torch
from linalg.svd_gesdd import SVDGESDD
from linalg.svd_symeig import SVDSYMEIG
from linalg.svd_arnoldi import SVDSYMARNOLDI, SVDARNOLDI
from linalg.svd_rsvd import RSVD
from linalg.svd_af import SVDAF

def _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol):
    # estimate the chi_new 
    chi_new= chi
    # regularize by discarding small values
    gaps=S[:chi+1].clone().detach()
    # S[S < abs_tol]= 0.
    gaps[gaps < abs_tol]= 0.
    # compute gaps and normalize by larger sing. value. Introduce cutoff
    # for handling vanishing values set to exact zero
    gaps=(gaps[:chi]-S[1:chi+1])/(gaps[:chi]+1.0e-16)
    gaps[gaps > 1.0]= 0.

    if gaps[chi-1] < eps_multiplet:
        # the chi is within the multiplet - find the largest chi_new < chi
        # such that the complete multiplets are preserved
        for i in range(chi-1,-1,-1):
            if gaps[i] > eps_multiplet:
                chi_new= i
                break

    St = S[:chi].clone()
    St[chi_new+1:]=0.

    Ut = U[:, :St.shape[0]].clone()
    Ut[:, chi_new+1:]=0.
    Vt = V[:, :St.shape[0]].clone()
    Vt[:, chi_new+1:]=0.
    return Ut, St, Vt


def truncated_svd_gesdd(M, chi, abs_tol=1.0e-14, rel_tol=None, ad_decomp_reg=1.0e-12,\
    keep_multiplets=False, eps_multiplet=1.0e-12, verbosity=0, diagnostics=None):
    r"""
    :param M: matrix of dimensions :math:`N \times L`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
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
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    Returns leading :math:`\chi`-singular triples of a matrix M by computing the full 
    SVD :math:`M= USV^T`. Returned tensors have dimensions

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(L,\chi)
    """
    reg= torch.as_tensor(ad_decomp_reg, dtype=M.real.dtype if M.is_complex() else M.dtype,\
        device=M.device)
    U, S, V = SVDGESDD.apply(M, reg, diagnostics)

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets and chi<S.shape[0]:
        # regularize by discarding small values
        gaps=S[:chi+1].clone().detach()
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:chi]-S[1:chi+1])/(gaps[:chi]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        St = S[:chi].clone()
        St[chi_new+1:]=0.

        Ut = U[:, :St.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        Vt = V[:, :St.shape[0]].clone()
        Vt[:, chi_new+1:]=0.
        return Ut, St, Vt

    St = S[:min(chi,S.shape[0])]
    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]
    
    return Ut, St, Vt

def truncated_svd_af(M, chi, abs_tol=1.0e-14, rel_tol=None, ad_decomp_reg=1.0e-12,\
    keep_multiplets=False, eps_multiplet=1.0e-12, verbosity=0, diagnostics=None):
    r"""
    :param M: matrix of dimensions :math:`N \times L`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
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
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    Returns leading :math:`\chi`-singular triples of a matrix M by computing the full 
    SVD :math:`M= USV^T`. Returned tensors have dimensions

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(L,\chi)
    """
    reg= torch.as_tensor(ad_decomp_reg, dtype=M.real.dtype if M.is_complex() else M.dtype,\
        device=M.device)
    U, S, V = SVDAF.apply(M, reg, diagnostics)

    # estimate the chi_new 
    if keep_multiplets and chi<S.shape[0]:
        return _keep_multiplets(U,S,V,chi,eps_multiplet,abs_tol)

    St = S[:min(chi,S.shape[0])]
    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]
    
    return Ut, St, Vt

def truncated_svd_symeig(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: square matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
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
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    Returns leading :math:`\chi`-singular triples of a matrix M, where M is a symmetric 
    matrix :math:`M=M^T`, by computing the full symmetric decomposition :math:`M= UDU^T`. 
    Returned tensors have dimensions

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(N,\chi)

    .. note::
        This function does not support autograd.
    """
    U, S, V = SVDSYMEIG.apply(M)

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets and chi<S.shape[0]:
        # regularize by discarding small values
        gaps=S[:chi+1].clone().detach()
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:chi]-S[1:chi+1])/(gaps[:chi]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        St = S[:chi].clone()
        St[chi_new+1:]=0.

        Ut = U[:, :St.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        Vt = V[:, :St.shape[0]].clone()
        Vt[:, chi_new+1:]=0.
        return Ut, St, Vt

    St = S[:min(chi,S.shape[0])]
    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]

    return Ut, St, Vt

def truncated_svd_symarnoldi(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: square matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
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
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    **Note:** `depends on scipy`

    Returns leading :math:`\chi`-singular triples of a matrix M, where M is a symmetric matrix :math:`M=M^T`,
    by computing the partial symmetric decomposition :math:`M= UDU^T` up to rank :math:`\chi`. 
    Returned tensors have dimensions 

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(N,\chi)

    .. note::
        This function does not support autograd.
    """
    U, S, V = SVDSYMARNOLDI.apply(M, chi+int(keep_multiplets))

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets:
        # regularize by discarding small values
        gaps=S.clone().detach()
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:len(S)-1]-S[1:len(S)])/(gaps[:len(S)-1]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        St = S[:chi].clone()
        St[chi_new+1:]=0.

        Ut = U[:, :St.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        Vt = V[:, :St.shape[0]].clone()
        Vt[:, chi_new+1:]=0.
        return Ut, St, Vt

    return U, S, V

def truncated_svd_arnoldi(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: square matrix of dimensions :math:`N \times N`
    :param chi: desired maximal rank :math:`\chi`
    :param abs_tol: absolute tolerance on minimal singular value 
    :param rel_tol: relative tolerance on minimal singular value
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
    :return: leading :math:`\chi` left singular vectors U, right singular vectors V, and
             singular values S
    :rtype: torch.tensor, torch.tensor, torch.tensor

    **Note:** `depends on scipy`

    Returns leading :math:`\chi`-singular triples of a matrix M,
    by computing the partial symmetric decomposition of :math:`H=M^TM` as :math:`H= UDU^T` 
    up to rank :math:`\chi`. Returned tensors have dimensions 

    .. math:: dim(U)=(N,\chi),\ dim(S)=(\chi,\chi),\ \textrm{and}\ dim(V)=(N,\chi)

    .. note::
        This function does not support autograd.
    """
    U, S, V = SVDARNOLDI.apply(M, chi+int(keep_multiplets))

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets:
        # regularize by discarding small values
        gaps=S.clone().detach()
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by larger sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:len(S)-1]-S[1:len(S)])/(gaps[:len(S)-1]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= i
                    break

        St = S[:chi].clone()
        St[chi_new+1:]=0.

        Ut = U[:, :St.shape[0]].clone()
        Ut[:, chi_new+1:]=0.
        Vt = V[:, :St.shape[0]].clone()
        Vt[:, chi_new+1:]=0.
        return Ut, St, Vt

    return U, S, V

def truncated_svd_rsvd(M, chi, abs_tol=None, rel_tol=None):
    return RSVD.apply(M, chi)
