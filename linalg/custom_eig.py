import torch
import config as cfg
from linalg.eig_sym import SYMEIG
from linalg.eig_arnoldi import SYMARNOLDI, ARNOLDI
from linalg.eig_lobpcg import SYMLOBPCG

def truncated_eig_sym(M, chi, abs_tol=1.0e-14, rel_tol=None, ad_decomp_reg=1.0e-12, \
    keep_multiplets=False, eps_multiplet=1.0e-12, verbosity=0):
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
    reg= torch.as_tensor(ad_decomp_reg, dtype=M.real.dtype if M.is_complex() else M.dtype,\
        device=M.device)
    D, U= SYMEIG.apply(M,reg)

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

    .. note::
        This function does not support autograd.
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

def truncated_eig_arnoldi(M, chi, v0=None, dtype=None, device=None, 
    abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, eps_multiplet=1.0e-12, verbosity=0):
    r"""
    :param M: matrix of dimensions :math:`N \times N` or numpy LinearOperator
    :param chi: desired maximal rank :math:`\chi`
    :param v0: initial vector
    :param abs_tol: absolute tolerance on minimal(in magnitude) eigenvalue 
    :param rel_tol: relative tolerance on minimal(in magnitude) eigenvalue
    :param keep_multiplets: truncate spectrum down to last complete multiplet
    :param eps_multiplet: allowed splitting within multiplet
    :param verbosity: logging verbosity
    :type M: torch.Tensor or scipy.sparse.linalg.LinearOperator
    :type chi: int
    :type v0: torch.Tensor
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

    .. note::
        This function does not support autograd.
    """

    D, U= ARNOLDI.apply(M, chi+int(keep_multiplets), v0, dtype, device)

    if keep_multiplets:
        raise Exception("keep_multiplets not implemented")

    return D, U

def truncated_eig_symlobpcg(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-8, verbosity=0):
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

    Returns leading (by magnitude) :math:`\chi` eigenpairs of a matrix M, where M is 
    a symmetric matrix :math:`M=M^T`, by computing the partial symmetric decomposition 
    :math:`MM= UDU^T` up to rank :math:`\chi`. The decomposition is computed by LOBPCG.

    Returned tensors have dimensions 

    .. math:: dim(D)=(\chi),\ dim(U)=(N,\chi)
    """
    # (optional) verify hermicity
    M_asymm_norm= torch.norm(M-M.t())
    assert M_asymm_norm/torch.abs(M).max() < 1.0e-8, "M is not symmetric"

    MM= M@M
    D2, U= SYMLOBPCG.apply(MM, chi+int(keep_multiplets), min(abs_tol,eps_multiplet))

    # find multiplets
    m=[]
    l=0
    for i in range(chi):
        l+=1
        g=D2[i]-D2[i+1]
        if g>eps_multiplet:
            m.append(l)
            l=0
            if D2[i+1]<abs_tol: break

    mixed_spec= U.t() @ M @ U
    D= torch.diag(mixed_spec)
    i=0
    for ml in m:
        if ml>1:
            Dml, Uml= torch.symeig(mixed_spec[i:i+ml,i:i+ml], eigenvectors=True)
            U[:,i:i+ml]= U[:,i:i+ml] @ Uml
            D[i:i+ml]= Dml
        i+= ml

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets:
        # regularize by discarding small values
        gaps=torch.abs(D2.clone().detach())
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by the largest sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:len(D2)-1]-torch.abs(D2[1:len(D2)]))/(gaps[:len(D2)-1]+1.0e-16)
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

def truncated_eig_symlobpcg_v2(M, chi, abs_tol=1.0e-14, rel_tol=None, keep_multiplets=False, \
    eps_multiplet=1.0e-8, verbosity=0):
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

    Returns leading (by magnitude) :math:`\chi` eigenpairs of a matrix M, where M is 
    a symmetric matrix :math:`M=M^T`, by computing the partial symmetric decomposition 
    :math:`MM= UDU^T` up to rank :math:`\chi`. The decomposition is computed by LOBPCG.

    Returned tensors have dimensions 

    .. math:: dim(D)=(\chi),\ dim(U)=(N,\chi)
    """
    # (optional) verify hermicity
    M_asymm_norm= torch.norm(M-M.t())
    assert M_asymm_norm/torch.abs(M).max() < 1.0e-8, "M is not symmetric"

    # get both negative and positive leading vectors
    Dp, Up= SYMLOBPCG.apply(M, chi+int(keep_multiplets), None)
    Dn, Un= SYMLOBPCG.apply(-M, chi+int(keep_multiplets), None)

    # sort by magnitude in descending order, keeping the reference to two spectra
    tmp= [(Dp[i],i,1) for i in range(Dp.size(0))]+[(Dn[i],i,0) for i in range(Dn.size(0))]
    tmp.sort(key=lambda e: e[0], reverse=True)

    # estimate the chi_new 
    chi_new= chi
    if keep_multiplets:
        Dpn= torch.cat((Dp, Dn))
        Dpn, ppn= torch.sort(Dpn, descending=True)
        Dpn= Dpn[:chi+1]

        # regularize by discarding small values
        gaps=torch.abs(Dpn.clone().detach())
        # S[S < abs_tol]= 0.
        gaps[gaps < abs_tol]= 0.
        # compute gaps and normalize by the largest sing. value. Introduce cutoff
        # for handling vanishing values set to exact zero
        gaps=(gaps[:len(Dpn)-1]-torch.abs(Dpn[1:len(Dpn)]))/(gaps[:len(Dpn)-1]+1.0e-16)
        gaps[gaps > 1.0]= 0.

        if gaps[chi-1] < eps_multiplet:
            # the chi is within the multiplet - find the largest chi_new < chi
            # such that the complete multiplets are preserved
            for i in range(chi-1,-1,-1):
                if gaps[i] > eps_multiplet:
                    chi_new= min(i+1, chi)
                    break

    # pick the leading chi_new eigenpairs and recover two lists of indices
    ind_p= filter(lambda x: x[1][2]==1, enumerate(tmp[:chi_new]))
    ind_n= filter(lambda x: x[1][2]==0, enumerate(tmp[:chi_new]))

    # maps of joined_index, original_index
    ind_p= list(torch.as_tensor(list(l)) for l in zip(*[(i[0],i[1][1]) for i in ind_p]) )
    ind_n= list(torch.as_tensor(list(l)) for l in zip(*[(i[0],i[1][1]) for i in ind_n]) )

    D= torch.zeros(chi, dtype=M.dtype, device=M.device)
    D.put_(ind_p[0], Dp[ind_p[1]])
    D.put_(ind_n[0], -Dn[ind_n[1]])

    U= torch.zeros((M.size(0),chi), dtype=M.dtype, device=M.device)
    U[:,ind_p[0]]= Up[:,ind_p[1]]
    U[:,ind_n[0]]= Un[:,ind_n[1]]

    return D, U
