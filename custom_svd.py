import torch
import config as cfg
from linalg.svd_gesdd import SVDGESDD
from linalg.svd_symeig import SVDSYMEIG
from linalg.svd_rsvd import RSVD
from linalg.svd_arnoldi import SVDSYMARNOLDI

def truncated_svd_gesdd(M, chi, abs_tol=None, rel_tol=None):
    """
    Performs a truncated SVD on a matrix M.     
    M ~ (Ut)(St)(Vt)^{T}
    
    inputs:
        M (torch.Tensor):
            tensor of shape (dim0, dim1)

        chi (int):
            maximum allowed dimension of S

        abs_tol (float):
            absolute tollerance on singular eigenvalues

        rel_tol (float):
            relative tollerance on singular eigenvalues

    where S is diagonal matrix of of shape (dimS, dimS)
    and dimS <= chi

    returns Ut, St, Vt
    """
    
    U, S, V = SVDGESDD.apply(M)
    St = S[:chi]
    # if abs_tol is not None: St = St[St > abs_tol]
    # if abs_tol is not None: St = torch.where(St > abs_tol, St, Stzeros)
    # if rel_tol is not None: St = St[St/St[0] > rel_tol]
    # magnitude = St[0]
    # if rel_tol is not None: St = torch.where(St/magnitude > rel_tol, St, Stzeros)
    # print("[truncated_svd] St "+str(St.shape[0]))
    # print(St)

    Ut = U[:, :St.shape[0]]
    Vt = V[:, :St.shape[0]]
    # print("Ut "+str(Ut.shape))
    # print("Vt "+str(Vt.shape))

    return Ut, St, Vt

def truncated_svd_symeig(M, chi, abs_tol=1.0e-14, rel_tol=None, eps_multiplet=1.0e-10, \
    verbosity=0):
    """
    Return a truncated SVD of a symmertic matrix M     
    M ~ (Ut)(St)(Vt)^t
    by computing the symmetric decomposition of M

    inputs:
        M (torch.Tensor):
            tensor of shape (dim0, dim1)

        chi (int):
            maximum allowed dimension of S

        abs_tol (float):
            absolute tolerance on singular eigenvalues

        rel_tol (float):
            relative tolerance on singular eigenvalues

    where S is diagonal matrix of of shape (dimS, dimS)
    and dimS <= chi

    returns Ut, St, Vt
    """
    U, S, V = SVDSYMEIG.apply(M)

    gaps=S.clone().detach()
    gaps[gaps < abs_tol]= 0.
    # compute gaps and normalize by larger sing. value. Introduce cutoff
    # for handling vanishing values set to exact zero
    gaps=(gaps[0:len(S)-1]-S[1:len(S)])/(gaps[0:len(S)-1]+1.0e-16)
    gaps[gaps > 1.0]= 0.

    # estimate the chi_new 
    chi_new= chi
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

def truncated_svd_rsvd(M, chi, abs_tol=None, rel_tol=None):
    return RSVD.apply(M, chi)

def truncated_svd_arnoldi(M, chi, abs_tol=None, rel_tol=None):
    return SVDSYMARNOLDI.apply(M, chi)
