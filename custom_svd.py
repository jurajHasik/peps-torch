import torch
from linalg.svd_gesdd import SVDGESDD
from linalg.svd_rsvd import RSVD
# from linalg.svd_rsvd import rsvd

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

def truncated_svd_rsvd(M, chi, abs_tol=None, rel_tol=None):
    return RSVD.apply(M, chi)
    # return rsvd(M, chi)