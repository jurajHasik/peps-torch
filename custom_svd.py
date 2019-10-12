import torch
import config as cfg
from linalg.svd_gesdd import SVDGESDD
from linalg.svd_rsvd import RSVD
from linalg.svd_arnoldi import ARNOLDISVD


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

def truncated_svd_symeig(M, chi, abs_tol=None, rel_tol=None):

    def flip_tensor(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,\
            dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    """
    Return a truncated SVD of a matrix M     
    M ~ (Ut)(St)(Vt)^{T}
    by computing the symmetric decomposition of MM^T

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
    sym_error= torch.norm(M-M.t())
    assert sym_error < 1.0e+8, f"enlarged corner(M) is not a symmetric matrix norm(M-M^T)>{sym_error}"
    D, U = torch.symeig(M@M, eigenvectors=True)
    # torch.symeig returns eigenpairs ordered in the ascending order with respect to eigenvalues 
    D= flip_tensor(D, 0)
    U= flip_tensor(U, 1)
    # M = USV^T => (M^T)US^-1 = V => (M^T=M) => MUS^-1 = V
    eps_cutoff=D[0] * 1.0e-8
    Dinvqsrt= torch.rsqrt(D[D>eps_cutoff])
    Dzeros= torch.zeros(D.size()[0]-Dinvqsrt.size()[0],dtype=D.dtype,device=D.device) 
    Dinvqsrt= torch.cat((Dinvqsrt,Dzeros))
    D= torch.sqrt(D)
    V= M@U@torch.diag(Dinvqsrt)

    St = D[:chi]
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

def truncated_svd_arnoldi(M, chi, abs_tol=None, rel_tol=None):
    return ARNOLDISVD.apply(M, chi)
