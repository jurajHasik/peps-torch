import numpy as np
import torch
import torch.nn.functional as Functional
try:
    import scipy.sparse.linalg
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")

class SVDSYMARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        """
        Return leading k-singular triples of a matrix M, 
        where M is symmetric M=M^t: M ~ (U)_{dim0,k}(S)_{k,k} (V)_{k,dim0}^{t}
        by computing the symmetric decomposition of M= UDU^t
        up to rank k. D is a diagonal rank-k matirx and U is 
        a set of orthonormal eigenvectors.
        
        inputs:
            M (torch.Tensor):
                tensor of shape (dim0, dim0)
            k (int):
                desired rank

        returns U, S, V
        """
        # input validation (M is square and symmetric) is provided by 
        # the scipy.sparse.linalg.eigsh
        
        # get M as numpy ndarray and wrap back to torch
        M_nograd = M.clone().detach().numpy()
        D, U= scipy.sparse.linalg.eigsh(M_nograd, k=k)

        D= torch.as_tensor(D)
        U= torch.as_tensor(U)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        S,p= torch.sort(torch.abs(D),descending=True)
        U= U[:,p]
        
        # 1) M = UDU^t = US(sgn)U^t = U S (sgn)U^t = U S V^t
        # (sgn) is a diagonal matrix with signs of the eigenvales D
        V= U@torch.diag(torch.sign(D[p]))

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("[ARNOLDISVD_SYM] backward not implemented")
        U, S, V = self.saved_tensors
        return dA, None

def test_SVDSYMARNOLDI_random():
    m= 50
    k= 10
    M= torch.rand(m, m, dtype=torch.float64)
    M= 0.5*(M+M.t())

    D0, U0= torch.symeig(M)
    S0,p= torch.sort(torch.abs(D0),descending=True)

    U,S,V= SVDSYMARNOLDI.apply(M,k)
    # |M|=\sqrt{Tr(MM^t)}=\sqrt{Tr(D^2)} => 
    # |M-US_kV^t|=\sqrt{Tr(D^2)-Tr(S^2)}=\sqrt{\sum_i>k D^2_i}
    assert( torch.norm(M-U@torch.diag(S)@V.t())-torch.sqrt(torch.sum(S0[k:]**2)) 
        < S0[0]*(m**2)*1e-14 )

if __name__=='__main__':
    test_SVDSYMARNOLDI_random()