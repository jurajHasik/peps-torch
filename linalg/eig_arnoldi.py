import numpy as np
import torch
import torch.nn.functional as Functional
try:
    import scipy.sparse.linalg
    from scipy.sparse.linalg import LinearOperator
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")

class SYMARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        r"""
        :param M: square symmetric matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.tensor
        :type k: int
        :return: eigenvalues D, leading k eigenvectors U
        :rtype: torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-eigenpairs of a matrix M, where M is symmetric 
        :math:`M=M^T`, by computing the symmetric decomposition :math:`M= UDU^T` 
        up to rank k. Partial eigendecomposition is done through Arnoldi method.
        """
        # input validation (M is square and symmetric) is provided by 
        # the scipy.sparse.linalg.eigsh
        
        # get M as numpy ndarray and wrap back to torch
        # allow for mat-vec ops to be carried out on GPU
        def mv(v):
            V= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            V= torch.mv(M,V)
            return V.detach().cpu().numpy()
        M_nograd= LinearOperator(M.size(), matvec=mv)

        D, U= scipy.sparse.linalg.eigsh(M_nograd, k=k)
        D= torch.as_tensor(D)
        U= torch.as_tensor(U)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        absD, p= torch.sort(torch.abs(D),descending=True)
        D= D[p]
        U= U[:,p]

        if M.is_cuda:
            U= U.cuda()
            D= D.cuda()

        self.save_for_backward(D, U)
        return D, U

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("backward not implemented")
        D, U= self.saved_tensors
        return dA, None

def test_SYMARNOLDI_random():
    m= 50
    k= 10
    M= torch.rand(m, m, dtype=torch.float64)
    M= 0.5*(M+M.t())

    D0, U0= torch.symeig(M)
    absD0,p= torch.sort(torch.abs(D0),descending=True)

    D,U= SYMARNOLDI.apply(M,k)
    # |M|=\sqrt{Tr(MM^t)}=\sqrt{Tr(D^2)} => 
    # |M-US_kV^t|=\sqrt{Tr(D^2)-Tr(S^2)}=\sqrt{\sum_i>k D^2_i}
    assert( torch.norm(M-U@torch.diag(D)@U.t())-torch.sqrt(torch.sum(absD0[k:]**2)) 
        < absD0[0]*(m**2)*1e-14 )

if __name__=='__main__':
    test_SYMARNOLDI_random()