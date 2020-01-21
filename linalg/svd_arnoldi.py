import numpy as np
import torch
import torch.nn.functional as Functional
try:
    import scipy.sparse.linalg
    from scipy.sparse.linalg import LinearOperator
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")

class SVDSYMARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        r"""
        :param M: square symmetric matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.tensor
        :type k: int
        :return: leading k left eigenvectors U, singular values S, and right 
                 eigenvectors V
        :rtype: torch.tensor, torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, where M is symmetric 
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
        
        # M_nograd = M.clone().detach().cpu().numpy()
        M_nograd= LinearOperator(M.size(), matvec=mv)

        D, U= scipy.sparse.linalg.eigsh(M_nograd, k=k)

        D= torch.as_tensor(D)
        U= torch.as_tensor(U)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        S,p= torch.sort(torch.abs(D),descending=True)
        U= U[:,p]
        
        # 1) M = UDU^t = US(sgn)U^t = U S (sgn)U^t = U S V^t
        # (sgn) is a diagonal matrix with signs of the eigenvales D
        V= U@torch.diag(torch.sign(D[p]))

        if M.is_cuda:
            U= U.cuda()
            V= V.cuda()
            S= S.cuda()

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("backward not implemented")
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

class SVDARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.tensor
        :type k: int
        :return: leading k left eigenvectors U, singular values S, and right 
                 eigenvectors V
        :rtype: torch.tensor, torch.tensor, torch.tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, by computing 
        the symmetric decomposition of H=MM^T as :math:`H= UDU^T` 
        up to rank k. Partial eigendecomposition is done through Arnoldi method.
        """
        # input validation is provided by the scipy.sparse.linalg.eigsh
        
        # allow for mat-vec ops to be carried out on GPU
        def mv(v):
            B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            B= torch.mv(M,B)
            return B.detach().cpu().numpy()
        def vm(v):
            B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            B= torch.matmul(M.t(),B)            
            return B.detach().cpu().numpy()

        M_nograd= LinearOperator(M.size(), matvec=mv, rmatvec=vm)

        U, S, V= scipy.sparse.linalg.svds(M_nograd, k=k)

        S= torch.as_tensor(S)
        U= torch.as_tensor(U)
        # transpose wrt to pytorch
        V= torch.as_tensor(V)
        V= V.t()

        if M.is_cuda:
            U= U.cuda()
            V= V.cuda()
            S= S.cuda()

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("backward not implemented")
        U, S, V = self.saved_tensors
        return dA, None