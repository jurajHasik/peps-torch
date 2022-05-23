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
        :rtype: torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-eigenpairs of a matrix M, where M is symmetric 
        :math:`M=M^\dagger`, by computing the symmetric decomposition :math:`M= UDU^\dagger` 
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
            U= U.to(M.device)
            D= D.to(M.device)

        self.save_for_backward(D, U)
        return D, U

    @staticmethod
    def backward(self, dD, dU):
        r"""
        The backward is not implemented.
        """
        raise Exception("backward not implemented")
        D, U= self.saved_tensors
        dA= None
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

class ARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, v0, dtype, device):
        r"""
        :param M_op: numpy linear operator with defined matrix-vector multiplication
                     function mv(v)->v or torch.Tensor
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :param v0: initial vector
        :type M_op: scipy.sparse.linalg.LinearOperator or torch.Tensor
        :type M: torch.Tensor
        :type k: int
        :type v0: torch.Tensor
        :return: leading k eigenvalues D and (left) eigenvectors U 
        :rtype: torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-eigenpairs of a matrix M by solving MU=DU problem.
        """
        # input validation is provided by the scipy.sparse.linalg.eigsh / 
        # scipy.sparse.linalg.svds
        
        # ----- Option 0
        if isinstance(M, torch.Tensor):
            # assume M is a matrix and build LinearOperator
            dtype=M.dtype
            device=M.device
            M_nograd = M.clone().detach()
            def mv(v):
                B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
                B= torch.mv(M_nograd,B)
                return B.detach().cpu().numpy()
            M_op= LinearOperator(M.size(), matvec=mv)
        else:
            # otherwise we assume, that M is LinearOperator itself
            assert dtype is not None,"missing dtype for LinearOperator M"
            assert device is not None,"missing device for LinearOperator M"
            M_op=M

        if v0 is not None:
            v0_nograd= v0.detach().cpu().numpy()
        else:
            v0_nograd= None

        D_, U_= scipy.sparse.linalg.eigs(M_op, k=k, v0=v0_nograd)
        # here D, U are complex numpy tensors
        D= torch.zeros((k,2), dtype=dtype, device=device)
        D[:,0]= torch.as_tensor(np.real(D_),device=device)
        D[:,1]= torch.as_tensor(np.imag(D_),device=device)
        U= torch.zeros((M_op.shape[0],k,2), dtype=dtype, device=device)
        U[:,:,0]= torch.as_tensor(np.real(U_),device=device)
        U[:,:,1]= torch.as_tensor(np.imag(U_),device=device)

        # reorder the eigenpairs by the largest magnitude of eigenvalues
        Dabs= torch.norm(D,dim=1)
        Dabs,p= torch.sort(Dabs,descending=True)
        D= D[p,:]
        U= U[:,p,:]

        if device.type==torch.device('cuda').type:
            U= U.to(device)
            D= D.to(device)

        self.save_for_backward(D, U)
        return D, U

    @staticmethod
    def backward(self, dD, dU):
        r"""
        The backward is not implemented.
        """
        raise Exception("backward not implemented")
        D, U = self.saved_tensors
        dA= None
        return dA, None, None, None, None

def test_ARNOLDI_random():
    m= 50
    k= 5
    M= torch.rand(m, m, dtype=torch.float64)

    D0, U0= torch.eig(M, eigenvectors=True)
    D0abs= torch.norm(D0,dim=1)
    D0abs,p= torch.sort(D0abs,descending=True)
    D0= D0[p,:]
    U0= U0[:,p]

    D, U= ARNOLDI.apply(M,k)
    
    # verify spectrum
    assert( torch.norm(D0[:k,0]-D[:,0])/(D0abs[0]*k)<1.0e-14 )
    assert( torch.norm(torch.abs(D0[:k,1])-torch.abs(D[:,1]))/(D0abs[0]*k)<1.0e-14 )

if __name__=='__main__':
    test_SYMARNOLDI_random()
    test_ARNOLDI_random()
