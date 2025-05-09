import numpy as np
import torch
import torch.nn.functional as Functional
import logging
log = logging.getLogger(__name__)
try:
    import scipy.sparse.linalg
    from scipy.sparse.linalg import LinearOperator
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")
from linalg.svd_gesdd import safe_inverse, safe_inverse_2

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
        dA= None
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

class SVDARNOLDI(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, thresh, solver):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :param thresh: threshold for applying SVDARNOLDI instead of full SVD
        :param solver: solver for scipy.sparse.linalg.svds
        :type M: torch.Tensor
        :type k: int
        :type thresh: float
        :type solver: str
        :return: leading k left eigenvectors U, singular values S, and right
                 eigenvectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, by computing
        the symmetric decomposition of :math:`H=MM^\dagger` as :math:`H= UDU^\dagger`
        up to rank k. Partial eigendecomposition is done through Arnoldi method.
        """
        # input validation is provided by the scipy.sparse.linalg.eigsh /
        # scipy.sparse.linalg.svds

        # ----- Option 0
        # M_nograd = M.clone().detach()
        # MMt= M_nograd@M_nograd.t().conj()

        # def mv(v):
        #     B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
        #     B= torch.mv(MMt,B)
        #     return B.detach().cpu().numpy()

        # MMt_op= LinearOperator(M.size(), matvec=mv)

        # D, U= scipy.sparse.linalg.eigsh(MMt_op, k=k)
        # D= torch.as_tensor(D,device=M.device)
        # U= torch.as_tensor(U,device=M.device)

        # # reorder the eigenpairs by the largest magnitude of eigenvalues
        # S,p= torch.sort(torch.abs(D),descending=True)
        # S= torch.sqrt(S)
        # U= U[:,p]

        # # compute right singular vectors as Mt = V.S.Ut /.U => Mt.U = V.S
        # V = M_nograd.t().conj() @ U
        # V = Functional.normalize(V, p=2, dim=0)

        # ----- Option 1
        M_numpy = M.detach().cpu().numpy()
        if M.size(dim=0)*thresh <= k or M.size(dim=1)*thresh <= k:
            U, S, Vh = scipy.linalg.svd(M_numpy)
            U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
        else:
            U, S, Vh= scipy.sparse.linalg.svds(M_numpy, k=k, solver=solver, maxiter=k*10)

        S= torch.as_tensor(S.copy())
        U= torch.as_tensor(U.copy())
        Vh= torch.as_tensor(Vh.copy())

        self.save_for_backward(U, S, Vh)
        return U, S, Vh

    @staticmethod
    def backward(self, dU, dS, dV):
        r"""
        The backward is not implemented.
        """
        raise Exception("backward not implemented")
        U, S, V = self.saved_tensors
        dA= None
        return dA, None

def test_SVDARNOLDI_random():
    m= 50
    k= 10
    M= torch.rand(m, m, dtype=torch.float64)

    U0, S0, V0= torch.svd(M)

    U,S,V= SVDARNOLDI.apply(M,k)
    # |M|=\sqrt{Tr(MM^t)}=\sqrt{Tr(D^2)} =>
    # |M-US_kV^t|=\sqrt{Tr(D^2)-Tr(S^2)}=\sqrt{\sum_i>k D^2_i}
    assert( torch.norm(M-U@torch.diag(S)@V.t())-torch.sqrt(torch.sum(S0[k:]**2))
        < S0[0]*(m**2)*1e-14 )

def test_SVDARNOLDI_rank_deficient():
    m= 50
    k=15
    for r in [25,35,40,45]:
        M= torch.rand((m,m),dtype=torch.float64)
        U, S0, V= torch.svd(M)
        S0[-r:]=0
        M= U@torch.diag(S0)@V.t()

        U, S, V= SVDARNOLDI.apply(M, k)
        assert( torch.norm(M-U@torch.diag(S)@V.t())-torch.sqrt(torch.sum(S0[k:]**2))
            < S0[0]*(m**2)*1e-14 )

class SVD_PROPACK(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, k_extra, rel_cutoff, v0):
        r"""
        :param M: square matrix :math:`N \times N`
        :param k: desired rank (must be smaller than :math:`N`)
        :type M: torch.Tensor
        :type k: int
        :return: leading k left eigenvectors U, singular values S, and right 
                 eigenvectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        **Note:** `depends on scipy`

        Return leading k-singular triples of a matrix M, by computing 
        the partial SVD decomposition using SciPy's PROPACK wrapper up to rank k.
        """
        # input validation is provided by the scipy.sparse.linalg.eigsh / 
        # scipy.sparse.linalg.svds
        
        # reorder the eigenpairs by the largest magnitude of eigenvalues
        # S,p= torch.sort(torch.abs(D),descending=True)
        # S= torch.sqrt(S)
        # U= U[:,p]

        def mv(v):
            B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            B= torch.mv(M,B)
            return B.detach().cpu().numpy()
        def vm(v):
            B= torch.as_tensor(v,dtype=M.dtype,device=M.device)
            B= torch.einsum('i,ij->j',B,M)           
            return B.detach().cpu().numpy()

        M_nograd= LinearOperator(M.size(), matvec=mv, rmatvec=vm)
        U, S, Vh= scipy.sparse.linalg.svds(M_nograd, k=k+k_extra, v0=v0, solver='propack')

        S= torch.as_tensor(np.flip(S)).to(device=M.device)
        U= torch.as_tensor(np.flip(U,axis=1),dtype=M.dtype,device=M.device)
        Vh= torch.as_tensor(np.flip(Vh,axis=0),dtype=M.dtype,device=M.device)
        V= Vh.transpose(-2,-1).conj()

        self.save_for_backward(U, S, V, rel_cutoff)
        return U, S, V

    @staticmethod
    def backward(self, gu, gsigma, gv):
        r"""
        :param gu: gradient on U
        :type gu: torch.Tensor
        :param gsigma: gradient on S
        :type gsigma: torch.Tensor
        :param gv: gradient on V
        :type gv: torch.Tensor
        :return: gradient
        :rtype: torch.Tensor

        Computes backward gradient for SVD, adopted from 
        https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp
        
        For complex-valued input there is an additional term, see

            * https://giggleliu.github.io/2019/04/02/einsumbp.html
            * https://arxiv.org/abs/1909.02659

        The backward is regularized following
        
            * https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py
            * https://arxiv.org/abs/1903.09650

        using 

        .. math:: 
            S_i/(S^2_i-S^2_j) = (F_{ij}+G_{ij})/2\ \ \textrm{and}\ \ S_j/(S^2_i-S^2_j) = (F_{ij}-G_{ij})/2
        
        where 
        
        .. math:: 
            F_{ij}=1/(S_i-S_j),\ G_{ij}=1/(S_i+S_j)
        """
        # 
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");
        u, sigma, v, eps = self.saved_tensors
        m= u.size(0) # first dim of original tensor A = u sigma v^\dag 
        n= v.size(0) # second dim of A
        k= sigma.size(0)
        sigma_scale= sigma[0]

        # ? some
        if (u.size(-2)!=u.size(-1)) or (v.size(-2)!=v.size(-1)):
            # We ignore the free subspace here because possible base vectors cancel
            # each other, e.g., both -v and +v are valid base for a dimension.
            # Don't assume behavior of any particular implementation of svd.
            u = u.narrow(-1, 0, k)
            v = v.narrow(-1, 0, k)
            if not (gu is None): gu = gu.narrow(-1, 0, k)
            if not (gv is None): gv = gv.narrow(-1, 0, k)
        vh= v.conj().transpose(-2,-1)

        if not (gsigma is None):
            # computes u @ diag(gsigma) @ vh
            sigma_term = u * gsigma.unsqueeze(-2) @ vh
        else:
            sigma_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        # in case that there are no gu and gv, we can avoid the series of kernel
        # calls below
        if (gv is None) and (gv is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {dA.abs().max()} {S.max()}")
            return sigma_term, None, None

        sigma_inv= safe_inverse_2(sigma.clone(), sigma_scale*eps)

        F = sigma.unsqueeze(-2) - sigma.unsqueeze(-1)
        F = safe_inverse(F, sigma_scale*eps)
        F.diagonal(0,-2,-1).fill_(0)

        G = sigma.unsqueeze(-2) + sigma.unsqueeze(-1)
        G = safe_inverse(G, sigma_scale*eps)
        G.diagonal(0,-2,-1).fill_(0)

        uh= u.conj().transpose(-2,-1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1);
            u_term = u @ ( (F+G).mul( uh @ gu - guh @ u) ) * 0.5
            if m > k:
                # projection operator onto subspace orthogonal to span(U) defined as I - UU^H
                proj_on_ortho_u = -u @ uh
                proj_on_ortho_u.diagonal(0, -2, -1).add_(1);
                u_term = u_term + proj_on_ortho_u @ (gu * sigma_inv.unsqueeze(-2)) 
            u_term = u_term @ vh
        else:
            u_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        
        if not (gv is None):
            gvh = gv.conj().transpose(-2, -1);
            v_term = ( (F-G).mul(vh @ gv - gvh @ v) ) @ vh * 0.5
            if n > k:
                # projection operator onto subspace orthogonal to span(V) defined as I - VV^H
                proj_on_v_ortho =  -v @ vh
                proj_on_v_ortho.diagonal(0, -2, -1).add_(1);
                v_term = v_term + sigma_inv.unsqueeze(-1) * (gvh @ proj_on_v_ortho)
            v_term = u @ v_term
        else:
            v_term = torch.zeros(m,n,dtype=u.dtype,device=u.device)
        

        # // for complex-valued input there is an additional term
        # // https://giggleliu.github.io/2019/04/02/einsumbp.html
        # // https://arxiv.org/abs/1909.02659
        dA= u_term + sigma_term + v_term
        if u.is_complex() or v.is_complex():
            L= (uh @ gu).diagonal(0,-2,-1)
            L.real.zero_()
            L.imag.mul_(sigma_inv)
            imag_term= (u * L.unsqueeze(-2)) @ vh
            dA= dA + imag_term

        return dA, None, None, None, None


if __name__=='__main__':
    test_SVDSYMARNOLDI_random()
    test_SVDARNOLDI_random()
    test_SVDARNOLDI_rank_deficient()
