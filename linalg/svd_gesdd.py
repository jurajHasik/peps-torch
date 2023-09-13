import torch
import numpy as np
from config import _torch_version_check

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)
    
def safe_inverse_2(x, epsilon):
    x[abs(x)<epsilon]=float('inf')
    return x.pow(-1)

class SVDGESDD_legacy(torch.autograd.Function):
    @staticmethod
    def forward(self, A, cutoff, diagnostics):
        U, S, V = torch.svd(A)
        cutoff= torch.as_tensor(cutoff, dtype=S.dtype, device=S.device)
        self.save_for_backward(U, S, V, cutoff)
        return U, S, V
    
    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V, cutoff = self.saved_tensors
        
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = S.size(0)

        F = (S - S[:, None])
        # mask0= F==0
        F = safe_inverse(F, cutoff)
        F.diagonal().fill_(0)
        # F[mask0]= 0

        G = (S + S[:, None])
        G = safe_inverse(G)
        G.diagonal().fill_(0)

        UdU = Ut @ dU
        VdV = Vt @ dV

        # F_ij= 1/(S_i - S_j)
        # G_ij= 1/(S_i + S_j)
        # (F+G)_ij= 1/(S_i - S_j) - 1/(S_i + S_j) = ((S_i + S_j) - (S_i - S_j))/((S_i - S_j)*(S_i + S_j))
        #         = 2S_j / (S^2_i - S^2_j) 
        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*safe_inverse(S)) @ Vt 
        if (N>NS):
            dA = dA + (U*safe_inverse(S)) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)

        return dA, None, None

class SVDGESDD(torch.autograd.Function):
    if _torch_version_check("1.8.1"):
        @staticmethod
        def forward(self, A, cutoff, diagnostics):
            r"""
            :param A: rank-2 tensor
            :type A: torch.Tensor
            :param cutoff: cutoff for backward function
            :type cutoff: torch.Tensor
            :param diagnostics: optional dictionary for debugging purposes
            :type diagnostics: dict
            :return: U, S, V
            :rtype: torch.Tensor, torch.Tensor, torch.Tensor

            Computes SVD decompostion of matrix :math:`A = USV^\dagger`.
            """
            # A = U @ diag(S) @ Vh
            U, S, Vh = torch.linalg.svd(A)
            V= Vh.transpose(-2,-1).conj()
            self.save_for_backward(U, S, V, cutoff)
            self.diagnostics= diagnostics
            return U, S, V
    else:
        @staticmethod
        def forward(self, A, cutoff, diagnostics):
            r"""
            :param A: rank-2 tensor
            :type A: torch.Tensor
            :param cutoff: cutoff for backward function
            :type cutoff: torch.Tensor
            :param diagnostics: optional dictionary for debugging purposes
            :type diagnostics: dict
            :return: U, S, V
            :rtype: torch.Tensor, torch.Tensor, torch.Tensor

            Computes SVD decompostion of matrix :math:`A = USV^\dagger`.
            """
            U, S, V = torch.svd(A)
            self.diagnostics= diagnostics
            self.save_for_backward(U, S, V, cutoff)
            return U, S, V

    @staticmethod
    def v1_10_Fonly_backward(self, gu, gsigma, gv):
        # Adopted from
        # https://github.com/pytorch/pytorch/blob/v1.10.2/torch/csrc/autograd/FunctionsManual.cpp
        # 
        # TORCH_CHECK(compute_uv,
        #    "svd_backward: Setting compute_uv to false in torch.svd doesn't compute singular matrices, ",
        #    "and hence we cannot compute backward. Please use torch.svd(compute_uv=True)");

        diagnostics= self.diagnostics
        u, sigma, v, eps = self.saved_tensors
        m= u.size(-2) # first dim of original tensor A = u sigma v^\dag 
        n= v.size(-2) # second dim of A
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
        sigma_sq= sigma.pow(2)
        F= sigma_sq.unsqueeze(-2) - sigma_sq.unsqueeze(-1)

        # F_ij = 1/(S^2_i - S^2_j)
        # // The following two lines invert values of F, and fills the diagonal with 0s.
        # // Notice that F currently has 0s on diagonal. So we fill diagonal with +inf
        # // first to prevent nan from appearing in backward of this function.
        F.diagonal(0,-2,-1).fill_(float('inf'))
        F= safe_inverse_2(F, sigma_scale*eps)

        uh= u.conj().transpose(-2,-1)
        if not (gu is None):
            guh = gu.conj().transpose(-2, -1);
            u_term = u @ (F.mul( uh @ gu - guh @ u) * sigma.unsqueeze(-2))
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
            v_term = sigma.unsqueeze(-1) * (F.mul(vh @ gv - gvh @ v) @ vh)
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

        if not (diagnostics is None):
            print(f"{diagnostics} {dA.abs().max()} {S.max()}")
        return dA, None, None

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

        diagnostics= self.diagnostics
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

        if not (diagnostics is None):
            print(f"{diagnostics} {dA.abs().max()} {sigma.max()}")
        return dA, None, None

    # From https://github.com/pytorch/pytorch/blob/master/torch/csrc/autograd/FunctionsManual.cpp
    # commit 5375b2e (v1.11.+)
    # Tensor svd_backward(const Tensor& gU,
    #                     const Tensor& gS,
    #                     const Tensor& gVh,
    #                     const Tensor& U,
    #                     const Tensor& S,
    #                     const Tensor& Vh) {
    #   at::NoTF32Guard disable_tf32;
    @staticmethod
    def v1_11_backward(self, gU, gS, gVh):
        U, S, Vh, ad_decomp_reg = self.saved_tensors
        diagnostics = self.diagnostics
        #   // Throughout both the real and complex case we assume A has distinct singular values.
        #   // Furthermore, if A is rectangular or complex, we assume it's full-rank.
        #   //
        #   //
        #   // The real case (A \in R)
        #   // See e.g. https://j-towns.github.io/papers/svd-derivative.pdf
        #   //
        #   // Denote by skew(X) = X - X^T, and by A o B the coordinatewise product, then
        #   // if m == n
        #   //   gA = U [(skew(U^T gU) / E)S + S(skew(V^T gV) / E) + I o gS ]V^T
        #   // where E_{jk} = S_k^2 - S_j^2 if j != k and 1 otherwise
        #   //
        #   // if m > n
        #   //   gA = [term in m == n] + (I_m - UU^T)gU S^{-1} V^T
        #   // if m < n
        #   //   gA = [term in m == n] + U S^{-1} (gV)^T (I_n - VV^T)
        #   //
        #   //
        #   // The complex case (A \in C)
        #   // This one is trickier because the svd is not locally unique.
        #   // Denote L = diag(e^{i\theta_k}), then we have that if A = USV^H, then (UL, S, VL) is
        #   // another valid SVD decomposition of A as
        #   // A = ULS(VL)^H = ULSL^{-1}V^H = USV^H,
        #   // since L, S and L^{-1} commute, since they are all diagonal.
        #   //
        #   // Assume wlog that n >= k in what follows, as otherwise we could reason about A^H.
        #   // Denote by St_k(C^n) = {A \in C^{n,k} | A^H A = I_k} the complex Stiefel manifold.
        #   // What this invariance means is that the svd decomposition is not a map
        #   // svd: C^{n x k} -> St_k(C^n) x R^n x St_k(C^k)
        #   // (where St_k(C^k) is simply the unitary group U(k)) but a map
        #   // svd: C^{n x k} -> M x R^n
        #   // where M is the manifold given by quotienting St_k(C^n) x U(n) by the action (U, V) -> (UL, VL)
        #   // with L as above.
        #   // Note that M is a manifold, because the action is free and proper (as U(1)^k \iso (S^1)^k is compact).
        #   // For this reason, pi : St_k(C^n) x U(n) -> M forms a principal bundle.
        #   //
        #   // To think about M, consider the case case k = 1. The, we have the bundle
        #   // pi : St_1(C^n) x U(1) -> M
        #   // now, St_1(C^n) are just vectors of norm 1 in C^n. That's exactly the sphere of dimension 2n-1 in C^n \iso R^{2n}
        #   // S^{2n-1} = { z \in C^n | z^H z = 1}.
        #   // Then, in this case, we're quotienting out U(1) completely, so we get that
        #   // pi : S^{2n-1} x U(1) -> CP(n-1)
        #   // where CP(n-1) is the complex projective space of dimension n-1.
        #   // In other words, M is just the complex projective space, and pi is (pretty similar to)
        #   // the usual principal bundle from S^{2n-1} to CP(n-1).
        #   // The case k > 1 is the same, but requiring a linear inependence condition between the
        #   // vectors from the different S^{2n-1} or CP(n-1).
        #   //
        #   // Note that this is a U(1)^k-bundle. In plain words, this means that the fibres of this bundle,
        #   // i.e. pi^{-1}(x) for x \in M are isomorphic to U(1) x ... x U(1).
        #   // This is obvious as, if pi(U,V) = x,
        #   // pi^{-1}(x) = {(U diag(e^{i\theta}), V diag(e^{i\theta})) | \theta \in R^k}
        #   //            = {(U diag(z), V diag(z)) | z \in U(1)^k}
        #   // since U(1) = {z \in C | |z| = 1}.
        #   //
        #   // The big issue here is that M with its induced metric is not locally isometric to St_k(C^n) x U(k).
        #   // [The why is rather technical, but you can see that the horizontal distribution is not involutive,
        #   // and hence integrable due to Frobenius' theorem]
        #   // What this means in plain words is that, no matter how we choose to return the U and V from the
        #   // SVD, we won't be able to simply differentiate wrt. U and V and call it a day.
        #   // An example of a case where we can do this is when performing an eigendecomposition on a real
        #   // matrix that happens to have real eigendecomposition. In this case, even though you can rescale
        #   // the eigenvectors by any real number, you can choose them of norm 1 and call it a day.
        #   // In the eigenvector case, we are using that you can isometrically embed S^{n-1} into R^n.
        #   // In the svd case, we need to work with the "quotient manifold" M explicitly, which is
        #   // slightly more technically challenging.
        #   //
        #   // Since the columns of U and V are not uniquely defined, but are representatives of certain
        #   // classes of equivalence which represent elements M, the user may not depend on the particular
        #   // representative that we return from the SVD. In particular, if the loss function depends on U
        #   // or V, it must be invariant under the transformation (U, V) -> (UL, VL) with
        #   // L = diag(e^{i\theta})), for every \theta \in R^k.
        #   // In more geometrical terms, this means that the loss function should be constant on the fibres,
        #   // or, in other words, the gradient along the fibres should be zero.
        #   // We may see this by checking that the gradients as element in the tangent space
        #   // T_{(U, V)}(St(n,k) x U(k)) are normal to the fibres. Differentiating the map
        #   // (U, V) -> (UL, VL), we see that the space tangent to the fibres is given by
        #   // Vert_{(U, V)}(St(n,k) x U(k)) = { i[U, V]diag(\theta) | \theta in R^k}
        #   // where [U, V] denotes the vertical concatenation of U and V to form an (n+k, k) matrix.
        #   // Then, solving
        #   // <i[U,V]diag(\theta), [S, T]> = 0 for two matrices S, T \in T_{(U, V)}(St(n,k) x U(k))
        #   // where <A, B> = Re tr(A^H B) is the canonical (real) inner product in C^{n x k}
        #   // we get that the function is invariant under action of U(1)^k iff
        #   // Im(diag(U^H gU + V^H gV)) = 0
        #   //
        #   // Using this in the derviaton for the forward AD, one sees that, with the notation from those notes
        #   // Using this and writing sym(X) = X + X^H, we get that the forward AD for SVD in the complex
        #   // case is given by
        #   // dU = U (sym(dX S) / E + i Im(diag(dX)) / (2S))
        #   // if m > n
        #   //   dU = [dU for m == n] + (I_m - UU^H) dA V S^{-1}
        #   // dS = Re(diag(dP))
        #   // dV = V (sym(S dX) / E - i Im(diag(dX)) / (2S))
        #   // if m < n
        #   //   dV = [dV for m == n] + (I_n - VV^H) (dA)^H U S^{-1}
        #   // dVh = dV^H
        #   // with dP = U^H dA V
        #   //      dX = dP - dS
        #   //      E_{jk} = S_k^2 - S_j^2 if j != k
        #   //               1             otherwise
        #   //
        #   // Similarly, writing skew(X) = X - X^H
        #   // the adjoint wrt. the canonical metric is given by
        #   // if m == n
        #   //   gA = U [((skew(U^H gU) / E) S + i Im(diag(U^H gU)) / S + S ((skew(V^H gV) / E)) + I o gS] V^H
        #   // if m > n
        #   //   gA = [term in m == n] + (I_m - UU^H)gU S^{-1} V^H
        #   // if m < n
        #   //   gA = [term in m == n] + U S^{-1} (gV)^H (I_n - VV^H)
        #   // where we have used that Im(diag(U^H gU)) = - Im(diag(V^h gV)) to group the diagonal imaginary terms into one
        #   // that just depends on U^H gU.

        #   // Checks compute_uv=true
        #   TORCH_INTERNAL_ASSERT(U.dim() >= 2 && Vh.dim() >= 2);

        #   // Trivial case
        #   if (!gS.defined() && !gU.defined() && !gVh.defined()) {
        #     return {};
        #   }

        m = U.size(-2)
        n = Vh.size(-1)

        #   // Optimisation for svdvals: gA = U @ diag(gS) @ Vh
        if (gU is None) and (gVh is None):
            if not (diagnostics is None):
                print(f"{diagnostics} {gA.size()} {gA.abs().max()} {S.max()}")
            return U @ (gS.unsqueeze(-1) * Vh) if m>=n else (U * gS.unsqueeze(-2)) @ Vh
        #   // At this point, at least one of gU, gVh is defined

        is_complex = U.is_complex()
        def skew(A): return A - A.transpose(-2, -1).conj()
        #   const auto UhgU = gU.defined() ? skew(at::matmul(U.mH(), gU)) : Tensor{};
        #   const auto VhgV = gVh.defined() ? skew(at::matmul(Vh, gVh.mH())) : Tensor{};
        UhgU= skew( U.transpose(-2, -1).conj()@gU )
        VhgV= skew( Vh@gVh.transpose(-2, -1).conj() )

        #   // Check for the invariance of the loss function, i.e.
        #   // Im(diag(U^H gU)) + Im(diag(V^H gV)) = 0
        #   if (is_complex) {
        #     const auto imdiag_UhgU = gU.defined() ? at::imag(UhgU.diagonal(0, -2, -1))
        #                                           : at::zeros_like(S);
        #     const auto imdiag_VhgV = gVh.defined() ? at::imag(VhgV.diagonal(0, -2, -1))
        #                                            : at::zeros_like(S);
        #     // Rather lax atol and rtol, as we don't want false positives
        #     TORCH_CHECK(at::allclose(imdiag_UhgU, -imdiag_VhgV, /*rtol=*/1e-2, /*atol=*/1e-2),
        #                 "svd_backward: The singular vectors in the complex case are specified up to multiplication "
        #                 "by e^{i phi}. The specified loss function depends on this phase term, making "
        #                 "it ill-defined.");
        #   }
        if is_complex:
            imdiag_UhgU= UhgU.diagonal(0, -2, -1).imag
            imdiag_VhgV= VhgV.diagonal(0, -2, -1).imag
            if not torch.allclose( imdiag_UhgU, -imdiag_VhgV, 1e-2, 1e-2 ):
                warnings.warn("svd_backward: The singular vectors in the complex case are "\
                +"specified up to multiplication by e^{i phi}. The specified loss function depends on "\
                +"this phase term, making it ill-defined.",RuntimeWarning)
                # import pdb; pdb.set_trace()

        #   // gA = ((U^H gU) / E) S +  S (((V^H gV) / E) + I o (gS + diag(U^H gU) / (2 * S))
        #   Tensor gA = [&] {
        #     // ret holds everything but the diagonal of gA
        #     auto ret = [&] {
        #       const auto E = [&S]{
        #         const auto S2 = S * S;
        #         auto ret = S2.unsqueeze(-2) - S2.unsqueeze(-1);
        #         // Any number a != 0 would, as we are just going to use it to compute 0 / a later on
        #         ret.diagonal(0, -2, -1).fill_(1);
        #         return ret;
        #       }();

        #       if (gU.defined()) {
        #         if (gVh.defined()) {
        #           return (UhgU * S.unsqueeze(-2) + S.unsqueeze(-1) * VhgV) / E;
        #         } else {
        #           return (UhgU / E) * S.unsqueeze(-2);
        #         }
        #       } else { // gVh.defined();
        #         return S.unsqueeze(-1) * (VhgV / E);
        #       }
        #     }();
        #     // Fill the diagonal
        #     if (gS.defined()) {
        #       ret = ret + gS.diag_embed();
        #     }
        #     if (is_complex && gU.defined() && gVh.defined()) {
        #       ret = ret + (UhgU.diagonal(0, -2, -1) / (2. * S)).diag_embed();
        #     }
        #     return ret;
        #   }();
        def reg_preinv(x):
            x_reg= x.clone()
            x_scale= x.abs().max()
            if x_scale<ad_decomp_reg:
                x_reg= float('inf')
            else:
                x_reg[abs(x_reg/x_scale) < ad_decomp_reg] = float('inf')
            return x_reg

        S2= S*S
        E= S2.unsqueeze(-2) - S2.unsqueeze(-1) # S^2_i-S^2_j
        E.diagonal(0,-2,-1).fill_(1)
        
        gA= (UhgU * S.unsqueeze(-2) + S.unsqueeze(-1) * VhgV) * (1./reg_preinv(E)) + gS.diag_embed()
        if is_complex:
            gA = gA + (UhgU.diagonal(0, -2, -1) * (1./(2. * reg_preinv(S))) ).diag_embed()

        #   if (m > n && gU.defined()) {
        #     // gA = [UgA + (I_m - UU^H)gU S^{-1}]V^H
        #     gA  = at::matmul(U, gA);
        #     const auto gUSinv = gU / S.unsqueeze(-2);
        #     gA = gA + gUSinv - at::matmul(U, at::matmul(U.mH(), gUSinv));
        #     gA = at::matmul(gA, Vh);
        #   } else if (m < n && gVh.defined()) {
        #     //   gA = U[gA V^H + S^{-1} (gV)^H (I_n - VV^H)]
        #     gA = at::matmul(gA, Vh);
        #     const auto SinvgVh = gVh / S.unsqueeze(-1);
        #     gA = gA + SinvgVh - at::matmul(at::matmul(SinvgVh, Vh.mH()), Vh);
        #     gA = at::matmul(U, gA);
        #   } else {
        #     // gA = U gA V^H
        #     gA = m >= n ? at::matmul(U, at::matmul(gA, Vh))
        #                 : at::matmul(at::matmul(U, gA), Vh);
        #   }

        # TODO regularize 1/S
        if m>n:
            gA = U @ gA
            gUSinv = gU / S.unsqueeze(-2)
            gA = gA + gUSinv - U @ (U.transpose(-2, -1).conj() @ gUSinv)
            gA = gA @ Vh
        elif m<n:
            gA = gA @ Vh
            SinvgVh = gVh / S.unsqueeze(-1)
            gA = gA + SinvgVh - (SinvgVh @ Vh.transpose(-2, -1).conj()) @ Vh
            gA = U @ gA
        else:
            gA = U @ (gA @ Vh) if m>=n else (U @ gA) @ Vh

        #   return gA;
        # }
        if not (diagnostics is None):
            print(f"{diagnostics} {gA.size()} {gA.abs().max()} {S.max()}")
        return gA, None, None, None

def test_SVDGESDD_legacy_random():
    eps= 1.0e-12
    eps= torch.as_tensor(eps, dtype=torch.float64)

    # M, N = 50, 40
    # A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    # assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1e-5))

    # M, N = 40, 40
    # A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    # assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1e-5))

    M= 50
    A= torch.rand(M, M, dtype=torch.float64)
    A= 0.5*(A+A.t())

    D, U = torch.symeig(A, eigenvectors=True)
    # make random spectrum with almost degen
    for split_scale in [0.]: # 10.0, 1.0, 0.1, 0.01, 
        tot_scale=1000
        d0= torch.rand(M//2, dtype=torch.float64)
        splits= torch.rand(M//2, dtype=torch.float64)
        for i in range(M//2):
            D[2*i]= tot_scale*d0[i]
            D[2*i+1]= tot_scale*d0[i]+split_scale*splits[i]
        A= U.t() @ torch.diag(D) @ U
        print(f"split_scale {split_scale} {D}")

        try:
            A.requires_grad_()
            assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1.0, rtol=1.0e-3))
        except Exception as e:
            print(f"FAILED for splits: {split_scale}")
            print(e)

def test_SVDGESDD_random():
    eps= 1.0e-12
    eps= torch.as_tensor(eps, dtype=torch.float64)

    M, N = 50, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1e-5))

    M, N = 40, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1e-5))

    M= 50
    A= torch.rand(M, M, dtype=torch.float64)
    A= 0.5*(A+A.t())

    D, U = torch.symeig(A, eigenvectors=True)
    # make random spectrum with almost degen
    for split_scale in [10.0, 1.0, 0.1, 0.01, 0.]: 
        tot_scale=1000
        d0= torch.rand(M//2, dtype=torch.float64)
        splits= torch.rand(M//2, dtype=torch.float64)
        for i in range(M//2):
            D[2*i]= tot_scale*d0[i]
            D[2*i+1]= tot_scale*d0[i]+split_scale*splits[i]
        A= U.t() @ torch.diag(D) @ U
        print(f"split_scale {split_scale} {D}")

        try:
            A.requires_grad_()
            assert(torch.autograd.gradcheck(SVDGESDD_legacy.apply, (A, eps, None,) , eps=1e-6, atol=1.0, rtol=1.0e-3))
        except Exception as e:
            print(f"FAILED for splits: {split_scale}")
            print(e)

def test_SVDGESDD_COMPLEX_random():
    
    def test_f_1(M):
        U,S,V= SVDGESDD.apply(M)
        return torch.sum(S[0:1])

    def test_f_2(M):
        U,S,V= SVDGESDD.apply(M)
        T= U @ V.conj().transpose(-2,-1)
        return T.norm()

    M= 25
    A= torch.rand((M, M), dtype=torch.complex128)
    U,S,V= torch.svd(A) 
    print(S)

    for split_scale in [10.0, 1.0, 0.1, 0.01, 0.]: 
        tot_scale=1000
        d0= torch.rand(M//2, dtype=torch.float64)
        splits= torch.rand(M//2, dtype=torch.float64)
        for i in range(M//2):
            S[2*i]= tot_scale*d0[i]
            S[2*i+1]= tot_scale*d0[i]+split_scale*splits[i]
        A= U * torch.diag(S) @ V.conj().transpose(-2,-1)
        A.requires_grad_()
        print(f"split_scale {split_scale}")
        print(S)

        assert(torch.autograd.gradcheck(test_f_1, A, eps=1e-6, atol=1e-4))
        assert(torch.autograd.gradcheck(test_f_2, A, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    test_SVDGESDD_legacy_random()
    test_SVDGESDD_random()
    # test_SVDGESDD_COMPLEX_random()