import torch

class SYMLOBPCG(torch.autograd.Function):
    @staticmethod
    def forward(self, M, k, tol):
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
        up to rank k. Partial eigendecomposition is done through LOBPCG method.
        """
        # (optional) verify hermicity
        M_asymm_norm= torch.norm(M-M.t())
        assert M_asymm_norm/torch.abs(M).max() < 1.0e-8, "M is not symmetric"

        # X (tensor, optional) - the input tensor of size (∗,m,n) where k <= n <= m. 
        #                        When specified, it is used as initial approximation of eigenvectors. 
        #                        X must be a dense tensor.
        # iK (tensor, optional) - the input tensor of size (∗,m,m). When specified, it will be used 
        #                         as preconditioner.
        # tol (float, optional) - residual tolerance for stopping criterion. Default is 
        #                         feps ** 0.5 where feps is smallest non-zero floating-point 
        #                         number of the given input tensor A data type.
        # niter (int, optional) - maximum number of iterations. When reached, the iteration 
        #                         process is hard-stopped and the current approximation of eigenpairs 
        #                         is returned. For infinite iteration but until convergence criteria is met, 
        #                         use -1.
        # tracker (callable, optional) - a function for tracing the iteration process. When specified,  
        #                                it is called at each iteration step with LOBPCG instance as an argument.
        #                                The LOBPCG instance holds the full state of the iteration process in 
        #                                the following attributes:
        #     iparams, fparams, bparams - dictionaries of integer, float, and boolean valued input parameters, 
        #                                 respectively
        #     ivars, fvars, bvars, tvars - dictionaries of integer, float, boolean, and Tensor valued 
        #                                  iteration variables, respectively.
        #     A, B, iK - input Tensor arguments.
        #     E, X, S, R - iteration Tensor variables.
        # ortho_fparams, ortho_bparams (ortho_iparams,) - various parameters to LOBPCG algorithm when 
        #                                                 using (default) method=”ortho”.
        D, U= torch.lobpcg(M, k=k, \
            X=None, n=None, iK=None, largest=True, tol=tol, niter=None, \
            tracker=None, ortho_iparams=None, ortho_fparams=None, ortho_bparams=None)

        self.save_for_backward(D, U)
        return D, U

    @staticmethod
    def backward(self, dD, dU):
        raise Exception("backward not implemented")
        D, U= self.saved_tensors
        dA= None
        return dA, None, None, None