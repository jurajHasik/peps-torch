'''
PyTorch has its own implementation of backward function for SVD at 
https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577 
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class RSVD(torch.autograd.Function):
    r"""
    :param M: input matrix
    :param k: desired rank
    :param p: oversampling rank. Total rank sampled ``k+p``
    :param q: number of matrix-vector multiplications for power scheme
    :param s: reorthogonalization 
    :param vnum: ?
    :type M: torch.tensor 
    :type k: int
    :type p: int
    :type q: int
    :type s: int
    :type vnum: int
    :return: U, S, V
    :rtype: tuple(torch.tensor, torch.tensor, torch.tensor)

    Performs approximate truncated singular value decomposition
    using randomized sampling. Based on the implementation in

    https://arxiv.org/abs/1502.05366
    """

    @staticmethod
    def forward(self, M, k, p = 20, q = 2, s = 1, vnum = 1):
        # get dims - torch.size() -> [rows, columns]
        m = list(M.size())[0] 
        n = list(M.size())[1] 
        r = min(m,n)

        # setup mats
        l = k + p
        U = torch.zeros((m,l), dtype=M.dtype, device=M.device)
        S = torch.zeros(l, dtype=M.dtype, device=M.device)
        V = torch.zeros((n,l), dtype=M.dtype, device=M.device)

        # build random matrix - with entries drawn from normal distribution
        RN = torch.randn((n, l), dtype=M.dtype, device=M.device)

        # multiply RN by M to get matrix of random samples Y
        Y = torch.mm(M, RN)

        # now build up (M M^T)^q R
        Z = torch.zeros((n,l), dtype=M.dtype, device=M.device)
        Yorth = torch.zeros((m,l), dtype=M.dtype, device=M.device)
        Zorth = torch.zeros((n,l), dtype=M.dtype, device=M.device)

        for j in range(1,q):
            # printf("M M^T mult j=%d of %d\n", j, q-1);

            if (2*j-2) % s == 0:
                # printf("orthogonalize Y..\n");
                Yorth, r = torch.qr(Y)
                # printf("Z = M'*Yorth..\n");
                Z = torch.mm(M.transpose(0,1),Yorth)
            else:
                # printf("Z = M'*Y..\n");
                Z = torch.mm(M.transpose(0,1),Y)

        
            if (2*j-1) % s == 0:
                # printf("orthogonalize Z..\n");
                Zorth, r = torch.qr(Z)
                # printf("Y = M*Zorth..\n");
                Y = torch.mm(M,Zorth)
            else:
                # printf("Y = M*Z..\n");
                Y = torch.mm(M,Z)

        # orthogonalize on exit from loop to get Q = m x l
        Q, r = torch.qr(Y)

        # either QR of B^T method, or eigendecompose BB^T method
        if (vnum == 1 or vnum > 2):
            # printf("using QR of B^T method\n");
            
            # form Bt = Mt*Q : nxm * mxl = nxl
            # printf("form Bt..\n");
            Bt = torch.mm(M.transpose(0,1),Q);

            # compute QR factorization of Bt    
            # M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
            # printf("doing QR..\n");
            ### compact_QR_factorization(Bt,Qhat,Rhat);
            Qhat, Rhat = torch.qr(Bt)

            # compute SVD of Rhat (lxl)
            # printf("doing SVD..\n");
            # torch.svd M -> U, S, V such that M = U S V^T
            Uhat, S, Vhat_trans = torch.svd(Rhat)
            Vhat_trans = Vhat_trans.transpose(0,1)

            # U = Q*Vhat_trans
            # printf("form U..\n");
            U = torch.mm(Q,Vhat_trans.transpose(0,1))

            # V = Qhat*Uhat
            # printf("form V..\n");
            V = torch.mm(Qhat,Uhat)

            # resize matrices to rank k from beginning
            # printf("resize mats\n");
            # U m x l -> m x k
            U = U[:, :k]
            # V m x l -> n x k
            V = V[:, :k]
            # S l x l -> k x k
            S = S[:k]
        # else:
        #     printf("using eigendecomposition of B B^T method\n");
        #     // build the matrix B B^T = Q^T M M^T Q column by column 
        #     // Bt = M^T Q ; nxm * mxk = nxk
        #     printf("form BBt..\n");
        #     mat *B = matrix_new(l,n);
        #     matrix_transpose_matrix_mult(Q,M,B);

        #     mat *BBt = matrix_new(l,l);
        #     matrix_matrix_transpose_mult(B,B,BBt);    

        #     // compute eigendecomposition of BBt
        #     printf("eigendecompose BBt..\n");
        #     vec *evals = vector_new(l);
        #     mat *Uhat = matrix_new(l,l);
        #     matrix_copy_symmetric(Uhat,BBt);
        #     compute_evals_and_evecs_of_symm_matrix(Uhat, evals);


        #     // compute singular values and matrix Sigma
        #     printf("form S..\n");
        #     vec *singvals = vector_new(l);
        #     for(i=0; i<l; i++){
        #         vector_set_element(singvals,i,std::sqrt(vector_get_element(evals,i)));
        #     }
        #     initialize_diagonal_matrix(*S, singvals);
            
        #     // compute U = Q*Uhat mxk * kxk = mxk  
        #     printf("form U..\n");
        #     matrix_matrix_mult(Q,Uhat,*U);

        #     // compute nxk V 
        #     // V = B^T Uhat * Sigma^{-1}
        #     printf("form V..\n");
        #     mat *Sinv = matrix_new(l,l);
        #     mat *UhatSinv = matrix_new(l,l);
        #     invert_diagonal_matrix(Sinv,*S);
        #     matrix_matrix_mult(Uhat,Sinv,UhatSinv);
        #     matrix_transpose_matrix_mult(B,UhatSinv,*V);

        #     matrix_delete(BBt);
        #     matrix_delete(Sinv);
        #     matrix_delete(UhatSinv);

        #     // resize matrices to rank k from end
        #     printf("resize mats to rank %d from end\n",k);
        #     //resize_matrix_by_columns_from_end(U,k);
        #     //resize_matrix_by_columns_from_end(V,k);
        #     resize_matrix_by_columns_from_end(S,k);
        #     resize_matrix_by_rows_from_end(S,k);
        
        self.save_for_backward(U, S, V)
        return U, S, V

    # TODO This is not a strictly correct backward
    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G 

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA, None

# TODO Backward not working, due to no backward implementation for QR
#      in pytorch
def rsvd(M, k, p = 20, q = 2, s = 1, vnum = 1):
    # get dims - torch.size() -> [rows, columns]
    m = list(M.size())[0] 
    n = list(M.size())[1] 
    r = min(m,n)

    # setup mats
    l = k + p
    U = torch.zeros((m,l), dtype=M.dtype, device=M.device)
    S = torch.zeros(l, dtype=M.dtype, device=M.device)
    V = torch.zeros((n,l), dtype=M.dtype, device=M.device)

    # build random matrix - with entries drawn from normal distribution
    RN = torch.randn((n, l), dtype=M.dtype, device=M.device)

    # multiply RN by M to get matrix of random samples Y
    Y = torch.mm(M, RN)

    # now build up (M M^T)^q R
    Z = torch.zeros((n,l), dtype=M.dtype, device=M.device)
    Yorth = torch.zeros((m,l), dtype=M.dtype, device=M.device)
    Zorth = torch.zeros((n,l), dtype=M.dtype, device=M.device)

    for j in range(1,q):
        # printf("M M^T mult j=%d of %d\n", j, q-1);

        if (2*j-2) % s == 0:
            # printf("orthogonalize Y..\n");
            Yorth, r = torch.qr(Y)
            # printf("Z = M'*Yorth..\n");
            Z = torch.mm(M.transpose(0,1),Yorth)
        else:
            # printf("Z = M'*Y..\n");
            Z = torch.mm(M.transpose(0,1),Y)

    
        if (2*j-1) % s == 0:
            # printf("orthogonalize Z..\n");
            Zorth, r = torch.qr(Z)
            # printf("Y = M*Zorth..\n");
            Y = torch.mm(M,Zorth)
        else:
            # printf("Y = M*Z..\n");
            Y = torch.mm(M,Z)

    # orthogonalize on exit from loop to get Q = m x l
    Q, r = torch.qr(Y)

    # either QR of B^T method, or eigendecompose BB^T method
    if (vnum == 1 or vnum > 2):
        # printf("using QR of B^T method\n");
        
        # form Bt = Mt*Q : nxm * mxl = nxl
        # printf("form Bt..\n");
        Bt = torch.mm(M.transpose(0,1),Q);

        # compute QR factorization of Bt    
        # M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
        # printf("doing QR..\n");
        ### compact_QR_factorization(Bt,Qhat,Rhat);
        Qhat, Rhat = torch.qr(Bt)

        # compute SVD of Rhat (lxl)
        # printf("doing SVD..\n");
        # torch.svd M -> U, S, V such that M = U S V^T
        Uhat, S, Vhat_trans = torch.svd(Rhat)
        Vhat_trans = Vhat_trans.transpose(0,1)

        # U = Q*Vhat_trans
        # printf("form U..\n");
        U = torch.mm(Q,Vhat_trans.transpose(0,1))

        # V = Qhat*Uhat
        # printf("form V..\n");
        V = torch.mm(Qhat,Uhat)

        # resize matrices to rank k from beginning
        # printf("resize mats\n");
        # U m x l -> m x k
        U = U[:, :k]
        # V m x l -> n x k
        V = V[:, :k]
        # S l x l -> k x k
        S = S[:k]
    # else:
    #     printf("using eigendecomposition of B B^T method\n");
    #     // build the matrix B B^T = Q^T M M^T Q column by column 
    #     // Bt = M^T Q ; nxm * mxk = nxk
    #     printf("form BBt..\n");
    #     mat *B = matrix_new(l,n);
    #     matrix_transpose_matrix_mult(Q,M,B);

    #     mat *BBt = matrix_new(l,l);
    #     matrix_matrix_transpose_mult(B,B,BBt);    

    #     // compute eigendecomposition of BBt
    #     printf("eigendecompose BBt..\n");
    #     vec *evals = vector_new(l);
    #     mat *Uhat = matrix_new(l,l);
    #     matrix_copy_symmetric(Uhat,BBt);
    #     compute_evals_and_evecs_of_symm_matrix(Uhat, evals);


    #     // compute singular values and matrix Sigma
    #     printf("form S..\n");
    #     vec *singvals = vector_new(l);
    #     for(i=0; i<l; i++){
    #         vector_set_element(singvals,i,std::sqrt(vector_get_element(evals,i)));
    #     }
    #     initialize_diagonal_matrix(*S, singvals);
        
    #     // compute U = Q*Uhat mxk * kxk = mxk  
    #     printf("form U..\n");
    #     matrix_matrix_mult(Q,Uhat,*U);

    #     // compute nxk V 
    #     // V = B^T Uhat * Sigma^{-1}
    #     printf("form V..\n");
    #     mat *Sinv = matrix_new(l,l);
    #     mat *UhatSinv = matrix_new(l,l);
    #     invert_diagonal_matrix(Sinv,*S);
    #     matrix_matrix_mult(Uhat,Sinv,UhatSinv);
    #     matrix_transpose_matrix_mult(B,UhatSinv,*V);

    #     matrix_delete(BBt);
    #     matrix_delete(Sinv);
    #     matrix_delete(UhatSinv);

    #     // resize matrices to rank k from end
    #     printf("resize mats to rank %d from end\n",k);
    #     //resize_matrix_by_columns_from_end(U,k);
    #     //resize_matrix_by_columns_from_end(V,k);
    #     resize_matrix_by_columns_from_end(S,k);
    #     resize_matrix_by_rows_from_end(S,k);
    
    #self.save_for_backward(U, S, V)
    return U, S, V