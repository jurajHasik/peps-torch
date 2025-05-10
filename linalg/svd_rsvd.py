import torch

def rsvd(M, k, p = 20, q = 2, s = 1, vnum = 1, **kwargs):
    r"""
    :param M: real matrix
    :param k: desired rank
    :param p: oversampling rank. Total rank sampled ``k+p``
    :param q: number of matrix-vector multiplications for power scheme
    :param s: re-orthogonalization 
    :type M: torch.Tensor 
    :type k: int
    :type p: int
    :type q: int
    :type s: int
    :return: approximate leading k left singular vectors U, singular values S, 
                and right singular vectors V
    :rtype: torch.Tensor, torch.Tensor, torch.Tensor

    Performs approximate truncated SVD of real matrix M using randomized sampling 
    as :math:`M=USV^T`. Based on the implementation in https://arxiv.org/abs/1502.05366
    """

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
            Yorth, r = torch.linalg.qr(Y)
            # printf("Z = M'*Yorth..\n");
            Z = torch.mm(M.transpose(0,1).conj(),Yorth)
        else:
            # printf("Z = M'*Y..\n");
            Z = torch.mm(M.transpose(0,1).conj(),Y)

    
        if (2*j-1) % s == 0:
            # printf("orthogonalize Z..\n");
            Zorth, r = torch.linalg.qr(Z)
            # printf("Y = M*Zorth..\n");
            Y = torch.mm(M,Zorth)
        else:
            # printf("Y = M*Z..\n");
            Y = torch.mm(M,Z)

    # orthogonalize on exit from loop to get Q = m x l
    Q, r = torch.linalg.qr(Y)

    # either QR of B^T method, or eigendecompose BB^T method
    if (vnum == 1 or vnum > 2):
        # printf("using QR of B^T method\n");
        
        # form Bt = Mt*Q : nxm * mxl = nxl
        # printf("form Bt..\n");
        Bt = torch.mm(M.transpose(0,1).conj(),Q)

        # compute QR factorization of Bt    
        # M is mxn ; Q is mxn ; R is min(m,n) x min(m,n) */ 
        # printf("doing QR..\n");
        ### compact_QR_factorization(Bt,Qhat,Rhat);
        Qhat, Rhat = torch.linalg.qr(Bt)

        # compute SVD of Rhat (lxl)
        # printf("doing SVD..\n");
        # torch.svd M -> U, S, V such that M = U S V^T
        Uhat, S, Vhat_trans = torch.linalg.svd(Rhat)
        # Vhat_trans = Vhat_trans.transpose(0,1) # since torch.linalg.svd returns V

        # U = Q*Vhat_trans
        # printf("form U..\n");
        U = torch.mm(Q,Vhat_trans.transpose(0,1).conj())

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
    
    return U, S, V