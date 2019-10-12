import numpy as np
import torch
import torch.nn.functional as Functional
try:
    import scipy.sparse.linalg
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")

class ARNOLDISVD(torch.autograd.Function):

    # INPUT:
    # M - input matrix
    # k [int] - desired rank
    # OUTPUT
    # U, S, V - truncated svd decomposition M \approx U S V^dag
    @staticmethod
    def forward(self, M, k):
        # check symmetry of M
        #print("norm(0.5*(M-M^t)) "+str(torch.norm(M - M.t())))

        # get M as numpy ndarray
        M_nograd = M.clone().detach().numpy()
        M_nograd = M_nograd @ M_nograd
        #print("FWD k: "+ str(k) + " M: " + str(M_nograd.shape))

        # we get, say (since M is symmetric) left singular vectors of M
        # M = U.S.Vt -> M.Mt = U.S.S.Ut
        s, u = scipy.sparse.linalg.eigsh(M_nograd, k=k)

        # Debug info about properties of output tensors from eigsh
        #print("u "+str(u.shape))
        #print("u "+str(u.strides))

        # find range
        # print(s)
        # if (s[0] is np.nan) or (s[0] != s[0]):
        #     raise Exception('nan')
        # rankS = len(list(filter(lambda e: e > 0., s)))
        # print(rankS)

        # properly order the result (singular values are given in ascending
        # order) by ARPACK
        u = np.copy(u[:,::-1])
        s = np.copy(s[::-1])
        s = np.sqrt(np.abs(s))

        U = torch.as_tensor(u)
        S = torch.as_tensor(s)
        
        # compute right singular vectors as Mt = V.S.Ut /.U => Mt.U = V.S
        # since M = Mt, M.U = V.S
        V = M @ U
        V = Functional.normalize(V, p=2, dim=0)

        # print("U "+str(U.shape))
        # print(U.stride())
        # for i in range(k):
        #     print(str(i)+": "+str(np.linalg.norm(U[:,i])))
        #print(S.shape)
        #print(S)
        # print("V "+str(V.shape))
        # print(V.stride())
        # for i in range(k):
        #     print(str(i)+": "+str(np.linalg.norm(V[:,i])))

        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        raise Exception("[ARNOLDISVD_SYM] backward not implemented")
        U, S, V = self.saved_tensors
        return dA, None

# def test_svd():
#     M, N = 50, 40
#     torch.manual_seed(2)
#     input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
#     assert(torch.autograd.gradcheck(SVD.apply, input, eps=1e-6, atol=1e-4))
#     print("Test Pass!")

# if __name__=='__main__':
#     test_svd()