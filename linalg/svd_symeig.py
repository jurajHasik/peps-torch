'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''

import numpy as np
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

def flip_tensor(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,\
        dtype=torch.long, device=x.device)
    return x[tuple(indices)]

class SVDSYMEIG(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        """
        Return a truncated SVD of a matrix M     
        M ~ (Ut)(St)(Vt)^{T}
        by computing the symmetric decomposition of MM^T

        inputs:
            M (torch.Tensor):
                tensor of shape (dim0, dim1)

            chi (int):
                maximum allowed dimension of S

            abs_tol (float):
                absolute tollerance on singular eigenvalues

            rel_tol (float):
                relative tollerance on singular eigenvalues

        where S is diagonal matrix of of shape (dimS, dimS)
        and dimS <= chi

        returns Ut, St, Vt
        """
        sym_error= torch.norm(A-A.t())
        assert sym_error < 1.0e+8, f"enlarged corner(M) is not a symmetric matrix norm(M-M^T)>{sym_error}"
        
        D, U = torch.symeig(A@A, eigenvectors=True)
        # torch.symeig returns eigenpairs ordered in the ascending order with respect to eigenvalues 
        D= flip_tensor(D, 0)
        U= flip_tensor(U, 1)
        # M = USV^T => (M^T)US^-1 = V => (M^T=M) => MUS^-1 = V
        eps_cutoff=D[0] * 1.0e-8
        Dinvqsrt= torch.rsqrt(D[D>eps_cutoff])
        Dzeros= torch.zeros(D.size()[0]-Dinvqsrt.size()[0],dtype=D.dtype,device=D.device) 
        Dinvqsrt= torch.cat((Dinvqsrt,Dzeros))
        D= torch.sqrt(torch.clamp(D,0,np.inf))
        V= A@U@torch.diag(Dinvqsrt)

        #print("AMENDED D: ")
        #print(D)
        # print("U:")
        # print(U[:,0])
        # print("V:")
        # print(V[:,0])
        self.save_for_backward(U, D, V)
        return U, D, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = len(S)
        NINF= len(S[S==0])

        threshold= 1.0e+8
        F = (S - S[:, None])
        #F = safe_inverse(F)
        F= 1/F
        F.diagonal().fill_(0)
        F[abs(F)>threshold]=0.

        G = (S + S[:, None])
        G.diagonal().fill_(np.inf)
        G = 1/G
        G[-NINF:,-NINF:]=0

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt 
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        if torch.sum(torch.isnan(dA),(0,1)) > 0:
            print(F)
            print(G)
            print(dA)
            print(S)
            print(dS)
            exit()
        return dA

def test_svd():
    M, N = 50, 40
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, input, eps=1e-6, atol=1e-4))
    print("Test Pass!")

if __name__=='__main__':
    test_svd()
