'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch
import numpy as np
try:
    import scipy.linalg
except:
    print("Warning: Missing scipy. Complex SVD is not available.")
from tn_interface import mm
from tn_interface import conj, transpose

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVDGESDD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        U, S, V = torch.svd(A)
        self.save_for_backward(U, S, V)
        return U, S, V
    
    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vt = V.t()
        Ut = U.t()
        M = U.size(0)
        N = V.size(0)
        NS = S.size(0)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G = safe_inverse(G)
        G.diagonal().fill_(0)

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        
        if dA.isnan().any():
            pdb.set_trace()

        return dA

class SVDGESDD_COMPLEX(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        AA= A.detach().cpu().numpy()
        UU, SS, VV= scipy.linalg.svd(AA)
        
        U= torch.as_tensor(UU)
        V= conj(transpose(torch.as_tensor(VV)))
        
        S= torch.as_tensor(SS)
        
        self.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(self, dU, dS, dV):
        U, S, V = self.saved_tensors
        Vh = conj(transpose(V))
        Uh = conj(transpose(U))
        M = U.size(0)
        N = V.size(0)
        NS = S.size(0)
        
        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        G.diagonal().fill_(float('inf'))
        G = 1/G

        E= torch.eye(NS, dtype=dS.dtype, device=dS.device)
        SI = 1/S

        UhdU = mm(Uh, dU)
        VhdV = mm(Vh, dV)

        J = (F+G)*(UhdU - conj(transpose(UhdU)))/2
        K = (F-G)*(VhdV - conj(transpose(VhdV)))/2
        L = E*VhdV 
        SILhL = SI.view(-1,1)*(conj(transpose(L)) - L)/2

        dA = mm( mm( U, (J + K + SILhL + torch.diag(dS)) ), Vh)
        pdb.set_trace()
        return dA

def test_SVDGESDD_random():
    
    M, N = 50, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD.apply, A, eps=1e-6, atol=1e-5))

    M, N = 40, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD.apply, A, eps=1e-6, atol=1e-5))

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
        print(D)

        try:
            A.requires_grad_()
            assert(torch.autograd.gradcheck(SVDGESDD.apply, A, eps=1e-6, atol=1e-5))
        except Exception as e:
            print(f"FAILED for splits: {split_scale}")
            print(e)


def test_SVDGESDD_COMPLEX_random():
    M = 5
    A = torch.rand((M, M), dtype=torch.complex128, requires_grad=True)
    print(A)
    def test_f(M):
        U,S,V= SVDGESDD_COMPLEX.apply(M)
        return torch.sum(S[0:1])

    assert(torch.autograd.gradcheck(test_f, A, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    test_SVDGESDD_random()
    # test_SVDGESDD_COMPLEX_random()