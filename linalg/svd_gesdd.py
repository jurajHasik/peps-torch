'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import context
import torch
import numpy as np
try:
    import scipy.linalg
except:
    print("Warning: Missing scipy. Complex SVD is not available.")
from tn_interface import mm
from tn_interface import conj, transpose
from complex_num.complex_operation import *
import pdb

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

    # @staticmethod
    # def forward(self, A):
    #     U, S, V = torch.svd(A)
    #     AA = A.detach().cpu().numpy()
    #     A2 = AA[0] + 1j*AA[1]
    #     UU, SS, VV = scipy.linalg.svd(A2)

    #     U1 = torch.as_tensor(UU.real)
    #     U2 = torch.as_tensor(UU.imag)
    #     U = torch.stack((U1, U2), dim=0)

    #     V1 = torch.as_tensor(VV.real.transpose())
    #     V2 = torch.as_tensor(-VV.imag.transpose())
    #     V = torch.stack((V1, V2), dim=0)

    #     S1 = torch.as_tensor(SS)
    #     dim = S1.size()
    #     temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    #     S = torch.stack((S1, temp), dim=0)

    #     self.save_for_backward(U, S, V)
    #     return U, S, V

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
        return dA

    # @staticmethod
    # def backward(self, dU, dS, dV):
    #     U, S, V = self.saved_tensors
    #     Vt = complex_conjugate(transpose_complex(V))
    #     Ut = complex_conjugate(transpose_complex(U))
    #     M = size_complex(U)[0]
    #     N = size_complex(V)[0]
    #     NS = len(S[0])
        
    #     S2 = S[0]
    #     F = (S2 - S2[:, None])
    #     F = safe_inverse(F)
    #     F.diagonal().fill_(0)

    #     G = (S2 + S2[:, None])
    #     G.diagonal().fill_(float('inf'))
    #     G = 1/G

    #     Iden = torch.eye(S2.size(0), dtype=cfg.global_args.dtype, device=cfg.global_args.device)

    #     SIr = 1/S2
    #     dim = SIr.size()
    #     temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    #     SI = torch.stack((SIr, temp), dim=0)

    #     UdU = mm_complex(Ut, dU)
    #     VdV = mm_complex(Vt, dV)
    #     UdU2 = (UdU - complex_conjugate(transpose_complex(UdU)))/2.0
    #     VdV2 = (VdV - complex_conjugate(transpose_complex(VdV)))/2.0

    #     Jr = (F+G)*UdU2[0]; Ji = (F+G)*UdU2[1]
    #     J = torch.stack((Jr, Ji), dim=0)
    #     Kr = (F-G)*VdV2[0]; Ki = (F-G)*VdV2[1]
    #     K = torch.stack((Kr, Ki), dim=0)
    #     Lr = Iden*VdV[0]; Li = Iden*VdV[1]
    #     L = torch.stack((Lr, Li), dim=0)
    #     LLt = complex_conjugate(transpose_complex(L)) - L
    #     SILLt = mm_complex(diag_complex(SI), LLt)/2.0

    #     dA = mm_complex(mm_complex(U, (J + K + diag_complex(dS))), Vt)
    #     return dA

def test_SVDGESDD_random():
    M, N = 50, 40
    A = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVDGESDD.apply, A, eps=1e-6, atol=1e-4))

def test_SVDGESDD_COMPLEX_random():
    M = 5
    # A = torch.rand((2, M, M), dtype=torch.float64, requires_grad=True)
    A = torch.rand((2, M, M), dtype=torch.float64, requires_grad=True)
    A[1]= 0.
    U,S,V= torch.svd(A)
    print(S)
    assert(torch.autograd.gradcheck(SVDGESDD_COMPLEX.apply, A, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    # test_SVDSYMEIG_random()
    test_SVDGESDD_COMPLEX_random()