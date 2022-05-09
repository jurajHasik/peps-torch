'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SVDSYMEIG(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        r"""
        :param A: square symmetric matrix
        :type A: torch.Tensor
        :return: left singular vectors U, singular values S, and right singular
                vectors V
        :rtype: torch.Tensor, torch.Tensor, torch.Tensor

        Computes SVD of a matrix M, where M is symmetric :math:`M=M^T`,     
        through symmetric decomposition :math:`M= UDU^T`.
        """
        # input validation (A is square and symmetric) is provided by torch.symeig
        
        D, U = torch.symeig(A, eigenvectors=True)
        # torch.symeig returns eigenpairs ordered in the ascending order with 
        # respect to eigenvalues. Reorder the eigenpairs by abs value of the eigenvalues
        # abs(D)
        S,p= torch.sort(torch.abs(D),descending=True)
        U= U[:,p]
        # TODO how to handle case of vanishingly small S <=> Sinv -> inf
        # in principle, the multiplication by Sinv just scales the rows
        # of V to norm 1, such that V is unitary
        
        # 0) M = USV^t => (M^t)US^-1 = V => (M^t=M) => MUS^-1 = V
        # eps_cutoff=S[0] * 1.0e-14
        # Sinv= 1/S
        # Sinv[Sinv > 1/eps_cutoff]= 0.
        # V= A@U@torch.diag(Sinv)
        
        # 1) M = UDU^t = US(sgn)U^t = U S (sgn)U^t = U S V^t
        # (sgn) is a diagonal matrix with signs of the eigenvales D
        V= U@torch.diag(torch.sign(D[p]))

        self.save_for_backward(U, S, V)
        return U, S, V

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
        #F= 1/F
        F.diagonal().fill_(0)
        # F[abs(F) > 1.0e+8]=0

        G = (S + S[:, None])
        G = safe_inverse(G)
        G.diagonal().fill_(0)
        # G = 1/G
        # G[abs(G) > 1.0e+8]=0

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vt
        if (M>NS):
            dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt 
        if (N>NS):
            dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
        return dA

def test_SVDSYMEIG_random():
    m= 50
    M= torch.rand(m, m, dtype=torch.float64)
    M= 0.5*(M+M.t())

    U,S,V= SVDSYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(S)@V.t()) < S[0]*(m**2)*1e-14 )

    # since we always assume matrix M to be symmetric, the finite difference
    # perturbations should be symmetric as well
    M.requires_grad_(True)
    def force_sym_SVD(M):
        M=0.5*(M+M.t())
        return SVDSYMEIG.apply(M)
    assert(torch.autograd.gradcheck(force_sym_SVD, M, eps=1e-6, atol=1e-4))

def test_SVDSYMEIG_su2sym():
    import su2sym.sym_ten_parser as tenSU2
    # Available D: [3,5,7,9]
    for D in [3,5,7]:
        su2sym_t= tenSU2.import_sym_tensors(2,D,"A_1",dtype=torch.float64)
        c= torch.rand(len(su2sym_t), dtype=torch.float64)
        ts= torch.stack([tensor for meta,tensor in su2sym_t])
        a= torch.einsum('i,ipuldr->puldr',c,ts)
        D2= D**2
        M= torch.einsum('mijef,mijab->eafb',(a,a)).contiguous().view(D2, D2)

        U,S,V= SVDSYMEIG.apply(M)
        assert( torch.norm(M-U@torch.diag(S)@V.t()) < S[0]*(M.size()[0]**2)*1e-14 )

        M.requires_grad_(True)
        def force_sym_SVD(M):
            M=0.5*(M+M.t())
            return SVDSYMEIG.apply(M)
        assert(torch.autograd.gradcheck(force_sym_SVD, M, eps=1e-6, atol=1e-4))

def test_SVDSYMEIG_3x3degenerate():
    M= torch.zeros((3,3),dtype=torch.float64)
    M[0,1]=M[0,2]=M[1,2]=1.
    M= 0.5*(M+M.t())
    print(M)

    U,S,V= SVDSYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(S)@V.t()) < S[0]*(M.size()[0]**2)*1e-14 )

    M.requires_grad_(True)
    torch.set_printoptions(precision=9)
    def force_sym_SVD(M):
        M=0.5*(M+M.t())
        print(M)
        U,S,V= SVDSYMEIG.apply(M)
        return U
    assert(torch.autograd.gradcheck(force_sym_SVD, M, eps=1e-6, atol=1e-4))

def test_SVDSYMEIG_rank_deficient():
    m= 50
    r= 10
    M= torch.rand((m,m),dtype=torch.float64)
    M= M+M.t()
    D, U= torch.symeig(M, eigenvectors=True)
    D[-r:]=0
    M= U@torch.diag(D)@U.t()

    U,S,V= SVDSYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(S)@V.t()) < S[0]*(M.size()[0]**2)*1e-14 )

    M.requires_grad_(True)
    def force_sym_SVD(M):
        M=0.5*(M+M.t())
        U,S,V= SVDSYMEIG.apply(M)
        return U
    assert(torch.autograd.gradcheck(force_sym_SVD, M, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    test_SVDSYMEIG_random()
    test_SVDSYMEIG_rank_deficient()
    # test_SVDSYMEIG_3x3degenerate()
    # test_SVDSYMEIG_su2sym()
