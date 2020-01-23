'''
Implementation taken from https://arxiv.org/abs/1903.09650
which follows derivation given in https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
'''
import torch

def safe_inverse(x, epsilon=1E-12):
    return x/(x**2 + epsilon)

class SYMEIG(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        r"""
        :param A: square symmetric matrix
        :type A: torch.tensor
        :return: eigenvalues values D, eigenvectors vectors U
        :rtype: torch.tensor, torch.tensor

        Computes symmetric decomposition :math:`M= UDU^T`.
        """
        # input validation (A is square and symmetric) is provided by torch.symeig
        
        D, U = torch.symeig(A, eigenvectors=True)
        # torch.symeig returns eigenpairs ordered in the ascending order with 
        # respect to eigenvalues. Reorder the eigenpairs by abs value of the eigenvalues
        # abs(D)
        absD,p= torch.sort(torch.abs(D),descending=True)
        D= D[p]
        U= U[:,p]
        
        self.save_for_backward(D,U)
        return D,U

    @staticmethod
    def backward(self, dD, dU):
        D, U= self.saved_tensors
        Ut = U.t()

        F = (D - D[:, None])
        F = safe_inverse(F)
        #F= 1/F
        F.diagonal().fill_(0)
        # F[abs(F) > 1.0e+8]=0

        dA = U @ (torch.diag(dD) + F*(Ut@dU)) @ Ut
        return dA

def test_SYMEIG_random():
    m= 50
    M= torch.rand(m, m, dtype=torch.float64)
    M= 0.5*(M+M.t())

    D,U= SYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(D)@U.t()) < D[0]*(m**2)*1e-14 )

    # since we always assume matrix M to be symmetric, the finite difference
    # perturbations should be symmetric as well
    M.requires_grad_(True)
    def force_sym_eig(M):
        M=0.5*(M+M.t())
        return SYMEIG.apply(M)
    assert(torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4))

def test_SYMEIG_su2sym():
    import su2sym.sym_ten_parser as tenSU2
    # Available D: [3,5,7,9]
    for D in [3,5,7]:
        su2sym_t= tenSU2.import_sym_tensors(2,D,"A_1",dtype=torch.float64)
        c= torch.rand(len(su2sym_t), dtype=torch.float64)
        ts= torch.stack([tensor for meta,tensor in su2sym_t])
        a= torch.einsum('i,ipuldr->puldr',c,ts)
        D2= D**2
        M= torch.einsum('mijef,mijab->eafb',(a,a)).contiguous().view(D2, D2)

        D,U= SYMEIG.apply(M)
        assert( torch.norm(M-U@torch.diag(D)@U.t()) < D[0]*(M.size()[0]**2)*1e-14 )

        M.requires_grad_(True)
        def force_sym_eig(M):
            M=0.5*(M+M.t())
            return SYMEIG.apply(M)
        assert(torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4))

def test_SYMEIG_3x3degenerate():
    M= torch.zeros((3,3),dtype=torch.float64)
    M[0,1]=M[0,2]=M[1,2]=1.
    M= 0.5*(M+M.t())
    print(M)

    D,U= SYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(D)@U.t()) < D[0]*(M.size()[0]**2)*1e-14 )

    M.requires_grad_(True)
    torch.set_printoptions(precision=9)
    def force_sym_eig(M):
        M=0.5*(M+M.t())
        print(M)
        D,U= SYMEIG.apply(M)
        return U
    assert(torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4))

def test_SYMEIG_rank_deficient():
    m= 50
    r= 10
    M= torch.rand((m,m),dtype=torch.float64)
    M= M+M.t()
    D,U= torch.symeig(M, eigenvectors=True)
    D[-r:]=0
    M= U@torch.diag(D)@U.t()

    D,U= SYMEIG.apply(M)
    assert( torch.norm(M-U@torch.diag(D)@U.t()) < D[0]*(M.size()[0]**2)*1e-14 )

    M.requires_grad_(True)
    def force_sym_eig(M):
        M=0.5*(M+M.t())
        D,U= SYMEIG.apply(M)
        return U
    assert(torch.autograd.gradcheck(force_sym_eig, M, eps=1e-6, atol=1e-4))

if __name__=='__main__':
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    test_SYMEIG_random()
    test_SYMEIG_rank_deficient()
    # test_SYMEIG_3x3degenerate()
    # test_SYMEIG_su2sym()
