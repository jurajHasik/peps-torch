import numpy as np
import yamps.yast as yast
from math import factorial, sqrt

class SU3_DEFINING_U1xU1():
    _REF_S_DIRS=(-1,1)

    def __init__(self, settings, p=1, q=0):
        r"""
        :param (p, q): labels of highest weight for su(3) representations. (1, 0) - defining representation
        :param dtype: data type of matrix representation of operators
        :param device: device on which the torch.tensor objects are stored

        Build the defining representation "3" of su(3) Lie algebra using the Cartan-Weyl basis, $\lambda_i$.

        The U(1)xU(1) charges for states defining (1,0) irrep can be assigned
        as (rescaled) eigenvalues of T^z and Y operators

            ( 1,  1)
            (-1,  1)
            ( 0, -2)

        The quadratic Casimir operator of su(3) can be expressed in terms of the C-W basis, defined as follow.

        .. math::
            \begin{align*}
            T^\pm &= \frac{1}{2} (\lambda_1 \pm i\lambda_2) = (F_1 \pm iF_2)\\
            T^z   &= \frac{1}{2} \lambda_3 = F_3\\
            V^\pm &= \frac{1}{2} (\lambda_4 \pm i\lambda_5) = (F_4 \pm iF_5)\\
            U^\pm &= \frac{1}{2} (\lambda_6 \pm i\lambda_7) = (F_6 \pm iF_7)\\
            Y     &= \frac{1}{\sqrt{3}} \lambda_8 = \frac{2}{\sqrt{3}} F_8
            \end{align*}

            \begin{align*}
            C_1 = \sum_{k}{F_k F_k} &= \frac{1}{2} (T^+ T^- + T^- T^+ + V^+ V^- + V^- V^+ + U^+ U^- + U^- U^+) \\
                                    &+ T^z T^z + \frac{3}{4} Y Y
            \end{align*}
        """
        assert settings.sym.NSYM==2, "U(1)xU(1) abelian symmetry is assumed"
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.default_dtype
        self.device= 'cpu' if not hasattr(settings, 'device') else settings.device
        
        assert p==1 and q==0, "Only (1,0) irrep is implemented"
        self.p = p
        self.q = q
        self.charges= [(1,1), (-1,1), (0,-2)] 

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for c in self.charges:
            op.set_block(ts=(c,c), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def TZ(self):
        r"""
        :return: :math:`T^z` operator of irrep
        :rtype: yast.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for val, c in zip( [0.5, -0.5], self.charges[:2] ):
            op.set_block(ts=(c,c), Ds=(1,1), val=val*unit_block)
        op= op.to(self.device)
        return op

    def Y(self):
        r"""
        :return: :math:`Y` operator of irrep
        :rtype: yast.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for val, c in zip( [1./3, 1./3, -2./3], self.charges ):
            op.set_block(ts=(c,c), Ds=(1,1), val=val*unit_block)
        op= op.to(self.device)
        return op

    def TP(self):
        r"""
        :return: :math:`T^+` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(-2,0))
        # (1,1) <- (-1,1)
        op.set_block(ts=(self.charges[0],self.charges[1]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def TM(self):
        r"""
        :return: :math:`T^-` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(2,0))
        # (-1,1) <- (1,1)
        op.set_block(ts=(self.charges[1],self.charges[0]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def VP(self):
        r"""
        :return: :math:`V^+` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(-1,-3))
        # (1,1) <- (0,-2)
        op.set_block(ts=(self.charges[0],self.charges[2]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def VM(self):
        r"""
        :return: :math:`V^-` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(1,3))
        # (0,-2) <- (1,1)
        op.set_block(ts=(self.charges[2],self.charges[0]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def UP(self):
        r"""
        :return: :math:`U^+` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(1,-3))
        # (-1,1) <- (0,-2)
        op.set_block(ts=(self.charges[1],self.charges[2]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def UM(self):
        r"""
        :return: :math:`U^-` operator of irrep
        :rtype: yast.Tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=(-1,3))
        # (0,-2) <- (-1,1)
        op.set_block(ts=(self.charges[2],self.charges[1]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def G(self):
        # metric tensor on adjoint irrep
        unit_block= np.ones((1,1), dtype=self.dtype)
        
        # charges on 0-th index, indexing generators of su(3) defined in the space
        # of (1,0) irrep. Equivalently, these generators span (1,1) (adjoint) irrep. 
        #
        # (-2, 0): 1,  TP n=(-2,0)
        # (-1, -3): 1, VP n=(-1,-3)
        # (-1, 3): 1,  UM n=(-1,3)
        # (0, 0): 2,   TZ, Y n=(0,0)
        # (1, -3): 1,  UP n=(1,-3)
        # (1, 3): 1,   VM n=(1,3) 
        # (2, 0): 1    TM n=(2,0)
        G= yast.Tensor(self.engine, s=(1,1), n=(0,0))
        G.set_block(ts=(0,0,0,0), Ds=(2,2), val=np.asarray([[1.,0.],[0.,3./4]], self.dtype))
        G.set_block(ts=(-1,-3,1,3), Ds=(1,1), val=0.5*unit_block)
        G.set_block(ts=(1,3,-1,-3), Ds=(1,1), val=0.5*unit_block)
        G.set_block(ts=(-1,3,1,-3), Ds=(1,1), val=0.5*unit_block)
        G.set_block(ts=(1,-3,-1,3), Ds=(1,1), val=0.5*unit_block)
        G.set_block(ts=(-2,0,2,0), Ds=(1,1), val=0.5*unit_block)
        G.set_block(ts=(2,0,-2,0), Ds=(1,1), val=0.5*unit_block)
        return G

    def Cartan_Weyl(self):
        r"""
        :return: vector of generators forming Cartan-Weyl basis ordered
                 as [T^+, T^-, T^z, V^+, V^-, U^+, U^-, Y]
        :rtype: yast.Tensor
        
        The signature of this rank-3 tensor is::

             1(-1)
             |
             T--0(-1)
             |
             2(+1)
            
        The extra index is charged, such that the total tensors in U(1)xU(1)
        invariant.
        """
        op_v= yast.Tensor(self.engine, s=[-1]+list(self._REF_S_DIRS), n=(0,0))
        
        # center
        center= [self.TZ(), self.Y()]
        for c in self.charges:
            op_v.set_block(ts=((0,0),c,c), Ds=(len(center), 1,1), val='zeros')
        for i,op in enumerate(center):
            for c,block in op.A.items():
                op_v.A[(*op.get_tensor_charge(), *c)][i,:,:]= block 

        # lowering and raising operators
        for op in [self.TP(), self.TM(), self.VP(), self.VM(),\
            self.UP(), self.UM()]:
            for c,block in op.A.items():
                op_v.set_block(ts= (*op.get_tensor_charge(), *c), val=block[None,:,:])
        return op_v

    def C1(self):
        r"""
        :return: The quadratic Casimir of su(3) as rank-4 for tensor
        :rtype: torch.tensor
        """
        # spin-spin interaction \sum_k{\vec{F}_{1,k}\vec{S}_{2,k}} between F-spins on sites 1 and 2
        
        CW_basis= self.Cartan_Weyl()

        #             1    1->0       1->2
        # 0--G--1 0--CW => CW--0 0--GCW
        #             2    2->1       2->3
        C1= yast.tensordot(self.G(), CW_basis, ([1],[0]))
        C1= yast.tensordot(CW_basis, C1, ([0],[0])).transpose(axes=(0,2,1,3))
        return C1

    # def C2(self):
    #     r"""
    #     :return: The cubic Casimir of su(3) as rank-6 for tensor
    #     :rtype: torch.tensor
    #     """
    #     expr_kron = 'ia,jb,kc->ijkabc'
    #     Fs = dict()
    #     Fs["f1"] = 0.5 * (self.TP() + self.TM())
    #     Fs["f2"] = - 0.5j * (self.TP() - self.TM())
    #     Fs["f3"] = self.TZ()
    #     Fs["f4"] = 0.5 * (self.VP() + self.VM())
    #     Fs["f5"] = - 0.5j * (self.VP() - self.VM())
    #     Fs["f6"] = 0.5 * (self.UP() + self.UM())
    #     Fs["f7"] = - 0.5j * (self.UP() - self.UM())
    #     Fs["f8"] = np.sqrt(3.0) / 2 * self.Y()
    #     C2 = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128, device='cpu')
    #     # C2 = None
    #     for i in range(8):
    #         for j in range(8):
    #             for k in range(8):
    #                 d = 2 * torch.trace((Fs[f"f{i+1}"]@Fs[f"f{j+1}"]+Fs[f"f{j+1}"]@Fs[f"f{i+1}"])@Fs[f"f{k+1}"])
    #                 C2 += d * einsum(expr_kron, Fs[f"f{i+1}"], Fs[f"f{j+1}"], Fs[f"f{k+1}"])

    #     return C2