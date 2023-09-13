import numpy as np
import yastn.yastn as yastn
from math import factorial, sqrt

class SU3_DEFINING_U1xU1():
    _REF_S_DIRS=(-1,1)

    def __init__(self, settings, p=1, q=0):
        r"""
        :param p: (p,q) labels of the highest weight state of su(3) representation. 
                  For defining representation ``p=1, q=0``.
        :type p: int
        :param q:
        :type q: int
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)

        Build the defining representation :math:`\bf{3}` of su(3) Lie algebra using 
        the Cartan-Weyl basis. In terms of the standard Gell-Mann matrices :math:`\lambda`,
        the C-W basis is:

        .. math::
            \begin{align*}
            T^\pm &= \frac{1}{2} (\lambda_1 \pm i\lambda_2) = (F_1 \pm iF_2)\\
            T^z   &= \frac{1}{2} \lambda_3 = F_3\\
            V^\pm &= \frac{1}{2} (\lambda_4 \pm i\lambda_5) = (F_4 \pm iF_5)\\
            U^\pm &= \frac{1}{2} (\lambda_6 \pm i\lambda_7) = (F_6 \pm iF_7)\\
            Y     &= \frac{1}{\sqrt{3}} \lambda_8 = \frac{2}{\sqrt{3}} F_8
            \end{align*}

        The U(1)xU(1) charges for states spanning :math:`\mathbf{3}=(1,0)` irrep can be assigned
        as (rescaled) eigenvalues of diagonal :math:`T^z` and Y operators::

            ( 1,  1)
            (-1,  1)
            ( 0, -2)

        The signature convention :math:`O = \sum_{ij} O_{ij}|i\rangle\langle j|` is -1 for 
        index `i` (:math:`|ket\rangle`) and +1 for index `j` (:math:`\langle bra|`).


        The quadratic Casimir operator of su(3) can be expressed in terms of the C-W basis, defined as follow.

        .. math::
            \begin{align*}
            C_1 = \sum_{k}{F_k F_k} &= \frac{1}{2} (T^+ T^- + T^- T^+ + V^+ V^- + V^- V^+ + U^+ U^- + U^- U^+) \\
                                    &+ T^z T^z + \frac{3}{4} Y Y
            \end{align*}
        """
        assert settings.sym.NSYM==2, "U(1)xU(1) abelian symmetry is assumed"
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.default_dtype
        self.device= settings.device if hasattr(settings, 'device') else settings.default_device
        
        assert p==1 and q==0, "su(3) irrep ("+str(p)+","+str(q)+") not implemented."
        self.p = p
        self.q = q
        self.charges= [(1,1), (-1,1), (0,-2)] 

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for c in self.charges:
            op.set_block(ts=(c,c), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def TZ(self):
        r"""
        :return: :math:`T^z` operator of irrep
        :rtype: yastn.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for val, c in zip( [0.5, -0.5], self.charges[:2] ):
            op.set_block(ts=(c,c), Ds=(1,1), val=val*unit_block)
        op= op.to(self.device)
        return op

    def Y(self):
        r"""
        :return: :math:`Y` operator of irrep
        :rtype: yastn.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(0,0))
        for val, c in zip( [1./3, 1./3, -2./3], self.charges ):
            op.set_block(ts=(c,c), Ds=(1,1), val=val*unit_block)
        op= op.to(self.device)
        return op

    def TP(self):
        r"""
        :return: :math:`T^+` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(-2,0))
        # (1,1) <- (-1,1)
        op.set_block(ts=(self.charges[0],self.charges[1]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def TM(self):
        r"""
        :return: :math:`T^-` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(2,0))
        # (-1,1) <- (1,1)
        op.set_block(ts=(self.charges[1],self.charges[0]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def VP(self):
        r"""
        :return: :math:`V^+` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(-1,-3))
        # (1,1) <- (0,-2)
        op.set_block(ts=(self.charges[0],self.charges[2]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def VM(self):
        r"""
        :return: :math:`V^-` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(1,3))
        # (0,-2) <- (1,1)
        op.set_block(ts=(self.charges[2],self.charges[0]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def UP(self):
        r"""
        :return: :math:`U^+` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(1,-3))
        # (-1,1) <- (0,-2)
        op.set_block(ts=(self.charges[1],self.charges[2]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def UM(self):
        r"""
        :return: :math:`U^-` operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=(-1,3))
        # (0,-2) <- (-1,1)
        op.set_block(ts=(self.charges[2],self.charges[1]), Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def G(self):
        r"""
        :return: metric tensor on adjoint irrep :math:`\mathbf{8}=(1,1)`.
        :rtype: yastn.Tensor

        Returns rank-2 tensor G, such that the quadratic Casimir in terms of C-W basis :math:`\vec{T}`
        can be computed as :math:`\vec{T}^T G \vec{T}`.
        """
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
        #
        G= yastn.Tensor(self.engine, s=(1,1), n=(0,0))
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
        :rtype: yastn.Tensor
        
        The signature of this rank-3 tensor is::

             1(-1)
             |
             T--0(-1)
             |
             2(+1)
            
        The first index, which runs over generators, is charged, such that the total tensor 
        is U(1)xU(1)-invariant.
        """
        op_v= yastn.block({i: t.add_leg(axis=0,s=-1) for i,t in enumerate([\
            self.TZ(), self.Y(), self.TP(), self.TM(), self.VP(), self.VM(),\
            self.UP(), self.UM()])}, common_legs=[1,2]).drop_leg_history(axes=0)
        return op_v

    def C1(self):
        r"""
        :return: The quadratic Casimir of su(3) as rank-4 for tensor
        :rtype: yastn.Tensor
        """
        # spin-spin interaction \sum_k{\vec{F}_{1,k}\vec{S}_{2,k}} between F-spins on sites 1 and 2
        
        CW_basis= self.Cartan_Weyl()

        #             1    1->0       1->2
        # 0--G--1 0--CW => CW--0 0--GCW
        #             2    2->1       2->3
        C1= yastn.tensordot(self.G(), CW_basis, ([1],[0]))
        C1= yastn.tensordot(CW_basis, C1, ([0],[0])).transpose(axes=(0,2,1,3))
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