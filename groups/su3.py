import torch
from math import factorial, sqrt
from tn_interface import einsum
import numpy as np

class SU3_DEFINING():
    def __init__(self, p=1, q=0, dtype=torch.complex128, device='cpu'):
        r"""
        :param p: (p,q) labels of the highest weight state of su(3) representation. 
                  For defining representation ``p=1, q=0``.
        :type p: int
        :param q:
        :type q: int
        :param dtype: data type of matrix representation of operators
        :param device: device on which the torch.tensor objects are stored
        :type J: int
        :type dtype: torch.dtype
        :type device: int

        Build the defining representation :math:`\bf{3}` of su(3) Lie algebra using 
        the Cartan-Weyl (C-W) basis. In terms of the standard Gell-Mann matrices :math:`\lambda`,
        the C-W basis is:
    
        .. math::
            \begin{align*}
            T^\pm &= \frac{1}{2} (\lambda_1 \pm i\lambda_2) = (F_1 \pm iF_2)\\
            T^z   &= \frac{1}{2} \lambda_3 = F_3\\
            V^\pm &= \frac{1}{2} (\lambda_4 \pm i\lambda_5) = (F_4 \pm iF_5)\\
            U^\pm &= \frac{1}{2} (\lambda_6 \pm i\lambda_7) = (F_6 \pm iF_7)\\
            Y     &= \frac{1}{\sqrt{3}} \lambda_8 = \frac{2}{\sqrt{3}} F_8
            \end{align*}

        The quadratic Casimir operator of su(3) can be expressed in terms of the C-W basis, defined as follow.

        .. math::
            \begin{align*}
            C_1 = \sum_{k}{F_k F_k} &= \frac{1}{2} (T^+ T^- + T^- T^+ + V^+ V^- + V^- V^+ + U^+ U^- + U^- U^+) \\
                                    &+ T^z T^z + \frac{3}{4} Y Y
            \end{align*}
        """
        assert p==1 and q==0, "su(3) irrep ("+str(p)+","+str(q)+") not implemented."
        self.p = p
        self.q = q
        self.dtype = dtype
        self.device = device

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: torch.tensor
        """
        return get_op("I", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def TZ(self):
        r"""
        :return: :math:`T^z` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tz", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def Y(self):
        r"""
        :return: :math:`Y` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("y", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def TP(self):
        r"""
        :return: :math:`T^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tp", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def TM(self):
        r"""
        :return: :math:`T^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tm", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def VP(self):
        r"""
        :return: :math:`V^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("vp", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def VM(self):
        r"""
        :return: :math:`V^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("vm", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def UP(self):
        r"""
        :return: :math:`U^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("up", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def UM(self):
        r"""
        :return: :math:`U^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("um", pq=(self.p,self.q), dtype=self.dtype, device=self.device)

    def Cartan_Weyl(self):
        r"""
        :return: vector of generators forming Cartan-Weyl basis ordered
                 as [T^+, T^-, T^z, V^+, V^-, U^+, U^-, Y].
        :rtype: torch.tensor

        Returns a rank-3 tensor with first index running over generators.
        """
        J = torch.zeros(8, 3, 3, dtype=self.dtype, device=self.device)
        J[0, :, :] = self.TP()
        J[1, :, :] = self.TM()
        J[2, :, :] = self.TZ()
        J[3, :, :] = self.VP()
        J[4, :, :] = self.VM()        
        J[5, :, :] = self.UP()
        J[6, :, :] = self.UM()
        J[7, :, :] = self.Y()

        return J

    def J_Gell_Mann(self):
        r"""
        :return: :math:`\vec{\lambda}` vector of Gell-Mann matrices
        :rtype: torch.tensor

        Returns a rank-3 tensor with first index running over generators.
        """
        J = torch.zeros(8, 3, 3, dtype=self.dtype, device=self.device)
        J[0, :, :] = self.TP() + self.TM()
        J[1, :, :] = -1j * (self.TP() - self.TM())
        J[2, :, :] = 2 * self.TZ()
        J[3, :, :] = self.VP() + self.VM()
        J[4, :, :] = -1j * (self.VP() - self.VM())
        J[5, :, :] = self.UP() + self.UM()
        J[6, :, :] = -1j * (self.UP() - self.UM())
        J[7, :, :] = np.sqrt(3) * self.Y()

        return J

    def C1(self):
        r"""
        :return: The quadratic Casimir of su(3) as rank-4 for tensor
        :rtype: torch.tensor
        """
        expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \sum_k{\vec{F}_{1,k}\vec{S}_{2,k}} between F-spins on sites 1 and 2
        C1 = einsum(expr_kron, self.TZ(), self.TZ()) + 0.75 * einsum(expr_kron, self.Y(), self.Y())\
            + 0.5 * (einsum(expr_kron, self.TP(), self.TM()) + einsum(expr_kron, self.TM(), self.TP())
                     + einsum(expr_kron, self.VP(), self.VM()) + einsum(expr_kron, self.VM(), self.VP())
                     + einsum(expr_kron, self.UP(), self.UM()) + einsum(expr_kron, self.UM(), self.UP()))
        return C1

    def C2(self):
        r"""
        :return: The cubic Casimir of su(3) as rank-6 for tensor
        :rtype: torch.tensor
        """
        expr_kron = 'ia,jb,kc->ijkabc'
        Fs = dict()
        Fs["f1"] = 0.5 * (self.TP() + self.TM())
        Fs["f2"] = - 0.5j * (self.TP() - self.TM())
        Fs["f3"] = self.TZ()
        Fs["f4"] = 0.5 * (self.VP() + self.VM())
        Fs["f5"] = - 0.5j * (self.VP() - self.VM())
        Fs["f6"] = 0.5 * (self.UP() + self.UM())
        Fs["f7"] = - 0.5j * (self.UP() - self.UM())
        Fs["f8"] = np.sqrt(3.0) / 2 * self.Y()
        C2 = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128, device='cpu')
        # C2 = None
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    d = 2 * torch.trace((Fs[f"f{i+1}"]@Fs[f"f{j+1}"]+Fs[f"f{j+1}"]@Fs[f"f{i+1}"])@Fs[f"f{k+1}"])
                    C2 += d * einsum(expr_kron, Fs[f"f{i+1}"], Fs[f"f{j+1}"], Fs[f"f{k+1}"])

        return C2


def get_op(op, pq=(1,0), dtype=torch.complex128, device='cpu', dbg=False):
    assert pq==(1,0),"Unsupported irrep"
    if op == "I":
        if dbg:
            print(">>>>> Constructing 1sO: Id <<<<<")
        return torch.eye(3, dtype=dtype, device=device)
    elif op == "tz":
        if dbg:
            print(">>>>> Constructing 1sO: T^z <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[0, 0] = 0.5
        res[1, 1] = -0.5
        return res
    elif op == "y":
        if dbg:
            print(">>>>> Constructing 1sO: Y <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[0, 0] = 1.0 / 3.0
        res[1, 1] = 1.0 / 3.0
        res[2, 2] = - 2.0 / 3.0
        return res
    elif op == "tp":
        if dbg:
            print(">>>>> Constructing 1sO: T^+ <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[0, 1] = 1.0
        return res
    elif op == "tm":
        if dbg:
            print(">>>>> Constructing 1sO: T^- <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[1, 0] = 1.0
        return res
    elif op == "vp":
        if dbg:
            print(">>>>> Constructing 1sO: V^+ <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[0, 2] = 1.0
        return res
    elif op == "vm":
        if dbg:
            print(">>>>> Constructing 1sO: V^- <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[2, 0] = 1.0
        return res
    elif op == "up":
        if dbg:
            print(">>>>> Constructing 1sO: U^+ <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[1, 2] = 1.0
        return res
    elif op == "um":
        if dbg:
            print(">>>>> Constructing 1sO: U^- <<<<<")
        res = torch.zeros((3, 3), dtype=dtype, device=device)
        res[2, 1] = 1.0
        return res
    else:
        raise Exception("Unsupported operator requested: " + op)

#TODO: CG series of su(3), i.e., the expansion of the tensor product of two irrep into direct sum of irreps
