import torch
from math import factorial, sqrt
from tn_interface import einsum


class SU3_DEFINING():
    def __init__(self, p=1, q=0, dtype=torch.float64, device='cpu'):
        r"""
        :param (p, q): labels of highest weight for su(3) representations. (1, 0) - defining representation
        :param dtype: data type of matrix representation of operators
        :param device: device on which the torch.tensor objects are stored
        :type J: int
        :type dtype: torch.dtype
        :type device: int

        Build the defining representation "3" of su(3) Lie algebra using the Cartan-Weyl basis, $\lambda_i$.

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
        self.p = p
        self.q = q
        self.dtype = dtype
        self.device = device

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: torch.tensor
        """
        return get_op("I", dtype=self.dtype, device=self.device)

    def TZ(self):
        r"""
        :return: :math:`T^z` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tz", dtype=self.dtype, device=self.device)

    def Y(self):
        r"""
        :return: :math:`Y` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("y", dtype=self.dtype, device=self.device)

    def TP(self):
        r"""
        :return: :math:`T^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tp", dtype=self.dtype, device=self.device)

    def TM(self):
        r"""
        :return: :math:`T^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("tm", dtype=self.dtype, device=self.device)

    def VP(self):
        r"""
        :return: :math:`V^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("vp", dtype=self.dtype, device=self.device)

    def VM(self):
        r"""
        :return: :math:`V^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("vm", dtype=self.dtype, device=self.device)

    def UP(self):
        r"""
        :return: :math:`U^+` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("up", dtype=self.dtype, device=self.device)

    def UM(self):
        r"""
        :return: :math:`U^-` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("um", dtype=self.dtype, device=self.device)

    def C1(self):
        r"""
        :return: SU(3) spin-spin interaction as rank-4 for tensor
        :rtype: torch.tensor
        """
        expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \vec{S}_1.\vec{S}_2 between spins on sites 1 and 2
        # First as rank-4 tensor
        C1 = einsum(expr_kron, self.TZ(), self.TZ()) + 0.75 * einsum(expr_kron, self.Y(), self.Y())\
            + 0.5 * (einsum(expr_kron, self.TP(), self.TM()) + einsum(expr_kron, self.TM(), self.TP())
                     + einsum(expr_kron, self.VP(), self.VM()) + einsum(expr_kron, self.VM(), self.VP())
                     + einsum(expr_kron, self.UP(), self.UM()) + einsum(expr_kron, self.UM(), self.UP()))
        return C1


def get_op(op, dtype=torch.complex128, device='cpu', dbg=False):
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
