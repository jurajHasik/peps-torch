import torch
from math import factorial, sqrt
from tn_interface import einsum

class SU2():
    def __init__(self, J, dtype=torch.float64, device='cpu'):
        r"""
        :param J: highest weight
        :param dtype: data type of matrix representation of operators
        :param device: device on which the torch.tensor objects are stored
        :type J: int
        :type dtype: torch.dtype
        :type device: int

        Build a representation J of SU(2) group. The J corresponds to (physics) 
        spin irrep notation as spin :math:`S = (J-1)/2`.

        The raising and lowering operators are defined as:

        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        self.J = J
        self.dtype=dtype
        self.device=device

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: torch.tensor
        """
        return get_op("I",self.J,dtype=self.dtype,device=self.device)

    def I_N(self,N):
        r"""
        :param N: number of irreps
        :type N: int
        :return: Identity operator over N irreps, i.e. :math:`I_N = \otimes_N I`
                 as rank-2N tensor
        :rtype: torch.Tensor
        """
        I_N= get_op("I",self.J**N,dtype=self.dtype,device=self.device)
        return I_N.view( [self.J]*(2*N) )

    def SZ(self):
        r"""
        :return: :math:`S^z` operator of irrep
        :rtype: torch.tensor
        """
        return get_op("sz",self.J,dtype=self.dtype,device=self.device)

    def SP(self):
        r"""
        :return: :math:`S^+` operator of irrep.
        :rtype: torch.tensor
        """
        return get_op("sp",self.J,dtype=self.dtype,device=self.device)

    def SM(self):
        r"""
        :return: :math:`S^-` operator of irrep.
        :rtype: torch.tensor
        """
        return get_op("sm",self.J,dtype=self.dtype,device=self.device)

    def SY(self):
        r"""
        :return: :math:`S^y` operator of irrep.
        :rtype: torch.tensor
        """
        assert self.dtype in [torch.complex32, torch.complex128],"SY requires complex dtype"
        return -0.5j*(self.SP()-self.SM())

    def BP_rot(self):
        return get_rot_op(self.J,dtype=self.dtype,device=self.device)

    def S(self):
        r"""
        :return: rank-3 tensor containing spin generators [S^z, S^x, S^y]
        :rtype: torch.tensor
        """
        S= torch.zeros(3, self.J, self.J,dtype=self.dtype,device=self.device)
        S[0,:,:]= self.SZ()
        S[1,:,:]= 0.5*(self.SP() + self.SM())
        if S.is_complex():
            S[2,:,:]= -0.5j*(self.SP() - self.SM())
        return S

    # TODO: implement xyz for Sx and Sy terms
    def SS(self, xyz=(1.,1.,1.)):
        r"""
        :param xyz: coefficients of anisotropy of spin-spin interaction
                    xyz[0]*(S^z S^z) + xyz[1]*(S^x S^x) + xyz[2]*(S^y S^y)
        :type xyz: tuple(float)
        :return: spin-spin interaction as rank-4 for tensor 
        :rtype: torch.tensor
        """
        pd = self.J
        expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \vec{S}_1.\vec{S}_2 between spins on sites 1 and 2
        # First as rank-4 tensor
        SS = xyz[0]*einsum(expr_kron,self.SZ(),self.SZ()) \
            + 0.5*xyz[1]*einsum(expr_kron,self.SP(),self.SM()) \
            + 0.5*xyz[2]*einsum(expr_kron,self.SM(),self.SP())
        return SS

def get_op(op, m, dtype=torch.float64, device='cpu', dbg = False):
  
    if op == "I":
        if dbg:
            print(">>>>> Constructing 1sO: Id <<<<<")
        return torch.eye(m, dtype=dtype, device=device)
    
    elif op == "sz":
        if dbg:
            print(">>>>> Constructing 1sO: Sz <<<<<")
        res= torch.zeros((m, m), dtype=dtype, device=device)
        for i in range(m):
            res[i,i] = -0.5 * (-(m - 1) + i*2)
        return res
    
    # The s^+ operator maps states with s^z = x to states with
    # s^z = x+1 . Therefore as a matrix it must act as follows
    # on vector of basis elements of spin S representation (in
    # this particular order) |S M>
    #
    #     |-S  >    C_+|-S+1>           0 1 0 0 ... 0
    # s^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
    #      ...         ...              ...
    #     | S-1>    C_+| S  >           0    ...  0 1
    #     | S  >     0                  0    ...  0 0
    #
    # where C_+ = sqrt(S(S+1)-M(M+1))   
    elif op == "sp":
        if dbg:
            print(">>>>> Constructing 1sO: S^+ <<<<<")
        res= torch.zeros((m, m), dtype=dtype, device=device)
        for i in range(m-1):
            res[i,i+1] = sqrt(0.5 * (m - 1) * (0.5 * (m - 1) + 1) - \
                     (-0.5 * (m - 1) + i) * \
                      (-0.5 * (m - 1) + i + 1))
        return res

    # The s^- operator maps states with s^z = x to states with
    # s^z = x-1 . Therefore as a matrix it must act as follows
    # on vector of basis elements of spin S representation (in
    # this particular order) |S M>
    #
    #     |-S  >     0                  0 0 0 0 ... 0
    # s^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
    #      ...         ...              ...
    #     | S-1>    C_-| S-2>           0   ... 1 0 0
    #     | S  >    C_-| S-1>           0   ... 0 1 0
    #
    # where C_- = sqrt(S(S+1)-M(M-1))
    elif op == "sm":
        if dbg:
            print(">>>>> Constructing 1sO: S^- <<<<<")
        res= torch.zeros((m, m), dtype=dtype, device=device)
        for i in range(1,m):
            res[i, i - 1] = sqrt(0.5 * (m - 1) * (0.5 * (m - 1) + 1) - \
                     (-0.5 * (m - 1) + i) * \
                       (-0.5 * (m - 1) + i - 1))
        return res
    else:
        raise Exception("Unsupported operator requested: "+op)

def get_rot_op(m, dtype=torch.float64, device='cpu'):
    res = torch.zeros((m, m), dtype=dtype, device=device)
    for i in range(m):
        res[i,m-1-i] = (-1) ** i
    return res

# Assume tupples J1=(J1,m1), J2=(J2,m2) and J=(J,m)
# return scalar product of <J,m|J1,m1;J2,m2>
def get_CG(J, J1, J2):
    # (!) Use Dynkin notation to pass desired irreps
    # physical         J_Dynkin = 2*J_physical    Dynkin
    # J=0   m=0                 => J=0 m=0
    # J=1/2 m=-1/2,1/2          => J=1 m=-1,1
    # J=1   m=-1,0,1            => J=2 m=-2,0,2
    # J=3/2 m=-3/2,-1/2,1/2,3/2 => J=3 m=-3,-1,1,2

    cg=0.
    if (J[1] == J1[1] + J2[1]):
        prefactor = sqrt((J[0] + 1.0) * factorial((J[0] + J1[0] - J2[0]) / 2) * \
           factorial((J[0] - J1[0] + J2[0]) / 2) * factorial((J1[0] + J2[0] - J[0]) / 2) /\
           factorial((J1[0] + J2[0] + J[0]) / 2 + 1)) *\
      sqrt(factorial((J[0] + J[1]) / 2) * factorial((J[0] - J[1]) / 2) *\
           factorial((J1[0] - J1[1]) / 2) * factorial((J1[0] + J1[1]) / 2) *\
           factorial((J2[0] - J2[1]) / 2) * factorial((J2[0] + J2[1]) / 2))
    
        min_k = min((J1[0] + J2[0]) // 2, J2[0])
        sum_k = 0
        for k in range(min_k+1):
            if ((J1[0] + J2[0] - J[0]) / 2 - k >= 0) and ((J1[0] - J1[1]) / 2 - k >= 0) and \
                ((J2[0] + J2[1]) / 2 - k >= 0) and ((J[0] - J2[0] + J1[1]) / 2 + k >= 0) and \
                ((J[0] - J1[0] - J2[1]) / 2 + k >= 0): 
                sum_k += ((-1) ** k) / (factorial(k) * factorial((J1[0] + J2[0] - J[0]) / 2 - k) *\
                    factorial((J1[0] - J1[1]) / 2 - k) * factorial((J2[0] + J2[1]) / 2 - k) *\
                    factorial((J[0] - J2[0] + J1[1]) / 2 + k) * factorial((J[0] - J1[0] - J2[1]) / 2 + k))
        cg = prefactor * sum_k
    return cg
