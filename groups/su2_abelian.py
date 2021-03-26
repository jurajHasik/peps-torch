import yamps.yast as yast
import numpy as np
from math import factorial, sqrt

class SU2_NOSYM():
    _REF_S_DIRS=(-1,1)

    def __init__(self, settings, J):
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
        assert settings.sym.nsym==0, "No abelian symmetry is assumed"
        self.J = J
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.dtype
        self.device= 'cpu' if not hasattr(settings, 'device') else settings.device

    def _cast(self, op_id, J, dtype, device):
        tmp_block= self.get_op(op_id, J, dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS)
        op.set_block(val=tmp_block)
        op= op.to(device)
        return op

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: torch.tensor
        """
        return self._cast("I",self.J,self.dtype,self.device)

    def SZ(self):
        r"""
        :return: :math:`S^z` operator of irrep
        :rtype: torch.tensor
        """
        return self._cast("sz",self.J,self.dtype,self.device)

    def SP(self):
        r"""
        :return: :math:`S^+` operator of irrep.
        :rtype: torch.tensor
        """
        return self._cast("sp",self.J,self.dtype,self.device)

    def SM(self):
        r"""
        :return: :math:`S^-` operator of irrep.
        :rtype: torch.tensor
        """
        return self._cast("sm",self.J,self.dtype,self.device)

    def BP_rot(self):
        tmp_block= self.get_op("rot", self.J, self.dtype)
        op= yast.Tensor(self.engine, s=[1,1])
        op.set_block(val=tmp_block)
        op= op.to(self.device)
        return op

    def S_zpm(self):
        # 1(-1)
        # S--0(-1)
        # 2(+1)
        op= yast.Tensor(self.engine, s=[-1]+list(self._REF_S_DIRS))
        tmp_block= np.zeros((3,self.J,self.J), dtype=self.dtype)
        tmp_block[0,:,:]= self.get_op("sz", self.J, self.dtype)
        tmp_block[1,:,:]= self.get_op("sp", self.J, self.dtype)
        tmp_block[2,:,:]= self.get_op("sm", self.J, self.dtype)
        op.set_block(val=tmp_block)
        return op

    # TODO: implement xyz for Sx and Sy terms
    def SS(self, xyz=(1.,0.5,0.5)):
        r"""
        :param xyz: coefficients of anisotropy of spin-spin interaction
                    xyz[0]*(S^z S^z) + xyz[1]*(S^x S^x) + xyz[2]*(S^y S^y)
        :type xyz: tuple(float)
        :return: spin-spin interaction as rank-4 for tensor 
        :rtype: torch.tensor
        """
        # expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \vec{S}_1.\vec{S}_2 between spins on sites 1 and 2
        # First as rank-4 tensor
        # SS = xyz[0]*np.einsum(expr_kron,get_op("sz", J, self.dtype),get_op("sz", J, self.dtype)) \
        #     + 0.5*(np.einsum(expr_kron,get_op("sp", J, self.dtype),get_op("sm", J, self.dtype)) \
        #     + np.einsum(expr_kron,get_op("sm", J, self.dtype),get_op("sp", J, self.dtype)))
        S_vec= self.S_zpm()
        S_vec_dag= S_vec.conj().transpose((0,2,1))
        g= yast.Tensor(self.engine, s=self._REF_S_DIRS)
        tmp_block= np.diag(np.asarray(xyz, dtype=self.dtype))
        g.set_block(val=tmp_block)
        #
        # 1->0
        # S--0(-1) (+1)1--g--0->2(-1)
        # 2->1
        SS= S_vec.tensordot(g,([0],[1]))
        
        #
        # 0          1->2
        # S--g--2 0--S
        # 1          2->3
        SS= SS.tensordot(S_vec_dag,([2],[0]))
        SS= SS.transpose((0,2,1,3))
        return SS

    @staticmethod
    def get_op(op, m, dtype="float64", dbg = False):
  
        if op == "I":
            if dbg:
                print(">>>>> Constructing 1sO: Id <<<<<")
            return np.eye(m, dtype=dtype)
        
        elif op == "sz":
            if dbg:
                print(">>>>> Constructing 1sO: Sz <<<<<")
            res= np.zeros((m, m), dtype=dtype)
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
            res= np.zeros((m, m), dtype=dtype)
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
            res= np.zeros((m, m), dtype=dtype)
            for i in range(1,m):
                res[i, i - 1] = sqrt(0.5 * (m - 1) * (0.5 * (m - 1) + 1) - \
                         (-0.5 * (m - 1) + i) * \
                           (-0.5 * (m - 1) + i - 1))
            return res
        elif op == "rot":
            res = np.zeros((m, m), dtype=dtype)
            for i in range(m):
                res[i,m-1-i] = (-1) ** i
            return res
        else:
            raise Exception("Unsupported operator requested: "+op) 

class SU2_U1():
    _REF_S_DIRS=(-1,1)

    def __init__(self, settings, J):
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
        assert settings.sym.nsym==1, "U(1) abelian symmetry is assumed"
        self.J = J
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.dtype
        self.device= 'cpu' if not hasattr(settings, 'device') else settings.device

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: torch.tensor
        """
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=0)
        for j in range(-self.J,self.J+1,2):
            c= (j,j) #if j%2==1 else (j//2,j//2)
            op.set_block(ts=c, Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def SZ(self):
        r"""
        :return: :math:`S^z` operator of irrep
        :rtype: torch.tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=0)
        for j in range(-self.J,self.J+1,2):
            c= (j,j) #if j%2==1 else (j//2,j//2)
            op.set_block(ts=c, val= (0.5*j)*unit_block)
        op= op.to(self.device)
        return op

    def SP(self):
        r"""
        :return: :math:`S^+` operator of irrep.
        :rtype: torch.tensor

        The s^+ operator maps states with s^z = x to states with
        s^z = x+1 . Therefore as a matrix it must act as follows
        on vector of basis elements of spin S representation (in
        this particular order) |S M>
        
            |-S  >    C_+|-S+1>           0 1 0 0 ... 0
        s^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
             ...         ...              ...
            | S-1>    C_+| S  >           0    ...  0 1
            | S  >     0                  0    ...  0 0
        
        where C_+ = sqrt(S(S+1)-M(M+1))
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=-2)
        for j in range(-self.J,self.J,2):
            c= (j+2,j) #if j%2==1 else (j//2+1,j//2)
            c_p= sqrt(0.5 * self.J * (0.5 * self.J + 1) - 0.5*j * (0.5*j + 1))
            op.set_block(ts=c, val= c_p*unit_block)
        op= op.to(self.device) 
        return op

    def SM(self):
        r"""
        :return: :math:`S^-` operator of irrep.
        :rtype: torch.tensor

        The s^- operator maps states with s^z = x to states with
        s^z = x-1 . Therefore as a matrix it must act as follows
        on vector of basis elements of spin S representation (in
        this particular order) |S M>
        
            |-S  >     0                  0 0 0 0 ... 0
        s^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
             ...         ...              ...
            | S-1>    C_-| S-2>           0   ... 1 0 0
            | S  >    C_-| S-1>           0   ... 0 1 0
        
        where C_- = sqrt(S(S+1)-M(M-1))
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yast.Tensor(self.engine, s=self._REF_S_DIRS, n=2)
        for j in range(-self.J+2,self.J+1,2):
            c= (j-2,j) #if j%2==1 else (j//2,j//2-1)
            c_p= sqrt(0.5 * self.J * (0.5 * self.J + 1) - 0.5*j * (0.5*j - 1))
            op.set_block(ts=c, val= c_p*unit_block)
        op= op.to(self.device) 
        return op

    def BP_rot(self):
        raise NotImplementedError("AFM rotation operator is not compatible with U(1) symmetry")    
    
    def S_zpm(self):
        # 1(-1)
        # S--0(-1)
        # 2(+1)
        op_v= yast.Tensor(self.engine, s=[-1]+list(self._REF_S_DIRS), n=0)
        op_sz, op_sp, op_sm= self.SZ(), self.SP(), self.SM()
        for op in [op_sz, op_sp, op_sm]:
            for c in op.A:
                op_v.set_block(ts=(op.n[0],*c), val=op.A[c][None,:,:])
        return op_v

    # TODO: implement xyz for Sx and Sy terms
    def SS(self, zpm=(1.,0.5,0.5)):
        r"""
        :param zpm: coefficients of anisotropy of spin-spin interaction
                    zpm[0]*(S^z S^z) + zpm[1]*(S^p S^m) + zpm[2]*(S^m S^p)
        :type zpm: tuple(float)
        :return: spin-spin interaction as rank-4 for tensor 
        :rtype: torch.tensor
        """
        # expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \vec{S}_1.\vec{S}_2 between spins on sites 1 and 2
        # First as rank-4 tensor
        # SS = xyz[0]*np.einsum(expr_kron,get_op("sz", J, self.dtype),get_op("sz", J, self.dtype)) \
        #     + 0.5*(np.einsum(expr_kron,get_op("sp", J, self.dtype),get_op("sm", J, self.dtype)) \
        #     + np.einsum(expr_kron,get_op("sm", J, self.dtype),get_op("sp", J, self.dtype)))
        unit_block= np.ones((1,1), dtype=self.dtype)
        g= yast.Tensor(self.engine, s=self._REF_S_DIRS)
        g.set_block(ts=(2,2), val=zpm[1]*unit_block)
        g.set_block(ts=(0,0), val=zpm[0]*unit_block)
        g.set_block(ts=(-2,-2), val=zpm[2]*unit_block)
        g= g.to(self.device)

        S_vec= self.S_zpm()
        S_vec_dag= S_vec.conj().transpose((0,2,1))
        #
        # 1->0
        # S--0(-1) (+1)1--g--0->2(-1)
        # 2->1
        SS= S_vec.tensordot(g,([0],[1]))
        
        #
        # 0          1->2
        # S--g--2 0--S
        # 1          2->3
        SS= SS.tensordot(S_vec_dag,([2],[0]))
        SS= SS.transpose((0,2,1,3))
        return SS

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
