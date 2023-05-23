import yastn.yastn as yastn
import numpy as np
from math import factorial, sqrt

class SU2_NOSYM():
    _REF_S_DIRS=(-1,1)

    def __init__(self, settings, J):
        r"""
        :param J: dimension of irrep
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
        assert settings.sym.NSYM==0, "No abelian symmetry is assumed"
        self.J = J
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.default_dtype
        self.device= settings.device if hasattr(settings, 'device') else settings.default_device

    def _cast(self, op_id, J, dtype, device):
        tmp_block= self.get_op(op_id, J, dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS)
        op.set_block(Ds=tmp_block.shape, val=tmp_block)
        op= op.to(device)
        return op

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: yastn.Tensor
        """
        return self._cast("I",self.J,self.dtype,self.device)

    def SZ(self):
        r"""
        :return: :math:`S^z` operator of irrep
        :rtype: yastn.Tensor
        """
        return self._cast("sz",self.J,self.dtype,self.device)

    def SP(self):
        r"""
        :return: :math:`S^+` operator of irrep.
        :rtype: yastn.Tensor
        """
        return self._cast("sp",self.J,self.dtype,self.device)

    def SM(self):
        r"""
        :return: :math:`S^-` operator of irrep.
        :rtype: yastn.Tensor
        """
        return self._cast("sm",self.J,self.dtype,self.device)

    def BP_rot(self):
        tmp_block= self.get_op("rot", self.J, self.dtype)
        op= yastn.Tensor(self.engine, s=[1,1])
        op.set_block(Ds=tmp_block.shape, val=tmp_block)
        op= op.to(self.device)
        return op

    def S_zpm(self):
        # 1(-1)
        # S--0(-1)
        # 2(+1)
        op= yastn.Tensor(self.engine, s=[-1]+list(self._REF_S_DIRS))
        tmp_block= np.zeros((3,self.J,self.J), dtype=self.dtype)
        tmp_block[0,:,:]= self.get_op("sz", self.J, self.dtype)
        tmp_block[1,:,:]= self.get_op("sp", self.J, self.dtype)
        tmp_block[2,:,:]= self.get_op("sm", self.J, self.dtype)
        op.set_block(Ds=tmp_block.shape, val=tmp_block)
        return op

    # TODO: implement xyz for Sx and Sy terms
    def SS(self, zpm=(1.,1.,1.)):
        r"""
        :param zpm: coefficients of anisotropy of spin-spin interaction
                    zpm[0]*(S^z S^z) + zpm[1]*(S^p S^m)/2 + zpm[2]*(S^m S^p)/2
        :type zpm: tuple(float)
        :return: spin-spin interaction as rank-4 for tensor 
        :rtype: yastn.Tensor
        """
        # expr_kron = 'ij,ab->iajb'
        # spin-spin interaction \vec{S}_1.\vec{S}_2 between spins on sites 1 and 2
        # First as rank-4 tensor
        # SS = xyz[0]*np.einsum(expr_kron,get_op("sz", J, self.dtype),get_op("sz", J, self.dtype)) \
        #     + 0.5*(np.einsum(expr_kron,get_op("sp", J, self.dtype),get_op("sm", J, self.dtype)) \
        #     + np.einsum(expr_kron,get_op("sm", J, self.dtype),get_op("sp", J, self.dtype)))
        S_vec= self.S_zpm()
        S_vec_dag= S_vec.conj().transpose((0,2,1))
        g= yastn.Tensor(self.engine, s=self._REF_S_DIRS)
        tmp_block= np.diag(np.asarray([1.,0.5,0.5])*np.asarray(zpm, dtype=self.dtype))
        g.set_block(Ds=tmp_block.shape, val=tmp_block)
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
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param J: dimension of irrep
        :type J: int

        Build a representation J of SU(2) group. The J corresponds to (physics) 
        spin irrep notation as spin :math:`S = (J-1)/2`. This representation
        uses explicit U(1) symmetry (subgroup) making all operators/tensors block-sparse.  

        The signature convention :math:`O = \sum_{ij} O_{ij}|i\rangle\langle j|` is -1 for 
        index `i` (:math:`|ket\rangle`) and +1 for index `j` (:math:`\langle bra|`).

        The raising and lowering operators are defined as:

        .. math::
            
            \begin{align*}
            S^+ &=S^x+iS^y               & S^x &= 1/2(S^+ + S^-)\\
            S^- &=S^x-iS^y\ \Rightarrow\ & S^y &=-i/2(S^+ - S^-)
            \end{align*}
        """
        assert settings.sym.NSYM==1, "U(1) abelian symmetry is assumed"
        self.J, self.HW = J, (J-1) # HW highest weight state in mathematical notation
                                   # spin (J-1)/2 irrep with dim J has HW J-1 with states
                                   # spaced by 2 i.e.
                                   # spin 1/2, dim 2, HW 1, irrep 1, -1
                                   # spin 1,   dim 3, HW 2, irrep 2, 0, -2
                                   # spin 3/2, dim 4, HW 3, irrep 3, 1, -1, 3 
        self.engine= settings
        self.backend= settings.backend
        self.dtype= settings.default_dtype
        self.device= 'cpu' if not hasattr(settings, 'device') else settings.device

    def I(self):
        r"""
        :return: Identity operator of irrep
        :rtype: yastn.Tensor
        """
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=0)
        for j in range(-self.HW,self.HW+1,2):
            c= (j,j)
            op.set_block(ts=c, Ds=(1,1), val='ones')
        op= op.to(self.device)
        return op

    def SZ(self):
        r"""
        :return: :math:`S^z` operator of irrep
        :rtype: yastn.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=0)
        for j in range(-self.HW,self.HW+1,2):
            c= (j,j)
            op.set_block(ts=c, Ds=unit_block.shape, val= (0.5*j)*unit_block)
        op= op.to(self.device)
        return op

    def SP(self):
        r"""
        :return: :math:`S^+` operator of irrep.
        :rtype: yastn.Tensor

        The :math:`S^+` operator maps states with :math:`S^z = x` to states with
        :math:`S^z = x+1` . Therefore as a matrix it must act as follows
        on vector of basis elements of spin-S representation (in
        this particular order) :math:`|S M\rangle` ::
        
                |-S  >    C_+|-S+1>           0 1 0 0 ... 0
            S^+ |-S+1>  = C_+|-S+2>  => S^+ = 0 0 1 0 ... 0 x C_+
                 ...         ...                   ...
                | S-1>    C_+| S  >           0    ...  0 1
                | S  >        0               0    ...  0 0  
        
        where :math:`C_+ = \sqrt{S(S+1)-M(M+1)}`.
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=-2)
        for j in range(-self.HW,self.HW,2):
            c= (j+2,j) #if j%2==1 else (j//2+1,j//2)
            c_p= sqrt(0.5 * self.HW * (0.5 * self.HW + 1) - 0.5*j * (0.5*j + 1))
            op.set_block(ts=c, Ds=unit_block.shape, val= c_p*unit_block)
        op= op.to(self.device) 
        return op

    def SM(self):
        r"""
        :return: :math:`S^-` operator of irrep.
        :rtype: yastn.Tensor

        The :math:`S^-` operator maps states with :math:`S^z = x` to states with
        :math:`S^z = x-1` . Therefore as a matrix it must act as follows
        on vector of basis elements of spin S representation (in
        this particular order) :math:`|S M\rangle` ::
        
                |-S  >     0                  0 0 0 0 ... 0
            S^- |-S+1>  = C_-|-S  >  => S^- = 1 0 0 0 ... 0 x C_-
                 ...         ...              ...
                | S-1>    C_-| S-2>           0   ... 1 0 0
                | S  >    C_-| S-1>           0   ... 0 1 0
        
        where :math:`C_- = \sqrt{S(S+1)-M(M-1)}`.
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        op= yastn.Tensor(self.engine, s=self._REF_S_DIRS, n=2)
        for j in range(-self.HW+2,self.HW+1,2):
            c= (j-2,j) #if j%2==1 else (j//2,j//2-1)
            c_p= sqrt(0.5 * self.HW * (0.5 * self.HW + 1) - 0.5*j * (0.5*j - 1))
            op.set_block(ts=c, Ds=unit_block.shape, val= c_p*unit_block)
        op= op.to(self.device) 
        return op

    def BP_rot(self):
        raise NotImplementedError("AFM rotation operator is not compatible with U(1) symmetry")    
    
    def S_zpm(self):
        r"""
        :return: vector of su(2) generators as rank-3 tensor 
        :rtype: yastn.Tensor
        
        Returns vector with representation of su(2) generators, in order: :math:`S^z, S^+, S^-`.
        The generators are indexed by first index of the resulting rank-3 tensors.
        Signature convention is::    

            1(-1)
            S--0(-1)
            2(+1)
        """
        op_v= yastn.block({i: t.add_leg(axis=0,s=-1) for i,t in enumerate([\
            self.SZ(), self.SP(), self.SM()])}, common_legs=[1,2]).drop_leg_history(axes=0)
        return op_v

    # TODO: implement xyz for Sx and Sy terms
    def SS(self, zpm=(1.,1.,1.)):
        r"""
        :param zpm: coefficients of anisotropy of spin-spin interaction
                    zpm[0]*(S^z S^z) + zpm[1]*(S^p S^m)/2 + zpm[2]*(S^m S^p)/2
        :type zpm: tuple(float)
        :return: spin-spin interaction as rank-4 for tensor 
        :rtype: yastn.Tensor
        """
        unit_block= np.ones((1,1), dtype=self.dtype)
        g= yastn.Tensor(self.engine, s=self._REF_S_DIRS)
        g.set_block(ts=(2,2), Ds=unit_block.shape, val=zpm[1]/2*unit_block)
        g.set_block(ts=(0,0), Ds=unit_block.shape, val=zpm[0]*unit_block)
        g.set_block(ts=(-2,-2), Ds=unit_block.shape, val=zpm[2]/2*unit_block)
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