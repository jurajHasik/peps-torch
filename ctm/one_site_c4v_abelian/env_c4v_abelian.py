import config as cfg
import yamps.tensor as TA
import numpy as np

class ENV_C4V_ABELIAN():
    def __init__(self, chi=1, state=None, settings=None, init=False, 
        init_method=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param state: wavefunction
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type chi: int
        :type state: IPEPS_ABELIAN_C4V
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS

        Assuming C4v symmetric single-site ``state`` create corresponding half-row(column) tensor T 
        and corner tensor C. The corner tensor has dimensions :math:`\chi \times \chi`
        and the half-row(column) tensor has dimensions :math:`\chi \times \chi \times D^2`
        with :math:`D` given by ``state`` or ``bond_dim``::

            y\x -1 0 1
             -1  C T C
              0  T A T
              1  C T C 
        
        If both ``state`` and ``bond_dim`` are supplied, the ``bond_dim`` parameter is ignored.

        The environment tensors of an ENV object ``e`` are accesed through members ``C`` and ``T`` 
        The index-position convention is as follows: For upper left C and left T start 
        from the index in the **direction "up"** <=> (-1,0) and continue **anti-clockwise**::
        
            C--1 0--T--1 0--C   C(-1) (+1)T(+1)
            |       |       |  (-1)     (-1)
            0       2       1   
            0               0  (+1)
            |               |   T(-1)
            T--2         2--T  (+1)
            |               |
            1               1
            0       2       0
            |       |       |
            C--1 0--T--1 1--C

        All C's and T's in the above diagram are identical and they are symmetric under the exchange of
        environment bond indices :math:`C_{ij}=C_{ji}` and :math:`T_{ija}=C_{jia}`.  
        """
        if state:
            assert len(state.sites)==1, "Not a 1-site ipeps"
            self.engine= state.engine
            self.backend= state.backend
            self.dtype= state.dtype
            self.nsym = state.nsym
            self.sym= state.sym
        elif settings:
            self.engine= settings
            self.backend= settings.back
            self.dtype= settings.dtype
            self.nsym = settings.nsym
            self.sym= settings.sym
        else:
            raise RuntimeError("Either state or settings must be provided")
        self.device= global_args.device

        self.chi= chi

        # initialize environment tensors
        # The same structure is preserved as for generic ipeps ``ENV``. We store keys for access
        # to dummy dicts ``C`` and ``T``
        self.keyC= ((0,0),(-1,-1))
        self.keyT= ((0,0),(-1,0))
        self.C= dict()
        self.T= dict()

        if init or init_method:
            if not init_method: init_method= ctm_args.ctm_env_init_type 
            if state and init_method in ["CTMRG"]:
                init_env(state, self, init_method)
            else:
                raise RuntimeError("Cannot perform initialization for desired"\
                    +" ctm_env_init_type "+init_method+"."\
                    +" Missing state.")

    def __str__(self):
        s=f"ENV_C4V_abelian chi={self.chi}\n"
        s+=f"dtype {self.dtype}\n"
        s+=f"device {self.device}\n"
        s+=f"nsym {self.nsym}\n"
        s+=f"sym {self.sym}\n"
        if len(self.C)==0: s+="C is empty\n"
        for cr,t in self.C.items():
            s+=f"C({cr[0]} {cr[1]}): {t}\n"
        if len(self.T)==0: s+="T is empty\n"
        for cr,t in self.T.items():
            s+=f"T({cr[0]} {cr[1]}): {t}\n"
        return s

    def get_C(self):
        r"""
        :return: get corner tensor
        :rtype: yamps.Tensor
        """
        return self.C[self.keyC]

    def get_T(self):
        r"""
        :return: get half-row/-column tensor
        :rtype: yamps.Tensor
        """
        return self.T[self.keyT]

    def to_dense(self, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :return: returns a copy of the environment with all C,T tensors in their dense 
                 representation. If the environment already has just dense C,T tensors 
                 returns ``self``.
        :rtype: ENV_C4V_ABELIAN

        Create a copy of environment with all on-site tensors as dense possesing no explicit
        block structure (symmetry). This operations preserves gradients on returned
        dense environment.
        """
        if self.nsym==0: return self
        C_dense= {cid: c.to_dense() for cid,c in self.C.items()}
        T_dense= {tid: t.to_dense() for tid,t in self.T.items()}
        env_dense= ENV_C4V_ABELIAN(self.chi, settings=next(iter(C_dense.values())).conf, \
            ctm_args=ctm_args, global_args=global_args)
        env_dense.C= C_dense
        env_dense.T= T_dense
        return env_dense

    def detach(self):
        r"""
        :return: returns a view of the environment with all C,T tensors detached from
                 computational graph.
        :rtype: ENV_ABELIAN

        Create a view of environment with all on-site tensors (their blocks) detached 
        from computational graph. 
        """
        e= ENV_C4V_ABELIAN(self.chi, settings=self.engine)
        e.C= {cid: c.detach() for cid,c in self.C.items()}
        e.T= {tid: t.detach() for tid,t in self.T.items()}
        return e

    def clone(self):
        r"""
        :return: returns a clone of the environment. Cloned tensors C,T are part of the 
                 computational graph.
        :rtype: ENV_ABELIAN

        Create a clone of environment with all tensors (their blocks) part of  
        the computational graph. 
        """
        e= ENV_C4V_ABELIAN(self.chi, settings=self.engine)
        e.C= {cid: c.clone() for cid,c in self.C.items()}
        e.T= {tid: t.clone() for tid,t in self.T.items()}
        return e        

    def compute_multiplets(self, eps_multiplet_gap=1.0e-10):
        return compute_multiplets(self.get_C(), eps_multiplet_gap=eps_multiplet_gap)


def init_env(state, env, init_method=None, ctm_args=cfg.ctm_args):
    """
    :param state: wavefunction
    :param env: C4v symmetric CTM environment
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_C4V_ABELIAN
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    inside :class:`CTMARGS.ctm_env_init_type <config.CTMARGS>`
    
 
    * CONST - C and T tensors have all their elements intialized to a value 1
    * RANDOM - C and T tensors have elements with random numbers drawn from uniform
      distribution [0,1)
    * CTMRG - tensors C and T are built from the on-site tensor of `state` 
    """
    if not init_method: init_method= ctm_args.ctm_env_init_type
    if init_method=='CONST':
        init_const(env, ctm_args.verbosity_initialization)
    elif init_method=='RANDOM':
        init_random(env, ctm_args.verbosity_initialization)
    elif init_method=='CTMRG':
        init_from_ipeps_pbc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+str(ctm_args.ctm_env_init_type))

def init_const(env, verbosity=0):
    raise NotImplementedError
    # env.C[env.keyC]= TA.ones(settings= env.engine, s=[-1,-1], n=0, 
    #     t=([0],[0]), D=([1],[1]))
    # TODO what about T ?

def init_random(env, verbosity=0):
    raise NotImplementedError

def init_from_ipeps_pbc(state, env, verbosity=0):
    r"""
    Technically, we decorate the lattice with a single C4v symmetric tensor and 
    in order to define consistent network - change signature on every sublattice
    B-site::

           V    ^
        -> A <- B ->
           ^    V
        <- B -> A <-
           V    ^

    where the ingoing arrow signifies +1 and outgoing arrow signifies -1 signature.
    Tensors A and B are identical in the content & symmetry sectors (thus they have 
    exactly opposite total charge). The C tensor of tensor A is formed from A,
    while T tensors are formed by B:: 

        C--T--     A--B--
        T--A-- <=> B--A--
        |  |       |  |
    """

    if verbosity>0:
        print("ENV: init_from_ipeps_pbc")

    # Left-upper corner
    #
    #         i(-1)    = C--2(+1),3(-1)->1(-1)  
    # (-1)j--A*--4(-1)   0(+1),1(-1)->0(-1)
    #       /\(-1)
    #  (-1)3  m
    #          \(+1) i(+1)
    #    (+1)j--A--4(+1)
    #          /
    #     (+1)3
    vec = (-1,-1)
    A = state.site()
    ## a= contiguous(einsum('mijef,mijab->eafb',A,conj(A)))
    a= A.dot(A, ((0,1,2), (0,1,2)), conj=(0,1)) # mijef,mijab->efab
    a= a.transpose((0,2,1,3)) # efab->eafb
    ## here we need to group-legs / reshape
    a, lo1= a.group_legs((2,3), new_s=-1) # ea(fb->F)->eaF
    a, lo0= a.group_legs((0,1), new_s=-1) # (ea->E)F->EF
    a= a/a.max_abs()
    a._leg_fusion_data[0]= lo0
    a._leg_fusion_data[1]= lo1
    env.C[env.keyC]= a

    # left transfer matrix
    #
    #     (+1)1             0(-1),1(+1)->0(+1)     
    # (+1)i--A*--4(+1)    = T--4(-1),5(+1)->2(-1)
    #       /\(+1)          2(-1),3(+1)->1(+1)
    #  (+1)3  m
    #          \(-1) 1(-1)
    #    (-1)i--A--4(-1)
    #          /
    #     (-1)3
    vec = (-1,0)
    B= A.negate_signature()
    a= B.dot(B, ((0,2), (0,2)), conj=(0,1)) # meifg,maibc->efgabc
    a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
    a, leg_order_aux= a.group_legs((4,5), new_s=-1) # eafb(gc->G)->eafbG
    a, lo1= a.group_legs((2,3), new_s=1) # ea(fb->F)G->eaFG
    a, lo0= a.group_legs((0,1), new_s=1) # (ea->E)FG->EFG
    a= a/a.max_abs()
    a._leg_fusion_data[0]= lo0
    a._leg_fusion_data[1]= lo1
    a._leg_fusion_data[2]= leg_order_aux
    env.T[env.keyT]=a

def compute_multiplets(C, eps_multiplet_gap=1.0e-10):
    U,S,V= C.split_svd((0,1), sU=1)
    S_dense= np.diag(S.to_numpy())
    chi= len(S_dense)
    D= np.zeros(chi+1, dtype=S_dense.dtype) # numpy sorts in ascending fashion
    D[:chi]= np.sort(S_dense)[::-1]
    m=[]
    l=0
    for i in range(chi):
        l+=1
        g=D[i]-D[i+1]
        #print(f"{i} {D[i]} {g}", end=" ")
        if g>eps_multiplet_gap:
            #print(f"{l}", end=" ")
            m.append(l)
            l=0
    return D[:chi], m