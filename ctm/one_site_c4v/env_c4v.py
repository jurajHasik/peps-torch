import torch
import config as cfg
from ipeps.ipeps_c4v import IPEPS_C4V
from linalg.custom_eig import truncated_eig_sym
import pdb

class ENV_C4V():
    def __init__(self, chi, state=None, bond_dim=None, ctm_args=cfg.ctm_args, 
        global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param state: wavefunction
        :param bond_dim2: bond dimension
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type chi: int
        :type state: IPEPS_C4V
        :type bond_dim: int
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
        
            C--1 0--T--1 0--C
            |       |       |
            0       2       1
            0               0
            |               |
            T--2         2--T
            |               |
            1               1
            0       2       0
            |       |       |
            C--1 0--T--1 1--C

        All C's and T's in the above diagram are identical and they are symmetric under the exchange of
        environment bond indices :math:`C_{ij}=C_{ji}` and :math:`T_{ija}=C_{jia}`.  
        """
        assert state or bond_dim, "either state or bond_dim must be supplied"
        if state:
            assert len(state.sites)==1, "Not a 1-site ipeps"
            site= next(iter(state.sites.values()))
            assert site.size()[1]==site.size()[1]==site.size()[1]==site.size()[1],\
                "bond dimensions of on-site tensor are not equal"
            bond_dim= site.size()[1]
        super(ENV_C4V, self).__init__()
        self.dtype= global_args.dtype
        self.device= global_args.device
        self.chi= chi
        self.bond_dim= bond_dim

        # initialize environment tensors
        # The same structure is preserved as for generic ipeps ``ENV``. We store keys for access
        # to dummy dicts ``C`` and ``T``
        self.keyC= ((0,0),(-1,-1))
        self.keyT= ((0,0),(-1,0))
        self.C= dict()
        self.T= dict()

        self.C[self.keyC]= torch.zeros((self.chi,self.chi), dtype=self.dtype, device=self.device)
        self.T[self.keyT]= torch.zeros((self.chi,self.chi,bond_dim**2), \
            dtype=self.dtype, device=self.device)

    def get_C(self):
        r"""
        :return: get corner tensor
        :rtype: torch.Tensor
        """
        return self.C[self.keyC]

    def get_T(self):
        r"""
        :return: get half-row/-column tensor
        :rtype: torch.Tensor
        """
        return self.T[self.keyT]

    def clone(self, ctm_args=cfg.ctm_args, global_args=cfg.global_args, \
        requires_grad=False):
        new_env= ENV_C4V(self.chi, bond_dim=self.bond_dim, ctm_args=ctm_args, \
            global_args=global_args)
        new_env.C[new_env.keyC]= self.get_C().detach().clone()
        new_env.T[new_env.keyT]= self.get_T().detach().clone()
        return new_env

    def extend(self, new_chi, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        new_env= ENV_C4V(new_chi, bond_dim=self.bond_dim, ctm_args=ctm_args, \
            global_args=global_args)
        x= min(self.chi, new_chi)
        new_env.C[new_env.keyC][:x,:x]= self.get_C()[:x,:x]
        new_env.T[new_env.keyT][:x,:x,:self.bond_dim**2]= \
            self.get_T()[:x,:x,:self.bond_dim**2]
        return new_env

    def move_to(self, device):
        if device=='cpu' or device==torch.device('cpu'):
            self.C[self.keyC]= self.get_C().to(device)
            self.T[self.keyT]= self.get_T().to(device)
        elif device.type=='cuda':
            self.C[self.keyC]= self.get_C().to(device)
            self.T[self.keyT]= self.get_T().to(device)
        else:
            raise RuntimeError(f"Unsupported device {device}")

def init_env(state, env, ctm_args=cfg.ctm_args):
    """
    :param state: wavefunction
    :param env: C4v symmetric CTM environment
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    inside :class:`CTMARGS.ctm_env_init_type <config.CTMARGS>`
    
 
    * PROD - C and T tensors have just a single element intialized to a value 1
             corresponding to product state environment
    * RANDOM - C and T tensors have elements with random numbers drawn from uniform
      distribution [0,1)
    * CTMRG - tensors C and T are built from the on-site tensor of `state` 
    """
    if ctm_args.ctm_env_init_type=='PROD':
        init_prod(state, env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='RANDOM':
        init_random(env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='CTMRG':
        init_from_ipeps_pbc(state, env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='CTMRG_OBC':
        init_from_ipeps_obc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "\
            +str(ctm_args.ctm_env_init_type))

def init_prod(state, env, verbosity=0):
    for key,t in env.C.items():
        env.C[key]= torch.zeros(t.size(), dtype=env.dtype, device=env.device)
        env.C[key][0,0]= 1

    A= next(iter(state.sites.values()))
    # left transfer matrix
    #
    #     0      = 0     
    # i--A--j      T
    #   /\         1
    #  2  m
    #      \ 0
    #    i--A--j
    #      /
    #     2
    a = torch.einsum('meifj,maibj->eafb',(A,A)).contiguous().view(\
        A.size()[1]**2, A.size()[3]**2)
    a= a/torch.max(torch.abs(a))
    # check symmetry
    a_asymm_norm= torch.norm(a.t()-a)
    assert a_asymm_norm/torch.abs(a).max() < 1.0e-8, "a is not symmetric"
    D, U= truncated_eig_sym(a, 2)
    # leading eigenvector is unique 
    assert torch.abs(D[0]-D[1]) > 1.0e-8, "Leading eigenvector of T not unique"
    for key,t in env.T.items():
        env.T[key]= torch.zeros(t.size(), dtype=env.dtype, device=env.device)
        env.T[key][0,0,:]= U[:,0]

# TODO restrict random corners to have pos-semidef spectrum
def init_random(env, verbosity=0):
    for key,t in env.C.items():
        tmpC= torch.rand(t.size(), dtype=env.dtype, device=env.device)
        env.C[key]= 0.5*(tmpC+tmpC.t())
    for key,t in env.T.items():
        env.T[key]= torch.rand(t.size(), dtype=env.dtype, device=env.device)

# TODO handle case when chi < bond_dim^2
def init_from_ipeps_pbc(state, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps_pbc")

    # Left-upper corner
    #
    #     i      = C--1     
    # j--A--3      0
    #   /\
    #  2  m
    #      \ i
    #    j--A--3
    #      /
    #     2
    A= next(iter(state.sites.values()))
    dimsA= A.size()
    a= torch.einsum('mijef,mijab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
    a= a/torch.max(torch.abs(a))
    
    a_asymm_norm= torch.norm(a.t()-a)
    assert a_asymm_norm/torch.abs(a).max() < 1.0e-8, "a is not symmetric"
    D, U= truncated_eig_sym(a, a.size()[0])
    a= torch.diag(D)

    env.C[env.keyC]= torch.zeros(env.chi,env.chi, dtype=env.dtype, device=env.device)
    env.C[env.keyC][:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]=\
       a[:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]

    # left transfer matrix
    #
    #     0      = 0     
    # i--A--3      T--2
    #   /\         1
    #  2  m
    #      \ 0
    #    i--A--3
    #      /
    #     2
    a = torch.einsum('meifg,maibc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[3]**2, dimsA[4]**2)
    a= a/torch.max(torch.abs(a))
    
    a= torch.einsum('ai,abs,bj->ijs',U,a,U)

    env.T[env.keyT]= torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
    env.T[env.keyT][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]=\
        a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]

# TODO handle case when chi < bond_dim^2
def init_from_ipeps_obc(state, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps_obc")

    # Left-upper corner
    #
    #     i      = C--1     
    # j--A--3      0
    #   /\
    #  2  m
    #      \ k
    #    l--A--3
    #      /
    #     2
    A= next(iter(state.sites.values()))
    dimsA= A.size()
    a= torch.einsum('mijef,mklab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
    a= a/torch.max(torch.abs(a))
    # check symmetry
    # a_asymm_norm= torch.norm(a.t()-a)
    # assert a_asymm_norm/torch.abs(a).max() < 1.0e-8, "a is not symmetric"
    # D, U= truncated_eig_sym(a, a.size()[0])
    # a= torch.diag(D)
    env.C[env.keyC]= torch.zeros(env.chi,env.chi, dtype=env.dtype, device=env.device)
    env.C[env.keyC][:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]=\
        a[:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]

    # left transfer matrix
    #
    #     0      = 0     
    # i--A--3      T--2
    #   /\         1
    #  2  m
    #      \ 0
    #    k--A--3
    #      /
    #     2
    a= torch.einsum('meifg,makbc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[3]**2, dimsA[4]**2)
    a= a/torch.max(torch.abs(a))
    #a= torch.einsum('ia,abs,bj->ijs',U,a,U)
    env.T[env.keyT]= torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
    env.T[env.keyT][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]=\
        a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]

def print_env(env, verbosity=0):
    print("dtype "+str(env.dtype))
    print("device "+str(env.device))

    print("C "+str(env.C[env.keyC].size()))
    if verbosity>0: 
        print(env.C[env.keyC])
    
    print("T "+str(env.T[env.keyT].size()))
    if verbosity>0:
        print(env.T[env.keyT])

def compute_multiplets(env, eps_multiplet_gap=1.0e-10):
    D= torch.zeros(env.chi+1, dtype=env.dtype, device=env.device)
    D[:env.chi], U= torch.symeig(env.C[env.keyC])
    D, p= torch.sort(torch.abs(D),descending=True)
    m=[]
    l=0
    for i in range(env.chi):
        l+=1
        g=D[i]-D[i+1]
        #print(f"{i} {D[i]} {g}", end=" ")
        if g>eps_multiplet_gap:
            #print(f"{l}", end=" ")
            m.append(l)
            l=0
    return m