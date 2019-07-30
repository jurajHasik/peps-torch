import torch
import config as cfg
from ipeps import IPEPS

class ENV(torch.nn.Module):
    def __init__(self, chi, ipeps, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param ipeps: wavefunction
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type chi: int
        :type ipeps: IPEPS
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS

        For each pair (coord, site) create corresponding half-row/column tensors T's 
        and corner tensors C's. The corner tensors have dimensions :math:`\chi \times \chi`
        and the half-row/column tensors have dimensions :math:`\chi \times \chi \times D^2` 
        (D might vary depending on the corresponding dimension of on-site tensor)::

            y\x -1 0 1
             -1  C T C
              0  T A T
              1  C T C 
        
        The environment tensors of an ENV object ``e`` are accesed through members ``C`` and ``T`` 
        by providing a tuple of coordinates and directional vector to the environment tensor:: 
            
            coord=(0,0)                # tuple(x,y) identifying vertex on the square lattice
            rel_dir_vec_C=(-1,-1)      # tuple(rx,ry) identifying one of the four corner tensors
            rel_dir_vec_T=(-1,0)       # tuple(rx,ry) identifying one of the four half-row/column tensors
            C_upper_left= e.C[(coord,rel_dir_vec_C)] # return upper left corner tensor of site at coord
            T_left= e.T[(coord,rel_dir_vec_T)]       # return left half-row tensor of site at coord

        The directional vectors identifying individual tensors making up the environment of 
        a site are defined relative to the position of the site: `coord(environment tensor) - coord(A)`::

            C(-1,-1)   T        (1,-1)C 
                       |(0,-1)
            T--(-1,0)--A(0,0)--(1,0)--T 
                       |(0,1)
            C(-1,1)    T         (1,1)C
        
        The index-position convention is as follows: 
        Start from the index in the **direction "up"** <=> (0,-1) and continue **anti-clockwise**::
        
            C--1 0--T--2 0--C
            |       |       |
            0       1       1
            0               0
            |               |
            T--2         1--T
            |               |
            1               2
            0       0       0
            |       |       |
            C--1 1--T--2 1--C
        """
        super(ENV, self).__init__()
        self.dtype = global_args.dtype
        self.device = global_args.device
        self.chi = chi

        # initialize environment tensors
        self.C = dict()
        self.T = dict()

        for coord, site in ipeps.sites.items():
            #for vec in [(0,-1), (-1,0), (0,1), (1,0)]:
            #    self.T[(coord,vec)]="T"+str(ipeps.site(coord))
            self.T[(coord,(0,-1))]=torch.empty((self.chi,site.size()[1]*site.size()[1],self.chi), dtype=self.dtype, device=self.device)
            self.T[(coord,(-1,0))]=torch.empty((self.chi,self.chi,site.size()[2]*site.size()[2]), dtype=self.dtype, device=self.device)
            self.T[(coord,(0,1))]=torch.empty((site.size()[3]*site.size()[3],self.chi,self.chi), dtype=self.dtype, device=self.device)
            self.T[(coord,(1,0))]=torch.empty((self.chi,site.size()[4]*site.size()[4],self.chi), dtype=self.dtype, device=self.device)

            #for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            #     self.C[(coord,vec)]="C"+str(ipeps.site(coord))
            for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                self.C[(coord,vec)]=torch.empty((self.chi,self.chi), dtype=self.dtype, device=self.device)

def init_env(ipeps, env, ctm_args=cfg.ctm_args):
    """
    :param ipeps: wavefunction
    :param env: CTM environment
    :param ctm_args: CTM algorithm configuration
    :type ipeps: IPEPS
    :type env: ENV 
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    inside `CTMARGS.ctm_env_init_type` [TODO link here]
    
 
    * CONST - all C and T tensors have all their elements intialized to a value 1
    * RANDOM - all C and T tensors have elements with random numbers drawn from uniform
      distribution [0,1)
    * CTMRG - tensors C and T are built from the on-site tensors of `ipeps` 
    """
    if ctm_args.ctm_env_init_type=='CONST':
        init_const(env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='RANDOM':
        init_random(env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='CTMRG':
        init_from_ipeps(ipeps, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+str(ctm_args.ctm_env_init_type))

def init_const(env, verbosity=0):
    for key,t in env.C.items():
        env.C[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)
    for key,t in env.T.items():
        env.T[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)

# TODO restrict random corners to have pos-semidef spectrum
def init_random(env, verbosity=0):
    for key,t in env.C.items():
        env.C[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)
    for key,t in env.T.items():
        env.T[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)

# TODO handle case when chi < bond_dim^2
def init_from_ipeps(ipeps, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps")
    for coord, site in ipeps.sites.items():
        for rel_vec in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            env.C[(coord,rel_vec)] = torch.zeros(env.chi,env.chi, dtype=env.dtype, device=env.device)

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
        vec = (-1,-1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mijef,mijab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
        env.C[(coord,vec)][:dimsA[3]**2,:dimsA[4]**2]=a/torch.max(torch.abs(a))

        # right-upper corner
        #
        #     i      = 0--C     
        # 1--A--j         1
        #   /\
        #  2  m
        #      \ i
        #    1--A--j
        #      /
        #     2
        vec = (1,-1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('miefj,miabj->eafb',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2)
        env.C[(coord,vec)][:dimsA[2]**2,:dimsA[3]**2]= a/torch.max(torch.abs(a))

        # right-lower corner
        #
        #     0      =    0     
        # 1--A--j      1--C
        #   /\
        #  i  m
        #      \ 0
        #    1--A--j
        #      /
        #     i
        vec = (1,1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mefij,mabij->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2)
        env.C[(coord,vec)][:dimsA[1]**2,:dimsA[2]**2]=a/torch.max(torch.abs(a))

        # left-lower corner
        #
        #     0      = 0     
        # i--A--3      C--1
        #   /\
        #  j  m
        #      \ 0
        #    i--A--3
        #      /
        #     j
        vec = (-1,1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('meijf,maijb->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[4]**2)
        env.C[(coord,vec)][:dimsA[1]**2,:dimsA[4]**2]=a/torch.max(torch.abs(a))

        # upper transfer matrix
        #
        #     i      = 0--T--2     
        # 1--A--3         1
        #   /\
        #  2  m
        #      \ i
        #    1--A--3
        #      /
        #     2
        vec = (0,-1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('miefg,miabc->eafbgc',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2, dimsA[4]**2)
        env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[3]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:dimsA[2]**2,:,:dimsA[4]**2]=a/torch.max(torch.abs(a))

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
        vec = (-1,0)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('meifg,maibc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[3]**2, dimsA[4]**2)
        env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:dimsA[1]**2,:dimsA[3]**2,:]=a/torch.max(torch.abs(a))

        # lower transfer matrix
        #
        #     0      =    0     
        # 1--A--3      1--T--2
        #   /\
        #  i  m
        #      \ 0
        #    1--A--3
        #      /
        #     i
        vec = (0,1)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mefig,mabic->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[4]**2)
        env.T[(coord,vec)] = torch.zeros((dimsA[1]**2,env.chi,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:,:dimsA[2]**2,:dimsA[4]**2]=a/torch.max(torch.abs(a))

        # right transfer matrix
        #
        #     0      =    0     
        # 1--A--i      1--T
        #   /\            2
        #  2  m
        #      \ 0
        #    1--A--i
        #      /
        #     2
        vec = (1,0)
        A = ipeps.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = torch.einsum('mefgi,mabci->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2)
        env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[2]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:dimsA[1]**2,:,:dimsA[3]**2]=a/torch.max(torch.abs(a))

def print_env(env, verbosity=0):
    print("dtype "+str(env.dtype))
    print("device "+str(env.device))

    for key,t in env.C.items():
        print(str(key)+" "+str(t.size()))
        if verbosity>0: 
            print(t)
    for key,t in env.T.items():
        print(str(key)+" "+str(t.size()))
        if verbosity>0:
            print(t)
