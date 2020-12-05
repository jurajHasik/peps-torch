import config as cfg
from itertools import product

class ENV_ABELIAN():
    def __init__(self, chi=1, state=None, settings=None, init=False,\
        init_method=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param state: wavefunction
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type chi: int
        :type state: IPEPS_ABELIAN
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS

        For each pair of (vertex, on-site tensor) in the elementary unit cell of ``state``, 
        create corresponding environment tensors: Half-row/column tensors T's and corner tensors C's. 
        The corner tensors have dimensions :math:`\chi \times \chi`
        and the half-row/column tensors have dimensions :math:`\chi \times \chi \times D^2` 
        (D might vary depending on the corresponding dimension of on-site tensor). 
        The environment of each double-layer tensor (A) is composed of eight different tensors::

            y\x -1 0 1
             -1  C T C
              0  T A T
              1  C T C 

        The individual tensors making up the environment of a site are defined 
        by four directional vectors :math:`d = (x,y)_{\textrm{environment tensor}} - (x,y)_\textrm{A}`
        as follows::

            C(-1,-1)   T        (1,-1)C 
                       |(0,-1)
            T--(-1,0)--A(0,0)--(1,0)--T 
                       |(0,1)
            C(-1,1)    T         (1,1)C

        These environment tensors of some ENV object ``e`` are accesed through its members ``C`` and ``T`` 
        by providing a tuple of coordinates and directional vector to the environment tensor:: 
            
            coord=(0,0)                # tuple(x,y) identifying vertex on the square lattice
            rel_dir_vec_C=(-1,-1)      # tuple(rx,ry) identifying one of the four corner tensors
            rel_dir_vec_T=(-1,0)       # tuple(rx,ry) identifying one of the four half-row/column tensors
            C_upper_left= e.C[(coord,rel_dir_vec_C)] # return upper left corner tensor of site at coord
            T_left= e.T[(coord,rel_dir_vec_T)]       # return left half-row tensor of site at coord
        
        The index-position convention is as follows: 
        Start from the index in the **direction "up"** <=> (0,-1) and continue **anti-clockwise**.
        The reference symmetry signatures are shown on the right::

            C--1 0--T--2 0--C        C(-1) (+1)T(-1) (+1)C
            |       |       |       (-1)     (-1)      (-1)
            0       1       1  
            0               0  
            |               |       (+1)               (+1)
            T--2         1--T        T(-1)           (+1)T
            |               |       (-1)               (-1)
            1               2
            0       0       0
            |       |       |       (+1)     (+1)      (+1)
            C--1 1--T--2 1--C        C(-1) (+1)T(-1) (+1)C

        """
        
        if state:
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

        # initialize environment C,T dictionaries
        self.C = dict()
        self.T = dict()

        if init or init_method:
            if not init_method: init_method= ctm_args.ctm_env_init_type 
            if state and init_method in ["CTMRG"]:
                init_env(state, self, init_method)
            else:
                raise RuntimeError("Cannot perform initialization for desired"\
                    +" ctm_env_init_type "+init_method+"."\
                    +" Missing state.")

    def __str__(self):
        s=f"ENV_abelian chi={self.chi}\n"
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

    def extend(self, new_chi, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        raise NotImplementedError
        # new_env= ENV(new_chi, ctm_args=ctm_args, global_args=global_args)
        # x= min(self.chi, new_chi)
        # for k,old_C in self.C.items(): new_env.C[k]= old_C[:x,:x].clone().detach()
        # for k,old_T in self.T.items():
        #     if k[1]==(0,-1):
        #         new_env.T[k]= old_T[:x,:,:x].clone().detach()
        #     elif k[1]==(-1,0):
        #         new_env.T[k]= old_T[:x,:x,:].clone().detach()
        #     elif k[1]==(0,1):
        #         new_env.T[k]= old_T[:,:x,:x].clone().detach()
        #     elif k[1]==(1,0):
        #         new_env.T[k]= old_T[:x,:,:x].clone().detach()
        #     else:
        #         raise Exception(f"Unexpected direction {k[1]}")

        # return new_env

    def to_dense(self, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        C_dense= {cid: c.to_dense() for cid,c in self.C.items()}
        T_dense= {tid: t.to_dense() for tid,t in self.T.items()}
        env_dense= ENV_ABELIAN(self.chi, settings=next(iter(C_dense.values())).conf, \
            ctm_args=ctm_args, global_args=global_args)
        return env_dense

def init_env(state, env, init_method=None, ctm_args=cfg.ctm_args):
    """
    :param state: wavefunction
    :param env: CTM environment
    :param init_method: desired initialization method
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS_ABELIAN
    :type env: ENV 
    :type init_method: str
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    by :class:`CTMARGS.ctm_env_init_type <config.CTMARGS>` 
    
 
    * CONST - all C and T tensors have all their elements intialized to a value 1
    * RANDOM - all C and T tensors have elements with random numbers drawn from uniform
      distribution [0,1)
    * CTMRG - tensors C and T are built from the on-site tensors of `state` 
    """
    if not init_method: init_method= ctm_args.ctm_env_init_type
    if init_method=='CONST':
        init_const(env, ctm_args.verbosity_initialization)
    elif init_method=='RANDOM':
        init_random(env, ctm_args.verbosity_initialization)
    elif init_method=='CTMRG':
        init_from_ipeps_pbc_nofuse(state, env, ctm_args.verbosity_initialization)
    elif init_method=='CTMRG_OBC':
        init_from_ipeps_obc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+init_method)

def init_const(env, verbosity=0):
    raise NotImplementedError
    # for key,t in env.C.items():
    #     env.C[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)
    # for key,t in env.T.items():
    #     env.T[key] = torch.ones(t.size(), dtype=env.dtype, device=env.device)

# TODO restrict random corners to have pos-semidef spectrum
def init_random(env, verbosity=0):
    raise NotImplementedError
    # for key,t in env.C.items():
    #     env.C[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)
    # for key,t in env.T.items():
    #     env.T[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)

# REQUIRES
# view, reshape, einsum/tensor_dot
# TODO compress/extend along environment dimension
def init_from_ipeps_pbc_nofuse(state, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps_pbc")
    
    # corners
    for coord,site in state.sites.items():

        # Left-upper corner
        #
        #     i      = C--1(-1)  
        # j--A--4      0(-1)
        #   /\
        #  3  m
        #      \ i
        #    j--A--4
        #      /
        #     3
        vec = (-1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a= contiguous(einsum('mijef,mijab->eafb',A,conj(A)))
        a= A.dot(A, ((0,1,2), (0,1,2)), conj=(0,1)) # mijef,mijab->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        ## here we need to group-legs / reshape
        a, leg_order= a.group_legs((2,3), new_s=-1) # ea(fb->F)->eaF
        a, leg_order= a.group_legs((0,1), new_s=-1) # (ea->E)F->EF
        a= a/a.max_abs()
        env.C[(coord,vec)]= a

        # right-upper corner
        #
        #     i      = (+1)0--C     
        # 2--A--j         (-1)1
        #   /\
        #  3  m
        #      \ i
        #    2--A--j
        #      /
        #     3
        vec = (1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a= contiguous(einsum('miefj,miabj->eafb',A,conj(A)))
        a= A.dot(A, ((0,1,4), (0,1,4)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a, leg_order= a.group_legs((2,3), new_s=-1) # ea(fb->F)->eaF
        a, leg_order= a.group_legs((0,1), new_s=1) # F(ea->E)->EF
        a= a/a.max_abs()
        env.C[(coord,vec)]=a

        # right-lower corner
        #
        #     1      =    (+1)0     
        # 2--A--j      (+1)1--C
        #   /\
        #  i  m
        #      \ 1
        #    2--A--j
        #      /
        #     i
        vec = (1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a= contiguous(einsum('mefij,mabij->eafb',A,conj(A)))
        a= A.dot(A, ((0,3,4), (0,3,4)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a, leg_order= a.group_legs((2,3), new_s=1) # ea(fb->F)->eaF
        a, leg_order= a.group_legs((0,1), new_s=1) # F(ea->E)->EF
        a= a/a.max_abs()
        env.C[(coord,vec)]=a

        # left-lower corner
        #
        #     1      = 0(+1)     
        # i--A--4      C--1(-1)
        #   /\
        #  j  m
        #      \ 1
        #    i--A--4
        #      /
        #     j
        vec = (-1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a = contiguous(einsum('meijf,maijb->eafb',A,conj(A)))
        a= A.dot(A, ((0,2,3), (0,2,3)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a, leg_order= a.group_legs((2,3), new_s=-1) # ea(fb->F)->eaF
        a, leg_order= a.group_legs((0,1), new_s=1) # F(ea->E)->EF
        a= a/a.max_abs()
        env.C[(coord,vec)]=a

    # half-row/-column transfer tensor
    for coord,site in state.sites.items():
        # upper transfer matrix
        #
        #     i      = (+1)0--T--2(-1)     
        # 2--A--4         (-1)1
        #   /\
        #  3  m
        #      \ i
        #    2--A--4
        #      /
        #     3
        vec = (0,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a = contiguous(einsum('miefg,miabc->eafbgc',A,conj(A)))
        a= A.dot(A, ((0,1), (0,1)), conj=(0,1)) # miefg,miabc->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a, leg_order= a.group_legs((4,5), new_s=-1) # eafb(gc->G)->eafbG
        # a, leg_order= a.group_legs((2,3), new_s=-1) # ea(fb->F)G->eaFG
        a, leg_order= a.group_legs((0,1), new_s=1) # (ea->E)FG->EFG
        a= a/a.max_abs()
        env.T[(coord,vec)]=a 

        # left transfer matrix
        #
        #     1      = 0(+1)     
        # i--A--4      T--2(-1)
        #   /\         1(-1)
        #  3  m
        #      \ 1
        #    i--A--4
        #      /
        #     3
        vec = (-1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a = contiguous(einsum('meifg,maibc->eafbgc',A,conj(A)))
        a= A.dot(A, ((0,2), (0,2)), conj=(0,1)) # meifg,maibc->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        # a, leg_order= a.group_legs((4,5), new_s=-1) # eafb(gc->G)->eafbG
        a, leg_order= a.group_legs((2,3), new_s=-1) # ea(fb->F)G->eaFG
        a, leg_order= a.group_legs((0,1), new_s=1) # (ea->E)FG->EFG
        a= a/a.max_abs()
        env.T[(coord,vec)]=a

        # lower transfer matrix
        #
        #     1      =    (+1)0     
        # 2--A--4      (+1)1--T--2(-1)
        #   /\
        #  i  m
        #      \ 1
        #    2--A--4
        #      /
        #     i
        vec = (0,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a = contiguous(einsum('mefig,mabic->eafbgc',A,conj(A)))
        a= A.dot(A, ((0,3), (0,3)), conj=(0,1)) # mefig,mabic->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a, leg_order= a.group_legs((4,5), new_s=-1) # eafb(gc->G)->eafbG
        a, leg_order= a.group_legs((2,3), new_s=1) # ea(fb->F)G->eaFG
        # a, leg_order= a.group_legs((0,1), new_s=1) # (ea->E)FG->EFG
        a= a/a.max_abs()
        env.T[(coord,vec)]=a

        # right transfer matrix
        #
        #     1      =    (+1)0     
        # 2--A--i      (+1)1--T
        #   /\            (-1)2
        #  3  m
        #      \ 1
        #    2--A--i
        #      /
        #     3
        vec = (1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        ## a = contiguous(einsum('mefgi,mabci->eafbgc',A,conj(A)))
        a= A.dot(A, ((0,4), (0,4)), conj=(0,1)) # mefig,mabic->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a, leg_order= a.group_legs((4,5), new_s=-1) # eafb(gc->G)->eafbG
        # a, leg_order= a.group_legs((2,3), new_s=1) # ea(fb->F)G->eaFG
        a, leg_order= a.group_legs((0,1), new_s=1) # (ea->E)FG->EFG
        a= a/a.max_abs()
        env.T[(coord,vec)]=a

def init_from_ipeps_obc(state, env, verbosity=0):
    raise NotImplementedError
    # if verbosity>0:
    #     print("ENV: init_from_ipeps")
    # for coord, site in state.sites.items():
    #     for rel_vec in [(-1,-1),(1,-1),(1,1),(-1,1)]:
    #         env.C[(coord,rel_vec)] = torch.zeros(env.chi,env.chi, dtype=env.dtype, device=env.device)

    #     # Left-upper corner
    #     #
    #     #     i      = C--1     
    #     # j--A--3      0
    #     #   /\
    #     #  2  m
    #     #      \ k
    #     #    l--A--3
    #     #      /
    #     #     2
    #     vec = (-1,-1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('mijef,mklab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.C[(coord,vec)][:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]=\
    #         a[:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]

    #     # right-upper corner
    #     #
    #     #     i      = 0--C     
    #     # 1--A--j         1
    #     #   /\
    #     #  2  m
    #     #      \ k
    #     #    1--A--l
    #     #      /
    #     #     2
    #     vec = (1,-1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('miefj,mkabl->eafb',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.C[(coord,vec)][:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[3]**2)]=\
    #         a[:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[3]**2)]

    #     # right-lower corner
    #     #
    #     #     0      =    0     
    #     # 1--A--j      1--C
    #     #   /\
    #     #  i  m
    #     #      \ 0
    #     #    1--A--l
    #     #      /
    #     #     k
    #     vec = (1,1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('mefij,mabkl->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.C[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[2]**2)]=\
    #         a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[2]**2)]

    #     # left-lower corner
    #     #
    #     #     0      = 0     
    #     # i--A--3      C--1
    #     #   /\
    #     #  j  m
    #     #      \ 0
    #     #    k--A--3
    #     #      /
    #     #     l
    #     vec = (-1,1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('meijf,maklb->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[4]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.C[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[4]**2)]=\
    #         a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[4]**2)]

    #     # upper transfer matrix
    #     #
    #     #     i      = 0--T--2     
    #     # 1--A--3         1
    #     #   /\
    #     #  2  m
    #     #      \ k
    #     #    1--A--3
    #     #      /
    #     #     2
    #     vec = (0,-1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('miefg,mkabc->eafbgc',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2, dimsA[4]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[3]**2,env.chi), dtype=env.dtype, device=env.device)
    #     env.T[(coord,vec)][:min(env.chi,dimsA[2]**2),:,:min(env.chi,dimsA[4]**2)]=\
    #         a[:min(env.chi,dimsA[2]**2),:,:min(env.chi,dimsA[4]**2)]

    #     # left transfer matrix
    #     #
    #     #     0      = 0     
    #     # i--A--3      T--2
    #     #   /\         1
    #     #  2  m
    #     #      \ 0
    #     #    k--A--3
    #     #      /
    #     #     2
    #     vec = (-1,0)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('meifg,makbc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[3]**2, dimsA[4]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
    #     env.T[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]=\
    #         a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]

    #     # lower transfer matrix
    #     #
    #     #     0      =    0     
    #     # 1--A--3      1--T--2
    #     #   /\
    #     #  i  m
    #     #      \ 0
    #     #    1--A--3
    #     #      /
    #     #     k
    #     vec = (0,1)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('mefig,mabkc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[4]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.T[(coord,vec)] = torch.zeros((dimsA[1]**2,env.chi,env.chi), dtype=env.dtype, device=env.device)
    #     env.T[(coord,vec)][:,:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[4]**2)]=\
    #         a[:,:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[4]**2)]

    #     # right transfer matrix
    #     #
    #     #     0      =    0     
    #     # 1--A--i      1--T
    #     #   /\            2
    #     #  2  m
    #     #      \ 0
    #     #    1--A--k
    #     #      /
    #     #     2
    #     vec = (1,0)
    #     A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
    #     dimsA = A.size()
    #     a = torch.einsum('mefgi,mabck->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2)
    #     a= a/torch.max(torch.abs(a))
    #     env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[2]**2,env.chi), dtype=env.dtype, device=env.device)
    #     env.T[(coord,vec)][:min(env.chi,dimsA[1]**2),:,:min(env.chi,dimsA[3]**2)]=\
    #         a[:min(env.chi,dimsA[1]**2),:,:min(env.chi,dimsA[3]**2)]