import warnings
import config as cfg
from itertools import product
import yamps.yast as yast
try:
    import torch
    from ctm.generic.env import ENV
except ImportError as e:
    warnings.warn("torch not available", Warning)

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
            self.backend= settings.backend
            self.dtype= settings.default_dtype
            self.nsym = settings.sym.NSYM
            self.sym= settings.sym.SYM_ID
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

    def to_dense(self, state, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param state: state providing the relevant vertexToSite function
        :type state: IPEPS_ABELIAN
        :return: returns equivalent of the environment with all C,T tensors in their dense 
                 representation on torch backend. 
        :rtype: ENV

        Create a copy of environment with all on-site tensors as dense possesing no explicit
        block structure (symmetry). This operations preserves gradients on returned
        dense environment.
        """
        vts= state.vertexToSite
        dir_to_leg= {(0,-1): 1, (0,1): 0, (-1,0): 2, (1,0): 1}
        C_lss= { cid: dict() for cid in self.C.keys() }
        T_lss= dict()

        # 0) compute correct leg structure of T's. Unfuse the pair of 
        #    auxiliary legs connecting T's to on-site tensors. Merge the 
        #    environment virtual spaces on connected legs
        tmp_T= { tid: t.unfuse_legs_hard(dir_to_leg[tid[1]]) for tid,t in self.T.items() }
        for tid in tmp_T.keys():
            t_xy,t_dir= tid
            if t_dir==(0,-1): #UP
                # all legs of current T
                # 0--T(x-1,y)--3 0--T(x,y)--3 0--T(x+1,y)--3
                #    1,2            1,2          1,2
                T_lss[tid]= yast.leg_structures_for_dense( tensors=[tmp_T[tid],\
                    tmp_T[(vts((t_xy[0]-1, t_xy[1])), t_dir)], {3: 0},\
                    self.C[((t_xy),(-1,-1))], {1: 0},\
                    tmp_T[(vts((t_xy[0]+1, t_xy[1])), t_dir)], {0: 3},\
                    self.C[((t_xy),(1,-1))], {0: 3}] )
                # upper-left corner
                # C--1
                # 0
                C_lss[(tid[0],(-1,-1))][1]= T_lss[tid][0]
                # upper-right corner
                # 0--C
                #    1
                C_lss[(tid[0],(1,-1))][0]= T_lss[tid][3]
            elif t_dir==(0,1): #DOWN
                #    0,1              0,1        0,1
                # 2--T(x-1,y)--3 2--T(x,y)--3 2--T(x+1,y)--3
                T_lss[tid]= yast.leg_structures_for_dense( tensors=[tmp_T[tid],\
                    tmp_T[(vts((t_xy[0]-1, t_xy[1])), t_dir)], {3: 2},\
                    self.C[((t_xy),(-1,1))], {1: 2},\
                    tmp_T[(vts((t_xy[0]+1, t_xy[1])), t_dir)], {2: 3},\
                    self.C[((t_xy),(1,1))], {1: 3}])
                # lower-left corner
                # 0
                # C--1
                C_lss[(tid[0],(-1,1))][1]= T_lss[tid][2]
                # lower-right corner
                #    0
                # 1--C
                C_lss[(tid[0],(1,1))][1]= T_lss[tid][3]
            elif t_dir==(-1,0): #LEFT
                # 0
                # T--2,3
                # 1
                T_lss[tid]= yast.leg_structures_for_dense( tensors=[tmp_T[tid],\
                    tmp_T[(vts((t_xy[0], t_xy[1]-1)), t_dir)], {1: 0},\
                    self.C[((t_xy),(-1,-1))], {0: 0},\
                    tmp_T[(vts((t_xy[0], t_xy[1]+1)), t_dir)], {0: 1},\
                    self.C[((t_xy),(-1,1))], {0: 1}])
                # upper-left corner
                # C--1
                # 0
                C_lss[(tid[0],(-1,-1))][0]= T_lss[tid][0]
                # lower-left corner
                # 0
                # C--1
                C_lss[(tid[0],(-1,1))][0]= T_lss[tid][1]
            elif t_dir==(1,0): #RIGHT
                #      0
                # 1,2--T
                #      3
                T_lss[tid]= yast.leg_structures_for_dense( tensors=[tmp_T[tid],\
                    tmp_T[(vts((t_xy[0], t_xy[1]-1)), t_dir)], {3: 0},\
                    self.C[((t_xy),(1,-1))], {1: 0},\
                    tmp_T[(vts((t_xy[0], t_xy[1]+1)), t_dir)], {0: 3},\
                    self.C[((t_xy),(1,1))], {0: 3}])
                # upper-right corner
                # 0--C
                #    1
                C_lss[(tid[0],(1,-1))][1]= T_lss[tid][0]
                # lower-right corner
                #    0
                # 1--C
                C_lss[(tid[0],(1,1))][0]= T_lss[tid][3]
            else:
                raise RuntimeError("Invalid T-tensor id "+str(tid))

        # 1) convert to dense representation. Reshape T's into double-layer form
        C_torch= {cid: c.to_dense(leg_structures=C_lss[cid]) for cid,c in self.C.items()}
        T_torch= dict()
        for tid,t in tmp_T.items():
            t_xy,t_dir= tid
            t= t.to_dense(leg_structures=T_lss[tid])
            if t_dir==(0,-1):
                T_torch[tid]= t.view(t.size(0), t.size(1)*t.size(2), t.size(3))
            elif t_dir==(0,1):
                T_torch[tid]= t.view(t.size(0)*t.size(1), t.size(2), t.size(3))
            elif t_dir==(-1,0):
                T_torch[tid]= t.view(t.size(0), t.size(1), t.size(2)*t.size(3))
            else:
                T_torch[tid]= t.view(t.size(0), t.size(1)*t.size(2), t.size(3))
        
        max_chi= max(self.chi, max(max([c.size() for c in C_torch.values()])))
        if max_chi>self.chi:
            warnings.warn("Increasing chi. Equivalent chi ("+str(max_chi)+") of symmetric"\
                +" environment is higher than original chi ("+str(self.chi)+").", Warning)

        # 2) Fill the dense environment with dimension chi by dense representations of 
        #    symmetric environment tensors
        env_torch= ENV(max_chi, ctm_args=ctm_args, global_args=global_args)
        for cid,c in C_torch.items():
            env_torch.C[cid]= torch.zeros(max_chi,max_chi,dtype=c.dtype,device=c.device)
            env_torch.C[cid][:c.size(0),:c.size(1)]= c
        for tid,t in T_torch.items():
            t_site, t_dir= tid
            if t_dir==(0,-1):
                env_torch.T[tid]= torch.zeros(max_chi,t.size(1),max_chi,dtype=t.dtype,device=t.device)
                env_torch.T[tid][:t.size(0),:,:t.size(2)]= t
            elif t_dir==(-1,0):
                env_torch.T[tid]= torch.zeros(max_chi,max_chi,t.size(2),dtype=t.dtype,device=t.device)
                env_torch.T[tid][:t.size(0),:t.size(1),:]= t
            elif t_dir==(0,1):
                env_torch.T[tid]= torch.zeros(t.size(0),max_chi,max_chi,dtype=t.dtype,device=t.device)
                env_torch.T[tid][:,:t.size(1),:t.size(2)]= t
            elif t_dir==(1,0):
                env_torch.T[tid]= torch.zeros(max_chi,t.size(1),max_chi,dtype=t.dtype,device=t.device)
                env_torch.T[tid][:t.size(0),:,:t.size(2)]= t

        return env_torch

    def clone(self):
        r"""
        :return: returns a clone of the environment with all C,T tensors attached to
                 computational graph.
        :rtype: ENV_ABELIAN

        Create a clone environment with all tensors (their blocks) attached to
        computational graph. 
        """
        e= ENV_ABELIAN(self.chi, settings=self.engine)
        e.C= {cid: c.clone() for cid,c in self.C.items()}
        e.T= {tid: t.clone() for tid,t in self.T.items()}
        return e

    def detach(self):
        r"""
        :return: returns a view of the environment with all C,T tensors detached from
                 computational graph.
        :rtype: ENV_ABELIAN

        Create a view of environment with all on-site tensors (their blocks) detached 
        from computational graph. 
        """
        e= ENV_ABELIAN(self.chi, settings=self.engine)
        e.C= {cid: c.detach() for cid,c in self.C.items()}
        e.T= {tid: t.detach() for tid,t in self.T.items()}
        return e

    def detach_(self):
        for c in self.C.values(): c.detach(inplace=True)
        for t in self.T.values(): t.detach(inplace=True)

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
        init_from_ipeps_pbc(state, env, ctm_args.verbosity_initialization)
    elif init_method=='CTMRG_OBC':
        init_from_ipeps_obc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+init_method)

def init_const(env, verbosity=0):
    raise NotImplementedError

def init_random(env, verbosity=0):
    # The symmetry sectors (= allowed blocks) are given by the on-site tensors of 
    # the iPEPS ansatz
    raise NotImplementedError

def init_from_ipeps_pbc(state, env, verbosity=0):
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
        a= A.tensordot(A, ((0,1,2), (0,1,2)), conj=(0,1)) # mijef,mijab->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a= a.fuse_legs_hard( axes=((0,1),(2,3)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,1,4), (0,1,4)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a= a.fuse_legs_hard( axes=((0,1),(2,3)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,3,4), (0,3,4)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a= a.fuse_legs_hard( axes=((0,1),(2,3)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,2,3), (0,2,3)), conj=(0,1)) # miefj,miabj->efab
        a= a.transpose((0,2,1,3)) # efab->eafb
        a= a.fuse_legs_hard( axes=((0,1),(2,3)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,1), (0,1)), conj=(0,1)) # miefg,miabc->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a= a.fuse_legs_hard( axes=((0,1),(2,3),(4,5)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,2), (0,2)), conj=(0,1)) # meifg,maibc->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a= a.fuse_legs_hard( axes=((0,1),(2,3),(4,5)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,3), (0,3)), conj=(0,1)) # mefig,mabic->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a= a.fuse_legs_hard( axes=((0,1),(2,3),(4,5)) )
        a= a/a.norm(p='inf')
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
        a= A.tensordot(A, ((0,4), (0,4)), conj=(0,1)) # mefig,mabic->efgabc
        a= a.transpose((0,3,1,4,2,5)) # efgabc->eafbgc
        a= a.fuse_legs_hard( axes=((0,1),(2,3),(4,5)) )
        a= a/a.norm(p='inf')
        env.T[(coord,vec)]=a

def init_from_ipeps_obc(state, env, verbosity=0):
    raise NotImplementedError
