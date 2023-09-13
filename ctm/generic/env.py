from math import sqrt
import torch
import config as cfg
from tn_interface import einsum
from tn_interface import conj
from tn_interface import contiguous, view
import logging
log = logging.getLogger(__name__)

class ENV():
    def __init__(self, chi, state=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param state: wavefunction
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type chi: int
        :type state: IPEPS
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

        Environment tensors of some ENV object ``e`` are accesed through its members ``C`` and ``T`` 
        by providing a tuple of coordinates and directional vector to the environment tensor:: 
            
            coord=(0,0)                # tuple(x,y) identifying vertex on the square lattice
            rel_dir_vec_C=(-1,-1)      # tuple(rx,ry) identifying one of the four corner tensors
            rel_dir_vec_T=(-1,0)       # tuple(rx,ry) identifying one of the four half-row/column tensors
            C_upper_left= e.C[(coord,rel_dir_vec_C)] # return upper left corner tensor of site at coord
            T_left= e.T[(coord,rel_dir_vec_T)]       # return left half-row tensor of site at coord
        
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

        .. note::

            The structure of fused double-layer legs, which are carried by T-tensors, is obtained
            by fusing on-site tensor (`ket`) with its conjugate (`bra`). The leg of `ket` always
            preceeds `bra` when fusing.
        """
        if state:
            self.dtype= state.dtype
            self.device= state.device
        else:
            self.dtype= global_args.torch_dtype
            self.device= global_args.device
        self.chi = chi

        # initialize environment tensors
        self.C = dict()
        self.T = dict()


        if state is not None:
            numl= 2 if len(next(iter(state.sites.values())).size())>4 else 1
            for coord, site in state.sites.items():
                #for vec in [(0,-1), (-1,0), (0,1), (1,0)]:
                #    self.T[(coord,vec)]="T"+str(ipeps.site(coord))
                self.T[(coord,(0,-1))]=torch.empty((self.chi,site.size(-4)**numl,self.chi), 
                    dtype=self.dtype, device=self.device)
                self.T[(coord,(-1,0))]=torch.empty((self.chi,self.chi,site.size(-3)**numl),
                    dtype=self.dtype, device=self.device)
                self.T[(coord,(0,1))]=torch.empty((site.size(-2)**numl,self.chi,self.chi), 
                    dtype=self.dtype, device=self.device)
                self.T[(coord,(1,0))]=torch.empty((self.chi,site.size(-1)**numl,self.chi), 
                    dtype=self.dtype, device=self.device)

                #for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                #     self.C[(coord,vec)]="C"+str(ipeps.site(coord))
                for vec in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    self.C[(coord,vec)]=torch.empty((self.chi,self.chi), dtype=self.dtype, device=self.device)

    def __str__(self):
        s=f"ENV chi={self.chi}\n"
        for cr,t in self.C.items():
            s+=f"C({cr[0]} {cr[1]}): {t.size()}\n"
        for cr,t in self.T.items():
            s+=f"T({cr[0]} {cr[1]}): {t.size()}\n"
        return s

    def clone(self, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS

        Create a clone of the environment.

        .. note::
            This operation preserves gradient tracking.
        """
        new_env= ENV(self.chi, ctm_args=ctm_args, global_args=global_args)
        new_env.C= { k: c.clone() for k,c in self.C.items() }
        new_env.T= { k: t.clone() for k,t in self.T.items() }
        return new_env

    def detach(self, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS
        
        Get a detached "view" of the environment. See 
        `torch.Tensor.detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_.

        .. note::
            This operation does not preserve gradient tracking.
        """
        new_env= ENV(self.chi, ctm_args=ctm_args, global_args=global_args)
        new_env.C= { k: c.detach() for k,c in self.C.items() }
        new_env.T= { k: t.detach() for k,t in self.T.items() }
        return new_env

    def detach_(self):
        for c in self.C.values(): c.detach_()
        for t in self.T.values(): t.detach_()

    def extend(self, new_chi, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param new_chi: new environment bond dimension
        :type new_chi: int
        :param ctm_args: CTM algorithm configuration
        :param global_args: global configuration
        :type ctm_args: CTMARGS
        :type global_args: GLOBALARGS

        Create a new environment with all environment tensors enlarged up to 
        environment dimension ``new_chi``. The enlarged C, T tensors are padded with zeros.

        .. note::
            This operation preserves gradient tracking.
        """
        new_env= ENV(new_chi, ctm_args=ctm_args, global_args=global_args)
        opts= {'dtype': self.dtype, 'device': self.device}
        x= min(self.chi, new_chi)
        for k,old_C in self.C.items(): 
            new_env.C[k]= torch.zeros(new_chi,new_chi,**opts)
            new_env.C[k][:x,:x]= old_C[:x,:x].clone().detach()
        for k,old_T in self.T.items():
            if k[1]==(0,-1):
                new_env.T[k]= torch.zeros((new_chi,old_T.size(1),new_chi),**opts)
                new_env.T[k][:x,:,:x]= old_T[:x,:,:x].clone().detach()
            elif k[1]==(-1,0):
                new_env.T[k]= torch.zeros((new_chi,new_chi,old_T.size(2)),**opts)
                new_env.T[k][:x,:x,:]= old_T[:x,:x,:].clone().detach()
            elif k[1]==(0,1):
                new_env.T[k]= torch.zeros((old_T.size(0),new_chi,new_chi),**opts)
                new_env.T[k][:,:x,:x]= old_T[:,:x,:x].clone().detach()
            elif k[1]==(1,0):
                new_env.T[k]= torch.zeros((new_chi,old_T.size(1),new_chi),**opts)
                new_env.T[k][:x,:,:x]= old_T[:x,:,:x].clone().detach()
            else:
                raise Exception(f"Unexpected direction {k[1]}")

        return new_env

    def get_spectra(self):
        spec= {}
        for c_key, c_t in self.C.items():
            spec[c_key]= torch.linalg.svdvals(c_t)
            spec[c_key]= spec[c_key]/spec[c_key][0]
        return spec

    def get_site_env_t(self,coord,state):
        r"""
        :return: environment tensors of site at ``'coord'`` in order
                 C1, C2, C3, C4, T1, T2, T3, T4
        :rtype: tuple(torch.Tensor)

        ::

            C1(-1,-1)   T1       (1,-1)C2 
                        |(0,-1)
            T4--(-1,0)--A(0,0)--(1,0)--T2 
                        |(0,1)
            C4(-1,1)    T3        (1,1)C3
        """
        C1= self.C[(state.vertexToSite(coord),(-1,-1))]
        C2= self.C[(state.vertexToSite(coord),(1,-1))]
        C3= self.C[(state.vertexToSite(coord),(1,1))]
        C4= self.C[(state.vertexToSite(coord),(-1,1))]
        T1= self.T[(state.vertexToSite(coord),(0,-1))]
        T2= self.T[(state.vertexToSite(coord),(1,0))]
        T3= self.T[(state.vertexToSite(coord),(0,1))]
        T4= self.T[(state.vertexToSite(coord),(-1,0))]
        return C1, C2, C3, C4, T1, T2, T3, T4

def init_env(state, env, ctm_args=cfg.ctm_args):
    """
    :param state: wavefunction
    :param env: CTM environment
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS
    :type env: ENV 
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    by :class:`CTMARGS.ctm_env_init_type <config.CTMARGS>` 
    
        * ``"CONST"`` - all C and T tensors have all their elements intialized to a value 1
        * ``"RANDOM"`` - all C and T tensors have elements with random numbers drawn from uniform
          distribution [0,1)
        * ``"CTMRG"`` - tensors C and T are built from the on-site tensors of `state` 
    """
    if len(next(iter(state.sites.values())).size())==4 and \
        not (ctm_args.ctm_env_init_type in ["PROD","CTMRG_OBC"]):
        raise RuntimeError("Incompatible ENV initialization")

    if ctm_args.ctm_env_init_type=='PROD':
        init_prod(state, env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='RANDOM':
        init_random(env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='CTMRG':
        init_from_ipeps_pbc(state, env, ctm_args.verbosity_initialization)
    elif ctm_args.ctm_env_init_type=='CTMRG_OBC':
        init_from_ipeps_obc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+str(ctm_args.ctm_env_init_type))

# TODO restrict random corners to have pos-semidef spectrum
def init_random(env, verbosity=0):
    for key,t in env.C.items():
        env.C[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)
    for key,t in env.T.items():
        env.T[key] = torch.rand(t.size(), dtype=env.dtype, device=env.device)

def init_prod(state, env, verbosity=0):
    for key,t in env.C.items():
        env.C[key]= torch.zeros(t.size(), dtype=env.dtype, device=env.device)
        env.C[key][0,0]= 1.0 + 0.j if env.C[key].is_complex() else 1.0

    for coord, site in state.sites.items():
        # upper transfer matrix
        #
        #     0      = 0--T--2     
        # 1--A--3         1
        #   /\
        #  f  m
        #      \ 0
        #    1--A--3
        #      /
        #     b
        vec = (0,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= contiguous(einsum('uldr->d',A))
        elif len(dimsA)==5:
            a = contiguous(einsum('miefg,miebg->fb',A,conj(A)))
            a = view(a, (a.size(0)**2))
        env.T[(coord,vec)]= torch.zeros((env.chi,a.size(0),env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,:,0]= a

        # left transfer matrix
        #
        #     0      = 0     
        # 1--A--g      T--2
        #   /\         1
        #  2  m
        #      \ 0
        #    1--A--c
        #      /
        #     2
        vec = (-1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= contiguous(einsum('uldr->r',A))
        elif len(dimsA)==5:
            a = contiguous(einsum('meifg,meifc->gc',A,conj(A)))
            a = view(a, (a.size(0)**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,a.size(0)), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,0,:]= a

        # lower transfer matrix
        #
        #     e      =    0     
        # 1--A--3      1--T--2
        #   /\
        #  2  m
        #      \ a
        #    1--A--3
        #      /
        #     2
        vec = (0,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= contiguous(einsum('uldr->u',A))
        elif len(dimsA)==5:
            a = contiguous(einsum('mefig,mafig->ea',A,conj(A)))
            a = view(a, (a.size(0)**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((a.size(0),env.chi,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:,0,0]= a

        # right transfer matrix
        #
        #     0      =    0     
        # f--A--3      1--T
        #   /\            2
        #  2  m
        #      \ 0
        #    b--A--3
        #      /
        #     2
        vec = (1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= contiguous(einsum('uldr->l',A))
        elif len(dimsA)==5:
            a = contiguous(einsum('mefgi,mebgi->fb',A,conj(A)))
            a = view(a, (a.size(0)**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((env.chi,a.size(0),env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,:,0]= a

def init_from_ipeps_pbc(state, env, verbosity=0):
    
    def _normalize_nograd(a, _ord='inf'):
        with torch.no_grad():
            scale= a.abs().max()
        return a/scale

    if verbosity>0:
        print("ENV: init_from_ipeps")
    for coord, site in state.sites.items():
        for rel_vec in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            env.C[(coord,rel_vec)] = torch.zeros(env.chi,env.chi, dtype=env.dtype, 
                device=env.device)

        # Left-upper corner
        #
        #     i           = C--1     
        # j--A*--3(b)       0
        #     /\
        # (a)2  m
        #        \ i
        #      j--A--3(f)
        #        /
        #       2(e)
        vec = (-1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a= contiguous(einsum('mijef,mijab->eafb',A,conj(A)))
        a= view(a, (dimsA[3]**2, dimsA[4]**2))
        a= _normalize_nograd(a)
        env.C[(coord,vec)][:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]=\
            a[:min(env.chi,dimsA[3]**2),:min(env.chi,dimsA[4]**2)]

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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a= contiguous(einsum('miefj,miabj->eafb',A,conj(A)))
        a= view(a, (dimsA[2]**2, dimsA[3]**2))
        a= _normalize_nograd(a)
        env.C[(coord,vec)][:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[3]**2)]=\
            a[:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[3]**2)]

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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a= contiguous(einsum('mefij,mabij->eafb',A,conj(A)))
        a= view(a, (dimsA[1]**2, dimsA[2]**2))
        a= _normalize_nograd(a)
        env.C[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[2]**2)]=\
            a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[2]**2)]

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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = contiguous(einsum('meijf,maijb->eafb',A,conj(A)))
        a = view(a, (dimsA[1]**2, dimsA[4]**2))
        a= _normalize_nograd(a)
        env.C[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[4]**2)]=\
            a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[4]**2)]

        # upper transfer matrix
        #
        #        i         = 0--T--2     
        # (e)1--A--3(g)         1
        #      /\
        #  (f)2  m
        #         \ i
        #    (a)1--A--3(c)
        #         /
        #     (b)2
        vec = (0,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = contiguous(einsum('miefg,miabc->eafbgc',A,conj(A)))
        a = view(a, (dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))
        a= _normalize_nograd(a)
        env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[3]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,dimsA[2]**2),:,:min(env.chi,dimsA[4]**2)]=\
            a[:min(env.chi,dimsA[2]**2),:,:min(env.chi,dimsA[4]**2)]

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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = contiguous(einsum('meifg,maibc->eafbgc',A,conj(A)))
        a = view(a, (dimsA[1]**2, dimsA[3]**2, dimsA[4]**2))
        a= _normalize_nograd(a)
        env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]=\
            a[:min(env.chi,dimsA[1]**2),:min(env.chi,dimsA[3]**2),:]


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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = contiguous(einsum('mefig,mabic->eafbgc',A,conj(A)))
        a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[4]**2))
        a= _normalize_nograd(a)
        env.T[(coord,vec)] = torch.zeros((dimsA[1]**2,env.chi,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:,:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[4]**2)]=\
            a[:,:min(env.chi,dimsA[2]**2),:min(env.chi,dimsA[4]**2)]

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
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        a = contiguous(einsum('mefgi,mabci->eafbgc',A,conj(A)))
        a = view(a, (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2))
        a= _normalize_nograd(a)
        env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[2]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,dimsA[1]**2),:,:min(env.chi,dimsA[3]**2)]=\
            a[:min(env.chi,dimsA[1]**2),:,:min(env.chi,dimsA[3]**2)]

def init_from_ipeps_obc(state, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps_obc")
    for coord, site in state.sites.items():
        for rel_vec in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            env.C[(coord,rel_vec)] = torch.zeros(env.chi,env.chi, dtype=env.dtype, device=env.device)

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
        vec = (-1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('ijef->ef',A)
        elif len(dimsA)==5:
            a= torch.einsum('mijef,mklab->eafb',(A,A)).contiguous().view(dimsA[3]**2, dimsA[4]**2)
        a= a/torch.max(torch.abs(a))
        env.C[(coord,vec)][:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]=\
            a[:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]

        # right-upper corner
        #
        #     i      = 0--C     
        # 1--A--j         1
        #   /\
        #  2  m
        #      \ k
        #    1--A--l
        #      /
        #     2
        vec = (1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('iefj->ef',A)
        elif len(dimsA)==5:
            a= torch.einsum('miefj,mkabl->eafb',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2)
        a= a/torch.max(torch.abs(a))
        env.C[(coord,vec)][:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]=\
            a[:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]

        # right-lower corner
        #
        #     0      =    0     
        # 1--A--j      1--C
        #   /\
        #  i  m
        #      \ 0
        #    1--A--l
        #      /
        #     k
        vec = (1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('efij->ef',A)
        elif len(dimsA)==5:
            a= torch.einsum('mefij,mabkl->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2)
        a= a/torch.max(torch.abs(a))
        env.C[(coord,vec)][:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]=\
            a[:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]

        # left-lower corner
        #
        #     0      = 0     
        # i--A--3      C--1
        #   /\
        #  j  m
        #      \ 0
        #    k--A--3
        #      /
        #     l
        vec = (-1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('eijf->ef',A)
        elif len(dimsA)==5:
            a= torch.einsum('meijf,maklb->eafb',(A,A)).contiguous().view(dimsA[1]**2, dimsA[4]**2)
        a= a/torch.max(torch.abs(a))
        env.C[(coord,vec)][:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]=\
            a[:min(env.chi,a.size(0)),:min(env.chi,a.size(1))]

        # upper transfer matrix
        #
        #     i      = 0--T--2     
        # 1--A--3         1
        #   /\
        #  2  m
        #      \ k
        #    1--A--3
        #      /
        #     2
        vec = (0,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('iefg->efg',A)
        elif len(dimsA)==5:
            a= torch.einsum('miefg,mkabc->eafbgc',(A,A)).contiguous().view(dimsA[2]**2, dimsA[3]**2, dimsA[4]**2)
        a= a/torch.max(torch.abs(a))
        env.T[(coord,vec)] = torch.zeros((env.chi,a.size(1),env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,a.size(0)),:,:min(env.chi,a.size(2))]=\
            a[:min(env.chi,a.size(0)),:,:min(env.chi,a.size(2))]

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
        vec = (-1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('eifg->efg',A)
        elif len(dimsA)==5:
            a= torch.einsum('meifg,makbc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[3]**2, dimsA[4]**2)
        a= a/torch.max(torch.abs(a))
        env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,a.size(2)), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,a.size(0)),:min(env.chi,a.size(1)),:]=\
            a[:min(env.chi,a.size(0)),:min(env.chi,a.size(1)),:]

        # lower transfer matrix
        #
        #     0      =    0     
        # 1--A--3      1--T--2
        #   /\
        #  i  m
        #      \ 0
        #    1--A--3
        #      /
        #     k
        vec = (0,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('efig->efg',A)
        elif len(dimsA)==5:
            a= torch.einsum('mefig,mabkc->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[4]**2)
        a= a/torch.max(torch.abs(a))
        env.T[(coord,vec)] = torch.zeros((a.size(0),env.chi,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:,:min(env.chi,a.size(1)),:min(env.chi,a.size(2))]=\
            a[:,:min(env.chi,a.size(1)),:min(env.chi,a.size(2))]

        # right transfer matrix
        #
        #     0      =    0     
        # 1--A--i      1--T
        #   /\            2
        #  2  m
        #      \ 0
        #    1--A--k
        #      /
        #     2
        vec = (1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A.size()
        if len(dimsA)==4:
            a= torch.einsum('efgi->efg',A)
        elif len(dimsA)==5:
            a= torch.einsum('mefgi,mabck->eafbgc',(A,A)).contiguous().view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2)
        a= a/torch.max(torch.abs(a))
        env.T[(coord,vec)] = torch.zeros((env.chi,a.size(1),env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:min(env.chi,a.size(0)),:,:min(env.chi,a.size(2))]=\
            a[:min(env.chi,a.size(0)),:,:min(env.chi,a.size(2))]

def init_prod_overlap(state1, state2, env, verbosity=0):
    for key,t in env.C.items():
        env.C[key]= torch.zeros(t.size(), dtype=env.dtype, device=env.device)
        env.C[key][0,0]= 1.0 + 0.j if env.C[key].is_complex() else 1.0

    for coord, site in state1.sites.items():
        # upper transfer matrix
        #
        #     i      = 0--T--2
        # 1--A1--3        1
        #   /\
        #  2  m
        #      \ i
        #    1--A2--3
        #      /
        #     2
        vec = (0,-1)
        A1 = state1.site((coord[0]+vec[0],coord[1]+vec[1]))
        A2 = state2.site((coord[0]+vec[0],coord[1]+vec[1]))
        dimsA = A1.size()
        a = contiguous(einsum('miefg,miebg->fb',A1,conj(A2)))
        a = view(a, (dimsA[3]**2))
        a= a/a.abs().max()
        env.T[(coord,vec)]= torch.zeros((env.chi,dimsA[3]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,:,0]= a

        # left transfer matrix
        #
        #     0      = 0
        # i--A1--3     T--2
        #   /\         1
        #  2  m
        #      \ 0
        #    i--A2--3
        #      /
        #     2
        vec = (-1,0)
        A1 = state1.site((coord[0] + vec[0], coord[1] + vec[1]))
        A2 = state2.site((coord[0] + vec[0], coord[1] + vec[1]))
        dimsA = A1.size()
        a = contiguous(einsum('meifg,meifc->gc',A1,conj(A2)))
        a = view(a, (dimsA[4]**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((env.chi,env.chi,dimsA[4]**2), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,0,:]= a

        # lower transfer matrix
        #
        #     0      =    0
        # 1--A1--3     1--T--2
        #   /\
        #  i  m
        #      \ 0
        #    1--A2--3
        #      /
        #     i
        vec = (0,1)
        A1 = state1.site((coord[0] + vec[0], coord[1] + vec[1]))
        A2 = state2.site((coord[0] + vec[0], coord[1] + vec[1]))
        dimsA = A1.size()
        a = contiguous(einsum('mefig,mafig->ea',A1,conj(A2)))
        a = view(a, (dimsA[1]**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((dimsA[1]**2,env.chi,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][:,0,0]= a

        # right transfer matrix
        #
        #     0      =    0
        # 1--A1--i     1--T
        #   /\            2
        #  2  m
        #      \ 0
        #    1--A2--i
        #      /
        #     2
        vec = (1,0)
        A1 = state1.site((coord[0] + vec[0], coord[1] + vec[1]))
        A2 = state2.site((coord[0] + vec[0], coord[1] + vec[1]))
        dimsA = A1.size()
        a = contiguous(einsum('mefgi,mebgi->fb',A1,conj(A2)))
        a = view(a, (dimsA[2]**2))
        a= a/a.abs().max()
        env.T[(coord,vec)] = torch.zeros((env.chi,dimsA[2]**2,env.chi), dtype=env.dtype, device=env.device)
        env.T[(coord,vec)][0,:,0]= a

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

@torch.no_grad()
def ctmrg_conv_specC(state, env, history, p='inf', ctm_args=cfg.ctm_args):
    r"""
    :param state: wavefunction
    :param ènv: environment
    :type env: ENV
    :param history: dictionary with convergence data
    :type: dict(str,list)
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS
    :type ctm_args: CTMARGS
    :return: a tuple (``True``, ``history``) if CTMRG converged, otherwise a tuple (``False``, history) 
    :rtype: bool, dict(str,list)

    Generic convergence criterion for CTMRG based on the spectra 
    of the corner tensors

    .. math::

        \textrm{conv_crit}= \sqrt{\sum_{(r,d)} \left[\lambda^{(i)}_{(r,d)} - \lambda^{(i-1)}_{(r,d)}\right]^2}

    where *r* runs over all non-equivalent sites and *d* over all non-equivalent corners of *r*-th site.
    The superscript *i* denotes CTMRG iterations. Once the difference reaches required
    tolerance :attr:`CTMARGS.ctm_conv_tol` or maximal number of steps `CTMARGS.ctm_max_iter`,
    it returns ``True``.
    """
    if not history:
        history={'spec': [], 'diffs': [], 'conv_crit': []}
    # use corner spectra
    conv_crit=float('inf')
    diffs=None
    spec= env.get_spectra()
    spec_nosym_sorted= { s_key : s_t.sort(descending=True)[0] \
            for s_key, s_t in spec.items() }
    if len(history['spec'])>0:
        s_old= history['spec'][-1]
        diffs= [ sum((spec_nosym_sorted[k]-s_old[k])**2).item() \
            for k in spec.keys() ]
        # sqrt of sum of squares of all differences of all corner spectra - usual 2-norm
        if p in ['fro',2]: 
            conv_crit= sqrt(sum(diffs))
        # or take max of the differences
        elif p in [float('inf'),'inf']:
            conv_crit= sqrt(max(diffs))
    history['spec'].append(spec_nosym_sorted)
    history['diffs'].append(diffs)
    history['conv_crit'].append(conv_crit)
    
    if (len(history['diffs']) > 1 and conv_crit < ctm_args.ctm_conv_tol)\
        or len(history['diffs']) >= ctm_args.ctm_max_iter:
        log.info({"history_length": len(history['diffs']), "history": history['diffs']})
        return True, history
    return False, history
