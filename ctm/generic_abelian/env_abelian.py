import warnings
import config as cfg
from math import sqrt
import yastn.yastn as yastn
try:
    import torch
    from ctm.generic.env import ENV
except ImportError as e:
    warnings.warn("torch not available", Warning)
import logging
log = logging.getLogger(__name__)

class ENV_ABELIAN():
    def __init__(self, chi=1, state=None, settings=None, init=False,\
        init_method=None, ctm_args=cfg.ctm_args, global_args=cfg.global_args):
        r"""
        :param chi: environment bond dimension :math:`\chi`
        :param state: wavefunction
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param init: initialize environment tensors
        :type init: bool
        :param init_method: choice of environment initialization. See :meth:`init_env`.
        :type init_method: str
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

            C--1 0--T--2 0--C        C(+1) (-1)T(+1) (-1)C
            |       |       |       (+1)     (+1)      (+1)
            0       1       1  
            0               0  
            |               |       (-1)               (-1)
            T--2         1--T        T(+1)           (-1)T
            |               |       (+1)               (+1)
            1               2
            0       0       0
            |       |       |       (-1)     (-1)      (-1)
            C--1 1--T--2 1--C        C(+1) (-1)T(+1) (-1)C

        .. note::

            The structure of fused double-layer legs, which are carried by T-tensors, is obtained
            by fusing on-site tensor (`ket`) with its conjugate (`bra`). The leg of `ket` always
            preceeds `bra` when fusing.   


        """
        if state:
            self.engine= state.engine
            self.dtype= state.dtype
            self.nsym = state.nsym
            self.sym= state.sym
        elif settings:
            self.engine= settings
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
        :param state: abelian-symmetric iPEPS
        :type state: IPEPS_ABELIAN
        :return: returns equivalent of the environment with all C,T tensors in their dense 
                 representation on PyTorch backend. 
        :rtype: ENV

        Create a copy of environment with all on-site tensors as dense possesing no explicit
        block structure (symmetry). The virtual spaces of on-site tensors in ``state`` 
        are added to corresponding spaces of environment's T tensors to guarantee consistency.

        .. note::
            This operations preserves gradients on the returned dense environment.
        """
        vts= state.vertexToSite
        dir_to_leg= {(0,-1): 1, (0,1): 0, (-1,0): 2, (1,0): 1}
        C_lss= { cid: dict() for cid in self.C.keys() }
        T_lss= dict()

        # 0) compute correct leg structure of T's. Unfuse the pair of 
        #    auxiliary legs connecting T's to on-site tensors. Merge the 
        #    environment virtual spaces on connected legs
        tmp_T= { tid: t.unfuse_legs(dir_to_leg[tid[1]]) for tid,t in self.T.items() }
        for tid in tmp_T.keys():
            t_xy,t_dir= tid
            if t_dir==(0,-1): #UP
                # all legs of current T
                # 0--T(x-1,y)--3 0--T(x,y)--3 0--T(x+1,y)--3
                #    1,2            1,2          1,2
                T_lss[tid]= { 1: tmp_T[tid].get_legs(axes=1), 2: tmp_T[tid].get_legs(axes=2), \
                    0: yastn.legs_union(tmp_T[(vts((t_xy[0]-1, t_xy[1])), t_dir)].get_legs(axes=3),\
                        self.C[((t_xy),(-1,-1))].get_legs(axes=1)).conj(),\
                    3: yastn.legs_union(tmp_T[(vts((t_xy[0]+1, t_xy[1])), t_dir)].get_legs(axes=0),\
                        self.C[((t_xy),(1,-1))].get_legs(axes=0)).conj() }
                # upper-left corner
                # C--1
                # 0
                C_lss[(tid[0],(-1,-1))][1]= T_lss[tid][0].conj()
                # upper-right corner
                # 0--C
                #    1
                C_lss[(tid[0],(1,-1))][0]= T_lss[tid][3].conj()
            elif t_dir==(0,1): #DOWN
                #    0,1              0,1        0,1
                # 2--T(x-1,y)--3 2--T(x,y)--3 2--T(x+1,y)--3
                T_lss[tid]= { 0: tmp_T[tid].get_legs(axes=0), 1: tmp_T[tid].get_legs(axes=1), \
                    2: yastn.legs_union(tmp_T[(vts((t_xy[0]-1, t_xy[1])), t_dir)].get_legs(axes=3),\
                        self.C[((t_xy),(-1,1))].get_legs(axes=1)).conj(),\
                    3: yastn.legs_union(tmp_T[(vts((t_xy[0]+1, t_xy[1])), t_dir)].get_legs(axes=2),\
                        self.C[((t_xy),(1,1))].get_legs(axes=1)).conj() }
                # lower-left corner
                # 0
                # C--1
                C_lss[(tid[0],(-1,1))][1]= T_lss[tid][2].conj()
                # lower-right corner
                #    0
                # 1--C
                C_lss[(tid[0],(1,1))][1]= T_lss[tid][3].conj()
            elif t_dir==(-1,0): #LEFT
                # 0
                # T--2,3
                # 1
                T_lss[tid]= { 2: tmp_T[tid].get_legs(axes=2), 3: tmp_T[tid].get_legs(axes=3),\
                    0: yastn.legs_union(tmp_T[(vts((t_xy[0], t_xy[1]-1)), t_dir)].get_legs(axes=1),\
                        self.C[((t_xy),(-1,-1))].get_legs(axes=0)).conj(),\
                    1: yastn.legs_union(tmp_T[(vts((t_xy[0], t_xy[1]+1)), t_dir)].get_legs(axes=0),\
                        self.C[((t_xy),(-1,1))].get_legs(axes=0)).conj() }
                # upper-left corner
                # C--1
                # 0
                C_lss[(tid[0],(-1,-1))][0]= T_lss[tid][0].conj()
                # lower-left corner
                # 0
                # C--1
                C_lss[(tid[0],(-1,1))][0]= T_lss[tid][1].conj()
            elif t_dir==(1,0): #RIGHT
                #      0
                # 1,2--T
                #      3
                T_lss[tid]= { 1: tmp_T[tid].get_legs(axes=1), 2: tmp_T[tid].get_legs(axes=2),\
                    0: yastn.legs_union(tmp_T[(vts((t_xy[0], t_xy[1]-1)), t_dir)].get_legs(axes=3),\
                        self.C[((t_xy),(1,-1))].get_legs(axes=1)).conj(),\
                    3: yastn.legs_union(tmp_T[(vts((t_xy[0], t_xy[1]+1)), t_dir)].get_legs(axes=0),\
                        self.C[((t_xy),(1,1))].get_legs(axes=0)).conj() }
                # upper-right corner
                # 0--C
                #    1
                C_lss[(tid[0],(1,-1))][1]= T_lss[tid][0].conj()
                # lower-right corner
                #    0
                # 1--C
                C_lss[(tid[0],(1,1))][0]= T_lss[tid][3].conj()
            else:
                raise RuntimeError("Invalid T-tensor id "+str(tid))

        # 1) convert to dense representation. Reshape T's into double-layer form
        C_torch= {cid: c.to_dense(legs=C_lss[cid]) for cid,c in self.C.items()}
        T_torch= dict()
        for tid,t in tmp_T.items():
            t_xy,t_dir= tid
            t= t.to_dense(legs=T_lss[tid])
            if t_dir==(0,-1):
                T_torch[tid]= t.view(t.size(0), t.size(1)*t.size(2), t.size(3))
            elif t_dir==(0,1):
                T_torch[tid]= t.view(t.size(0)*t.size(1), t.size(2), t.size(3))
            elif t_dir==(-1,0):
                T_torch[tid]= t.view(t.size(0), t.size(1), t.size(2)*t.size(3))
            else:
                T_torch[tid]= t.view(t.size(0), t.size(1)*t.size(2), t.size(3))
        
        max_chi= max(self.chi, max([max(c.size()) for c in C_torch.values()]))
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
        :return: returns a clone of the environment
        :rtype: ENV_ABELIAN

        Create a clone of environment with all tensors (their blocks) attached 
        to the computational graph. 

        .. note::
            This operation preserves gradient tracking.
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

        In case of using PyTorch backend, get a detached "view" of the environment. See 
        `torch.Tensor.detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_.

        .. note::
            This operation does not preserve gradient tracking. 
        """
        e= ENV_ABELIAN(self.chi, settings=self.engine)
        e.C= {cid: c.detach() for cid,c in self.C.items()}
        e.T= {tid: t.detach() for tid,t in self.T.items()}
        return e

    def detach_(self):
        for c in self.C.values(): c.detach()
        for t in self.T.values(): t.detach()

    def get_spectra(self):
        spec= {}
        for c_key, c_t in self.C.items():
            _,S,_ = c_t.svd()
            spec[c_key]= S
        return spec


def init_env(state, env, init_method=None, ctm_args=cfg.ctm_args):
    """
    :param state: wavefunction
    :param env: CTM environment
    :param init_method: desired initialization method
    :param ctm_args: CTM algorithm configuration
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN 
    :type init_method: str
    :type ctm_args: CTMARGS

    Initializes the environment `env` according to one of the supported options specified 
    by :class:`CTMARGS.ctm_env_init_type <config.CTMARGS>` 
    
    * CTMRG - tensors C and T are built from the on-site tensors of `state` 
    """
    if not init_method: init_method= ctm_args.ctm_env_init_type
    if init_method=='CTMRG':
        init_from_ipeps_pbc(state, env, ctm_args.verbosity_initialization)
    else:
        raise ValueError("Invalid environment initialization: "+init_method)

def init_from_ipeps_pbc(state, env, verbosity=0):
    if verbosity>0:
        print("ENV: init_from_ipeps_pbc")
        
    # corners
    for coord,site in state.sites.items():

        # Left-upper corner
        #
        #       i          = C--(1,3)->1(+1)
        #   j--a*--4->3      |
        #      /\           (0,2)->0(+1)
        #  2<-3  m
        #         \ i
        #       j--a--4->1
        #         /
        #        3->0
        vec = (-1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,1,2), (0,1,2)), conj=(0,1)) # mijef,mijab->efab; efab->eafb
        a= a.fuse_legs( axes=((0,2),(1,3)) )
        a= a/a.norm(p='inf')
        env.C[(coord,vec)]= a

        # right-upper corner
        #
        #        i      = (-1)0<-(0,2)--C     
        # 2<-2--a*--j                   |
        #      /\             (+1)1<-(1,3)
        #  3<-3  m
        #         \ i
        #    0<-2--a--j
        #         /
        #     1<-3
        vec = (1,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,1,4), (0,1,4)), conj=(0,1)) # miefj,miabj->efab; efab->eafb
        a= a.fuse_legs( axes=((0,2),(1,3)) )
        a= a/a.norm(p='inf')
        env.C[(coord,vec)]=a

        # right-lower corner
        #
        #        1->1      =     (-1)0<-(0,2)     
        # 3<-2--a*--j                      |
        #       /\           (-1)1<-(1,3)--C
        #      i  m
        #          \ 1->0
        #     1<-2--a--j
        #          /
        #         i
        vec = (1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,3,4), (0,3,4)), conj=(0,1)) # miefj,miabj->efab; efab->eafb
        a= a.fuse_legs( axes=((0,2),(1,3)) )
        a= a/a.norm(p='inf')
        env.C[(coord,vec)]=a

        # left-lower corner
        #
        #     1->2       = (0,2)->2(-1) 
        # i--a*--4->3       |
        #   /\              C--(1,3)->1(+1)
        #  j  m
        #      \ 1->0
        #    i--a--4->1
        #      /
        #     j
        vec = (-1,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,2,3), (0,2,3)), conj=(0,1)) # miefj,miabj->efab; efab->eafb
        a= a.fuse_legs( axes=((0,2),(1,3)) )
        a= a/a.norm(p='inf')
        env.C[(coord,vec)]=a

    # half-row/-column transfer tensor
    for coord,site in state.sites.items():
        # upper transfer matrix
        #
        #        i          = (-1)0<-(0,3)--T--(2,5)->2(+1)     
        # 3<-2--a*--4->5                    |
        #      /\                 (+1)1<-(1,4)
        #  4<-3  m
        #         \ i
        #    0<-2--a--4->2
        #         /
        #     1<-3
        vec = (0,-1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,1), (0,1)), conj=(0,1)) # miefg,miabc->efgabc ; efgabc->eafbgc
        a= a.fuse_legs( axes=((0,3),(1,4),(2,5)) )
        a= a/a.norm(p='inf')
        env.T[(coord,vec)]=a 

        # left transfer matrix
        #
        #       1->3       = (0,3)->0(-1)     
        #   i--a*--4->5      T--(2,5)->2(+1)
        #      /\            (1,4)->1(+1)
        #  4<-3  m
        #         \ 1->0
        #       i--a--4->2
        #         /
        #     1<-3
        vec = (-1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,2), (0,2)), conj=(0,1)) # meifg,maibc->efgabc ; efgabc->eafbgc
        a= a.fuse_legs( axes=((0,3),(1,4),(2,5)) )
        a= a/a.norm(p='inf')
        env.T[(coord,vec)]=a

        # lower transfer matrix
        #
        #        1->3      =     (-1)0<-(0,3)     
        # 4<-2--a*--4->5                   |
        #      /\            (-1)1<-(1,4)--T--(2,5)->2(+1)
        #     i  m
        #         \ 1->0
        #    1<-2--a--4->2
        #         /
        #        i
        vec = (0,1)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,3), (0,3)), conj=(0,1)) # mefig,mabic->efgabc; efgabc->eafbgc
        a= a.fuse_legs( axes=((0,3),(1,4),(2,5)) )
        a= a/a.norm(p='inf')
        env.T[(coord,vec)]=a

        # right transfer matrix
        #
        #        1->3      =     (-1)0<-(0,3)     
        # 4<-2--a*--i        (-1)1<-(1,4)--T
        #      /\                (+1)2<-(2,5)
        #  5<-3  m
        #         \ 1->0
        #    1<-2--a--i
        #         /
        #     2<-3
        vec = (1,0)
        A = state.site((coord[0]+vec[0],coord[1]+vec[1]))
        a= A.tensordot(A, ((0,4), (0,4)), conj=(0,1)) # mefig,mabic->efgabc; efgabc->eafbgc
        a= a.fuse_legs( axes=((0,3),(1,4),(2,5)) )
        a= a/a.norm(p='inf')
        env.T[(coord,vec)]=a

def ctmrg_conv_specC(state, env, history, p='inf', ctm_args=cfg.ctm_args):
    if not history:
        history={'spec': [], 'diffs': [], 'conv_crit': []}
    # use corner spectra
    conv_crit=float('inf')
    diff=float('inf')
    diffs=None
    spec= env.get_spectra()
    if state.engine.backend.BACKEND_ID=='np':
        spec_nosym_sorted= { s_key : np.sort(s_t._data)[::-1] \
            for s_key, s_t in spec.items() }            
    else:
        spec_nosym_sorted= { s_key : s_t._data.sort(descending=True)[0] \
            for s_key, s_t in spec.items() }
    if len(history['spec'])>0:
        s_old= history['spec'][-1]
        diffs= []
        for k in spec.keys():
            x_0,x_1 = spec_nosym_sorted[k], s_old[k]
            n_x0= x_0.shape[0] if state.engine.backend.BACKEND_ID=='np' else x_0.size(0)
            n_x1= x_1.shape[0] if state.engine.backend.BACKEND_ID=='np' else x_1.size(0)
            if n_x0>n_x1:
                diffs.append( (sum((x_1-x_0[:n_x1])**2) \
                    + sum(x_0[n_x1:]**2)).item() )
            else:
                diffs.append( (sum((x_0-x_1[:n_x0])**2) \
                    + sum(x_1[n_x0:]**2)).item() )
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
