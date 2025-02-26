import torch
from collections import OrderedDict
import json
import math
import config as cfg
from ipeps.ipeps_kagome import IPEPS_KAGOME
from ipeps.tensor_io import *

class IPESS_KAGOME_GENERIC(IPEPS_KAGOME):
    def __init__(self, ipess_tensors,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param ipess_tensors: dictionary of five tensors, which make up Kagome iPESS
                              ansatz 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type ipess_tensors: dict(str, torch.tensor)
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        iPESS ansatz for Kagome lattice composes five tensors, specified
        by dictionary::

            ipess_tensors = {'T_u': torch.Tensor, 'T_d': ..., 'B_a': ..., 'B_b': ..., 'B_c': ...}

        into a single rank-5 on-site tensor of parent IPEPS. These iPESS tensors can
        be accessed through member ``ipess_tensors``. 

        The ``'B_*'`` are rank-3 tensors, with index structure [p,i,j] where the first index `p` 
        is for physical degree of freedom, while indices `i` and `j` are auxiliary with bond dimension D.
        These bond tensors reside on corners shared between different triangles of Kagome lattice. 
        Bond tensors are connected by rank-3 trivalent tensors ``'T_u'``, ``'T_d'`` on up and down triangles 
        respectively. Trivalent tensors have only auxiliary indices of matching bond dimension D.
    
        The on-site tensors of corresponding iPEPS is obtained by the following contraction::
                                                     
                 2(d)            2(c)                    a
                  \             /          rot. pi       |
             0(w)==B_a         B_b==0(v)   clockwise  b--\                     
                    \         /             =>            \
                    1(l)     1(k)                         s0--s2--d
                     2(l)   1(k)                           | / 
                       \   /                               |/   <- down triangle
                        T_d                               s1
                         |                                 |
                         0(j)                              c
                         1(j)                               
                         |                 
                         B_c==0(u)        
                         |
                         2(i)
                         0(i)  
                         |
                        T_u
                       /   \ 
                     1(a)   2(b) 

        By construction, the degrees of freedom on down triangle are all combined into 
        a single on-site tensor of iPEPS. Instead, DoFs on the upper triangle have 
        to be accessed by construction of 2x2 patch (which is then embedded into environment)::        
        
            C    T             T          C
                 a             a
                 |             |
            T b--\          b--\
                  \        /    \
                  s0--s2--d     s0--s2--d T
                   | /           | /
                   |/            |/
                  s1            s1
                   |             |
                   c             c  
                  /             /
                 a             a
                 |             |
            T b--\          b--\
                  \        /    \
                  s0--s2--d     s0--s2--d T
                   | /           | /
                   |/            |/
                  s1            s1
                   |             |
                   c             c
            C      T             T        C
        """
        #TODO verification?
        self.ipess_tensors= ipess_tensors
        sites = self.build_onsite_tensors()

        super().__init__(sites, lX=1, lY=1, peps_args=peps_args,
                         global_args=global_args)

    def get_parameters(self):
        r"""
        :return: variational parameters of IPESS_KAGOME_GENERIC
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        In this case member ``ipess_tensors``.
        """
        return self.ipess_tensors.values()

    def get_checkpoint(self):
        r"""
        :return: all data necessary to reconstruct the state. In this case member ``ipess_tensors`` 
        :rtype: dict[str: torch.tensor]
        
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return self.ipess_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.ipess_tensors= checkpoint["parameters"]
        for t in self.ipess_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()

    def build_onsite_tensors(self):
        r"""
        :return: elementary unit cell of underlying IPEPS
        :rtype: dict[tuple(int,int): torch.Tensor]

        Build rank-5 on-site tensor by contracting the iPESS tensors.
        """
        A= torch.einsum('iab,uji,jkl,vkc,wld->uvwabcd', self.ipess_tensors['T_u'],
            self.ipess_tensors['B_c'], self.ipess_tensors['T_d'], self.ipess_tensors['B_b'], \
            self.ipess_tensors['B_a'])
        total_phys_dim= self.ipess_tensors['B_a'].size(0)*self.ipess_tensors['B_b'].size(0)\
            *self.ipess_tensors['B_c'].size(0)
        A= A.reshape([total_phys_dim]+[self.ipess_tensors['T_u'].size(1), \
            self.ipess_tensors['T_u'].size(2), self.ipess_tensors['B_b'].size(2), \
            self.ipess_tensors['B_a'].size(2)])
        A= A/A.abs().max()
        sites= {(0, 0): A}
        return sites

    def add_noise(self, noise):
        r"""
        :param noise: magnitude of noise
        :type noise: float

        Add uniform random noise to iPESS tensors.
        """
        for k in self.ipess_tensors:
            rand_t= torch.rand( self.ipess_tensors[k].size(), dtype=self.dtype, device=self.device)
            self.ipess_tensors[k]= self.ipess_tensors[k] + noise * (rand_t-1.0)
        self.sites = self.build_onsite_tensors()

    def get_physical_dim(self):
        assert self.ipess_tensors["B_a"].size(0)==self.ipess_tensors["B_b"].size(0) and \
            self.ipess_tensors["B_b"].size(0)==self.ipess_tensors["B_c"].size(0),\
            "Different physical dimensions across iPESS bond tensors"
        return self.ipess_tensors["B_a"].size(0)

    def get_aux_bond_dims(self):
        aux_bond_dims= set()
        aux_bond_dims= aux_bond_dims | set(self.ipess_tensors["T_u"].size()) \
            | set(self.ipess_tensors["T_d"].size())
        assert len(aux_bond_dims)==1,"iPESS does not have a uniform aux bond dimension"
        return list(aux_bond_dims)[0]

    def write_to_file(self, outputfile, aux_seq=None, tol=1.0e-14, normalize=False):
        r"""
        See :meth:`write_ipess_kagome_generic`.
        """
        write_ipess_kagome_generic(self, outputfile, tol=tol, normalize=normalize)

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type state: IPESS_KAGOME_GENERIC
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPESS_KAGOME_GENERIC

        Take IPESS_KAGOME_GENERIC and enlarge all auxiliary bond dimensions of ``T_u``, ``T_d``, 
        ``B_a``, ``B_b``, and ``B_c`` tensors to the new size ``new_d``.
        """
        ad= self.get_aux_bond_dims()
        assert new_d>=ad, "Desired dimension is smaller than current aux dimension"
        new_ipess_tensors= dict()
        for k in ['T_u','T_d']:
            new_ipess_tensors[k]= torch.zeros(new_d,new_d,new_d, dtype=self.dtype, device=self.device)
            new_ipess_tensors[k][:ad,:ad,:ad]= self.ipess_tensors[k]
        for k in ['B_a','B_b', 'B_c']:
            new_ipess_tensors[k]= torch.zeros(self.ipess_tensors[k].size(0),new_d,new_d,\
                dtype=self.dtype, device=self.device)
            new_ipess_tensors[k][:,:ad,:ad]= self.ipess_tensors[k]

        new_state= self.__class__(new_ipess_tensors,\
            peps_args=peps_args, global_args=global_args)

        return new_state

def read_ipess_kagome_generic(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in JSON format`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_GENERIC

    Read state from file.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        ipess_tensors= OrderedDict()
        # legacy
        if "elem_tensors" in raw_state.keys():
            assert set(("UP_T","DOWN_T","BOND_S1","BOND_S2","BOND_S3"))\
                ==set(list(raw_state["elem_tensors"].keys())),"missing elementary tensors"
            keymap={"UP_T": "T_u", "DOWN_T": "T_d", "BOND_S1": "B_c","BOND_S3": "B_a","BOND_S2": "B_b"}
            for key,t in raw_state["elem_tensors"].items():
                ipess_tensors[keymap[key]]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

        # default
        elif "ipess_tensors" in raw_state.keys(): 
            assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
                "missing ipess tensors"
            for key,t in raw_state["ipess_tensors"].items():
                ipess_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))
        else:
            raise RuntimeError("Not a valid IPESS_KAGOME_GENERIC state.")

        # convert to correct device and/or dtype
        for key,t in ipess_tensors.items():
             ipess_tensors[key]=  ipess_tensors[key].to(device=global_args.device,dtype=dtype)                  

        state = IPESS_KAGOME_GENERIC(ipess_tensors, peps_args=peps_args, \
            global_args=global_args)
    return state

def write_ipess_kagome_generic(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPESS_KAGOME_GENERIC
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool

    Write state into file.
    """
    #TODO implement cutoff on elements with magnitude below tol
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "ipess_tensors": {}})

    # write list of considered elementary tensors
    for key, t in state.ipess_tensors.items():
        tmp_t= t/t.abs().max() if normalize else t
        json_state["ipess_tensors"][key]= serialize_bare_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

def read_ipess_kagome_generic_legacy(jsonfile, ansatz="IPESS", peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_KAGOME

    Legacy handling of iPESS ansatz. This functions reads input files
    in convention below and maps them into convention of IPESS_KAGOME_GENERIC.

    coord_kagome - ( 0, 0, x=0(B_a),1(B_b),2(B_c),3(T_d),4(T_u) )
    
    unrestricted (ansatz="IPESS")            pg_symmetric (ansatz="IPESS_PG")
    
    -- B_a^a_kl   B_b^b_mn --              -- B_a^a_kl   B_b^b_mn --
         \\       /                               \\       /
          T_d_lno                                   T_d_lno
            |                                        |
          B_c^c_op                                 B_c^c_po
            |                                        |
          T_u_pqr                                   T_u_qrp
        /       \\                                 /       \\
    
    
    where for IPESS_PG, T_d=T_u and B_a=B_b=B_c, where B_a is only symmetric part i.e.
    B_a = 0.5*(B_a + B_a.permute(0,2,1)). NOTE: in the legacy input files, the B_a contains
    also anti-symmetric part and hence it needs to be explicitly symmetrized.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        ipess_tensors= OrderedDict()
        assert "kagome_sites" in raw_state.keys(),"Not a legacy generic iPESS kagome ansatz"
        kagome_tensors= {}
        for ts in raw_state["map"]:
            coord_kagome = (ts["x"],ts["y"],ts["z"])
            t = None
            for s in raw_state["kagome_sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"")

            if "format" in t.keys():
                if t["format"] == "1D":
                    X = torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X = torch.from_numpy(read_bare_json_tensor_np_legacy(t))
            kagome_tensors[coord_kagome] = X.to(device=global_args.device,dtype=dtype)
        assert len(kagome_tensors)==5, "iPESS ansatz for more than single Kagome unit cell"
        
        # unrestricted IPESS
        if ansatz=="IPESS":
            ipess_tensors= {
                'T_u': kagome_tensors[(0,0,4)].permute(0,2,1).contiguous(), 
                'T_d': kagome_tensors[(0,0,3)].permute(2,1,0).contiguous(),\
                'B_a': kagome_tensors[(0,0,0)].permute(0,2,1).contiguous(),\
                'B_b': kagome_tensors[(0,0,1)].permute(0,2,1).contiguous(),\
                'B_c': kagome_tensors[(0,0,2)]}
            state = IPESS_KAGOME_GENERIC(ipess_tensors, peps_args=peps_args, \
                global_args=global_args)
        # pg_symmetric IPESS
        elif ansatz=="IPESS_PG":
            ipess_tensors= {
                'T_u': kagome_tensors[(0,0,4)].permute(2,1,0).contiguous(), 
                'T_d': kagome_tensors[(0,0,3)].permute(2,1,0).contiguous(),\
                'B_a': 0.5*(kagome_tensors[(0,0,0)] + kagome_tensors[(0,0,0)].permute(0,2,1)).contiguous(),\
                'B_b': 0.5*(kagome_tensors[(0,0,1)] + kagome_tensors[(0,0,1)].permute(0,2,1)).contiguous(),\
                'B_c': 0.5*(kagome_tensors[(0,0,2)] + kagome_tensors[(0,0,2)].permute(0,2,1)).contiguous()}
            state = IPESS_KAGOME_PG(ipess_tensors['T_u'], ipess_tensors['B_a'], ipess_tensors['T_d'],\
                SYM_UP_DOWN=True, SYM_BOND_S=True,\
                peps_args=peps_args, global_args=global_args)            
        
    return state


class IPESS_KAGOME_PG(IPESS_KAGOME_GENERIC):
    PG_A1_B= {'T_u': 'A_1', 'T_d': 'A_1', 'B_a': 'B', 'B_b': 'B', 'B_c': 'B'}
    PG_A2_B= {'T_u': 'A_2', 'T_d': 'A_2', 'B_a': 'B', 'B_b': 'B', 'B_c': 'B'}

    def __init__(self, T_u, B_c, T_d=None,\
                B_a=None, B_b=None,\
                SYM_UP_DOWN=True, SYM_BOND_S=True, pgs=None, pg_symmetrize=False,\
                peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param T_u: rank-3 tensor
        :param B_c: rank-3 tensor containing physical degree of freedom
        :param T_d: rank-3 tensor
        :param B_a: rank-3 tensor containing physical degree of freedom
        :param B_b: rank-3 tensor containing physical degree of freedom
        :param SYM_UP_DOWN: is up triangle equivalent to down triangle
        :param SYM_BOND_S: are bond tensors equivalent to each other 
        :param pgs: dictionary assigning point-group irreps to elementary tensors 
        :type pgs: dict(str,str)
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type T_u: torch.tensor
        :type B_c: torch.tensor
        :type T_u: torch.tensor
        :type B_a: torch.tensor
        :type B_b: torch.tensor
        :type SYM_UP_DOWN: bool
        :type SYM_BOND_S: bool
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Single unit-cell iPESS ansatz (3 sites per unit cell) for Kagome lattice 
        with additional spatial symmetries.
        
            * If ``SYM_UP_DOWN``, then `T_d` trivalent tensor is taken to be identical to `T_u`. 
              The choice of the contraction guarantees the same (direction of) chirality on up 
              and down triangles. 
            * If ``SYM_BOND_S``, then `B_a` and `B_b` bond tensors are taken to be identical to `B_c`.

        All non-equivalent tensors can be accessed through member ``elem_tensors``, which is 
        a dictionary::
        
            if SYM_UP_DOWN and SYM_BOND_S

                elem_tensors = {'T_u': torch.Tensor, 'B_c': torch.Tensor}

            if SYM_UP_DOWN

                elem_tensors = {'T_u': torch.Tensor, 'B_c': torch.Tensor, 'B_a': ..., 'B_c': ...}

            etc.

        Argument ``pgs`` is assumed to be dictionary, with keys being the names of elementary 
        ipess tensors. Predefined choices are::

            IPESS_KAGOME_PG.PG_A1_B= {'T_u': 'A_1', 'T_d': 'A_1', 'B_a': 'B', 'B_b': 'B', 'B_c': 'B'}
            IPESS_KAGOME_PG.PG_A2_B= {'T_u': 'A_2', 'T_d': 'A_2', 'B_a': 'B', 'B_b': 'B', 'B_c': 'B'}

        If the elementary tensor is present in the ``pgs``, it is constrained to given irrep 
        of the point group.
        """
        self.SYM_UP_DOWN= SYM_UP_DOWN
        self.SYM_BOND_S= SYM_BOND_S
        
        # default setup
        self.elem_tensors= OrderedDict({'T_u': T_u,'B_c': B_c})
        if not SYM_UP_DOWN:
            assert isinstance(T_d,torch.Tensor),\
                "rank-3 tensor for down triangle must be provided"
            self.elem_tensors['T_d'] = T_d
        if not SYM_BOND_S:
            assert isinstance(B_a,torch.Tensor) and isinstance(B_b,torch.Tensor),\
                "rank-3 tensor for bond 1 and bond 2 must be provided"
            self.elem_tensors['B_a']= B_a
            self.elem_tensors['B_b']= B_b

        # PGs
        if pgs==None: pgs=dict()
        assert isinstance(pgs,dict) and set(list(pgs.keys()))<=set(['T_u','T_d','B_a','B_b','B_c']),\
            "Invalid point-group specification "+str(pgs)
        self.pgs= pgs
        if pg_symmetrize:
            self.elem_tensors= _to_PG_symmetric(self.pgs, self.elem_tensors)

        ipess_tensors= OrderedDict({
            'T_u': self.elem_tensors['T_u'], 
            'T_d': self.elem_tensors['T_u'] if SYM_UP_DOWN else self.elem_tensors['T_d'],\
            'B_c': self.elem_tensors['B_c'], 
            'B_a': self.elem_tensors['B_c'] if SYM_BOND_S else self.elem_tensors['B_a'], 
            'B_b': self.elem_tensors['B_c'] if SYM_BOND_S else self.elem_tensors['B_b']
        })
        
        super().__init__(ipess_tensors, peps_args=peps_args,
                         global_args=global_args)

    def __str__(self):
        print(f"Equivalent up and down triangle: {self.SYM_UP_DOWN}")
        print(f"Equivalent bond tensors: {self.SYM_BOND_S}")
        print(f"Point groups irreps: {self.pgs}")
        super().__str__()
        return ""

    def get_parameters(self):
        r"""
        :return: variational parameters of IPESS_KAGOME_PG
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        In this case member ``elem_tensors``.
        """
        return self.elem_tensors.values()

    def get_checkpoint(self):
        r"""
        :return: all data necessary to reconstruct the state. In this case member ``elem_tensors`` 
        :rtype: dict[str: torch.tensor]
        
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return self.elem_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        elem_t= checkpoint["parameters"]

        # legacy handling
        if "BOND_S" in elem_t.keys() and "UP_T" in elem_t.keys():
            self.elem_tensors= {'T_u': elem_t["UP_T"], 'B_c': elem_t["BOND_S"]}
            if "DOWN_T" in elem_t.keys() and not self.SYM_UP_DOWN: 
                self.elem_tensors['T_d']= elem_t["DOWN_T"]
        elif "BOND_S1" in elem_t.keys() and "UP_T" in elem_t.keys():
            self.elem_tensors= {'T_u': elem_t["UP_T"], 'B_c': elem_t["BOND_S1"]}
            if "DOWN_T" in elem_t.keys() and not self.SYM_UP_DOWN: 
                self.elem_tensors['T_d']= elem_t["DOWN_T"]
            if "BOND_S2" in elem_t.keys() and "BOND_S3" in elem_t.keys() and not SYM_BOND_S: 
                self.elem_tensors['B_b']= elem_t["BOND_S2"]
                self.elem_tensors['B_a']= elem_t["BOND_S3"]
        else:
            self.elem_tensors= elem_t

        # default
        self.ipess_tensors= {'T_u': self.elem_tensors['T_u'], 'T_d': self.elem_tensors['T_u'], \
            'B_a': self.elem_tensors['B_c'], 'B_b': self.elem_tensors['B_c'], \
            'B_c': self.elem_tensors['B_c']}
        if not self.SYM_UP_DOWN:
            self.ipess_tensors['T_d']= self.elem_tensors['T_d']
        if not self.SYM_BOND_S:
            self.ipess_tensors['B_b']= self.elem_tensors['B_b']
            self.ipess_tensors['B_a']= self.elem_tensors['B_a']
        for t in self.elem_tensors.values(): t.requires_grad_(False)

        self.sites = self.build_onsite_tensors()

    def add_noise(self, noise):
        r"""
        :param noise: magnitude of noise
        :type noise: float

        Add uniform random noise to iPESS tensors, respecting the spatial symmetry constraints.
        """
        for k in self.elem_tensors:
            rand_t= torch.rand( self.elem_tensors[k].size(), dtype=self.dtype, device=self.device)
            self.elem_tensors[k]= self.elem_tensors[k] + noise * (rand_t-1.0)
        self.elem_tensors= _to_PG_symmetric(self.pgs, self.elem_tensors)

        # update parent generic kagome iPESS and invoke reconstruction of on-site tensor        
        # default
        self.ipess_tensors= {'T_u': self.elem_tensors['T_u'], 'T_d': self.elem_tensors['T_u'],\
            'B_a': self.elem_tensors['B_c'], 'B_b': self.elem_tensors['B_c'],\
            'B_c': self.elem_tensors['B_c']}
        if not self.SYM_UP_DOWN:
            self.ipess_tensors['T_d']= self.elem_tensors['T_d']
        if not self.SYM_BOND_S:
            self.ipess_tensors['B_b']= self.elem_tensors['B_b']
            self.ipess_tensors['B_a']= self.elem_tensors['B_a']        
        self.sites = self.build_onsite_tensors()

    def write_to_file(self, outputfile, aux_seq=None, tol=1.0e-14, normalize=False,\
        pg_symmetrize=True):
        r"""
        See :meth:`write_ipess_kagome_pg`.
        """
        write_ipess_kagome_pg(self, outputfile, tol=tol, normalize=normalize,\
            pg_symmetrize=pg_symmetrize)

    def extend_bond_dim(self, new_d):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type state: IPESS_KAGOME_PG
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPESS_KAGOME_PG

        Take IPESS_KAGOME_PG and enlarge all auxiliary bond dimensions of ipess tensors 
        to the new size ``new_d``
        """
        ad= self.get_aux_bond_dims()
        assert new_d>=ad, "Desired dimension is smaller than current aux dimension"
        new_elem_tensors= dict()
        new_elem_tensors['T_u']= torch.zeros(new_d,new_d,new_d, dtype=self.dtype, device=self.device)
        new_elem_tensors['T_u'][:ad,:ad,:ad]= self.elem_tensors['T_u']
        new_elem_tensors['B_c']= torch.zeros(self.elem_tensors['B_c'].size(0),new_d,new_d,\
            dtype=self.dtype, device=self.device)
        new_elem_tensors['B_c'][:,:ad,:ad]= self.elem_tensors['B_c']
        if not self.SYM_UP_DOWN:
            new_elem_tensors['T_d']= torch.zeros(new_d,new_d,new_d, dtype=self.dtype, device=self.device)
            new_elem_tensors['T_d'][:ad,:ad,:ad]= self.elem_tensors['T_d']
        if not self.SYM_BOND_S:
            new_elem_tensors['B_b']= torch.zeros(self.elem_tensors['B_b'].size(0),new_d,new_d,\
                dtype=self.dtype, device=self.device)
            new_elem_tensors['B_b'][:,:ad,:ad]= self.elem_tensors['B_b']
            new_elem_tensors['B_a']= torch.zeros(self.elem_tensors['B_a'].size(0),new_d,new_d,\
                dtype=self.dtype, device=self.device)
            new_elem_tensors['B_a'][:,:ad,:ad]= self.elem_tensors['B_a']

        new_state= self.__class__(new_elem_tensors['T_u'], new_elem_tensors['B_c'],\
            T_d=None if self.SYM_UP_DOWN else new_elem_tensors['T_d'],\
            B_a=None if self.SYM_BOND_S else new_elem_tensors['B_a'],\
            B_b=None if self.SYM_BOND_S else new_elem_tensors['B_b'],\
            SYM_UP_DOWN=self.SYM_UP_DOWN, SYM_BOND_S=self.SYM_BOND_S, pgs= self.pgs,\
            peps_args=cfg.peps_args, global_args=cfg.global_args)

        return new_state

def _to_PG_symmetric(pgs, elem_ts):
    pg_elem_ts= OrderedDict(elem_ts)
    for t_id,pg in pgs.items():
        if pg is None: continue
        # bond-tensors        
        if t_id in ["B_a", "B_b", "B_c"] and t_id in elem_ts.keys():
            # A+iB
            if pg=="A":
                pg_elem_ts[t_id]= 0.5*(elem_ts[t_id]\
                    + elem_ts[t_id].permute(0,2,1).conj())
            elif pg=="B":
            # B + iA 
                pg_elem_ts[t_id]= 0.5*(elem_ts[t_id]\
                    - elem_ts[t_id].permute(0,2,1).conj())
            else:
                raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
        # trivalent tensor "up" and "down" 
        if t_id in ["T_u", "T_d"] and t_id in elem_ts.keys():    
            # A_1 + iA_2
            if pg=="A_1":
                tmp_t= (1./3)*(elem_ts[t_id]\
                    + elem_ts[t_id].permute(1,2,0)\
                    + elem_ts[t_id].permute(2,0,1))
                tmp_t= 0.5*(tmp_t + tmp_t.permute(0,2,1).conj())
                pg_elem_ts[t_id]= tmp_t
            # A_2 + iA_1
            elif pg=="A_2":
                tmp_t= (1./3)*(elem_ts[t_id]\
                    + elem_ts[t_id].permute(1,2,0)\
                    + elem_ts[t_id].permute(2,0,1))
                tmp_t= 0.5*(tmp_t - tmp_t.permute(0,2,1).conj())
                pg_elem_ts[t_id]= tmp_t
            else:
                raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
    return pg_elem_ts

def to_PG_symmetric(state, SYM_UP_DOWN=None, SYM_BOND_S=None, pgs=None):
    r"""
    :param state: wavefunction
    :type state: IPESS_KAGOME_PG
    :param SYM_UP_DOWN: make trivalent tensors ``'T_u'`` and ``'T_d'`` identical
    :type SYM_UP_DOWN: bool
    :param SYM_BOND_S: make bond tensors ``'B_a'``, ``'B_b'``, and ``'B_c'`` identical
    :type SYM_BOND_S: bool
    :param pgs: point group irreps for individual ipess tensors 
    :type pgs: dict[str: str]
    :return: symmetrized state
    :rtype: IPESS_KAGOME_PG

    Symmetrize IPESS_KAGOME_PG wavefunction by imposing additional spatial symmetries.
    """
    assert type(state)==IPESS_KAGOME_PG, "Expected IPESS_KAGOME_PG instance"
    if SYM_UP_DOWN is None: 
        SYM_UP_DOWN= state.SYM_UP_DOWN
    if SYM_BOND_S is None: 
        SYM_BOND_S= state.SYM_BOND_S
    if pgs is None: pgs= state.pgs
    
    symm_elem_ts= _to_PG_symmetric(pgs, state.elem_tensors)

    symm_state= state.__class__(symm_elem_ts['T_u'], symm_elem_ts['B_c'],\
            T_d=None if SYM_UP_DOWN else symm_elem_ts['T_d'],\
            B_a=None if SYM_BOND_S else symm_elem_ts['B_a'],\
            B_b=None if SYM_BOND_S else symm_elem_ts['B_b'],\
            SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S, pgs= pgs,\
            peps_args=cfg.peps_args, global_args=cfg.global_args)

    return symm_state

def read_ipess_kagome_pg(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPESS_KAGOME_PG in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_PG

    Read IPESS_KAGOME_PG state from file.
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        SYM_UP_DOWN= True
        if "SYM_UP_DOWN" in raw_state.keys(): SYM_UP_DOWN= raw_state["SYM_UP_DOWN"]
        SYM_BOND_S= True
        if "SYM_BOND_S" in raw_state.keys(): SYM_BOND_S= raw_state["SYM_BOND_S"] 

        pgs=None
        if "pgs" in raw_state.keys():
            # legacy
            if not isinstance(raw_state["pgs"],dict):
                pgs= tuple( raw_state["pgs"] )
                if pgs==(None,None,None): pgs=None
                elif pgs==("A_2","A_2","B"):
                    pgs= {"T_u": "A_2", "T_d": "A_2", "B_c": "B", "B_a": "B", "B_b": "B"}
            else:
                pgs= raw_state["pgs"]

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        elem_t= OrderedDict()
        for key,t in raw_state["elem_tensors"].items():
            elem_t[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                .to(dtype=global_args.torch_dtype, device=global_args.device)

        # legacy
        if "UP_T" in elem_t.keys() and "BOND_S" in elem_t.keys():
            elem_tensors= {'T_u': elem_t["UP_T"], 'B_c': elem_t["BOND_S"]}
            if "DOWN_T" in elem_t.keys() and not SYM_UP_DOWN: 
                elem_tensors['T_d']= elem_t["DOWN_T"]
        elif "UP_T" in elem_t.keys() and "BOND_S1" in elem_t.keys():
            elem_tensors= {'T_u': elem_t["UP_T"], 'B_c': elem_t["BOND_S1"]}
            if "DOWN_T" in elem_t.keys() and not SYM_UP_DOWN: 
                elem_tensors['T_d']= elem_t["DOWN_T"]
            if "BOND_S2" in elem_t.keys() and "BOND_S3" in elem_t.keys() and not SYM_BOND_S: 
                elem_tensors['B_b']= elem_t["BOND_S2"]
                elem_tensors['B_a']= elem_t["BOND_S3"]
        else:
            elem_tensors= elem_t

        if SYM_UP_DOWN and SYM_BOND_S:
            assert set(('T_u', 'B_c')) <= set(list(elem_tensors.keys())),\
                "missing elementary tensors"
        elif not SYM_UP_DOWN and SYM_BOND_S:
            assert set(('T_u', 'B_c', 'T_d')) <= set(list(elem_tensors.keys())),\
                "missing elementary tensors"
        elif SYM_UP_DOWN and not SYM_BOND_S:
            assert set(('T_u', 'B_c', 'B_b','B_a')) <= set(list(elem_tensors.keys())),\
                "missing elementary tensors"
        else:
            assert set(('T_u', 'B_c', 'T_d','B_a','B_b')) <= set(list(elem_tensors.keys())),\
                "missing elementary tensors"

        if SYM_UP_DOWN: elem_tensors['T_d']=None
        if SYM_BOND_S: 
            elem_tensors['B_a']=None
            elem_tensors['B_b']=None

        state = IPESS_KAGOME_PG(elem_tensors['T_u'], elem_tensors['B_c'], \
            T_d=elem_tensors['T_d'], B_a= elem_tensors['B_a'],\
            B_b=elem_tensors['B_b'], SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S,\
            pgs= pgs, peps_args=peps_args, global_args=global_args)
    return state

def write_ipess_kagome_pg(state, outputfile, tol=1.0e-14, normalize=False, pg_symmetrize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPESS_KAGOME_PG
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    :param pg_symmetrize: symmetrize state before writing out
    :type pg_symmetrize: bool

    Write state to file.
    """
    # TODO drop constrain for aux bond dimension to be identical on all bond indices
    # TODO implement cutoff on elements with magnitude below tol
    sym_state= to_PG_symmetric(state) if pg_symmetrize else state
    json_state = dict({"elem_tensors": {}, "SYM_UP_DOWN": sym_state.SYM_UP_DOWN, \
        "SYM_BOND_S": sym_state.SYM_BOND_S, "pgs": sym_state.pgs})

    # write list of considered elementary tensors
    for key, t in sym_state.elem_tensors.items():
        tmp_t= t/t.abs().max() if normalize else t
        json_state["elem_tensors"][key]= serialize_bare_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))


class IPESS_KAGOME_PG_LC(IPESS_KAGOME_PG):

    def __init__(self, T_u, B_c, T_d=None,\
                B_a=None, B_b=None,\
                SYM_UP_DOWN=True, SYM_BOND_S=True, pgs=None,\
                peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param T_u: tuple of vector with real coefficients and list of basis tensors
                    defining trivalent tensor as linear combination    
        :param B_c: tuple of vector with real coefficients and list of basis tensors
                    defining bond tensor as linear combination 
        :param T_d: analogous to T_u
        :param B_a: analogous to B_c
        :param B_b: analogous to B_c
        :param SYM_UP_DOWN: is up triangle equivalent to down triangle
        :param SYM_BOND_S: are bond tensors equivalent to each other 
        :param pgs: dictionary assigning point-group irreps to basis tensors 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type T_u: tuple(torch.tensor, list(tuple(dict,torch.tensor)))
        :type B_c: tuple(torch.tensor, list(tuple(dict,torch.tensor)))
        :type T_u: tuple(torch.tensor, list(tuple(dict,torch.tensor)))
        :type B_a: tuple(torch.tensor, list(tuple(dict,torch.tensor)))
        :type B_b: tuple(torch.tensor, list(tuple(dict,torch.tensor)))
        :type SYM_UP_DOWN: bool
        :type SYM_BOND_S: bool
        :type pgs: dict[str : str]
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Single unit-cell iPESS ansatz (3 sites per unit cell) for Kagome lattice 
        with elementary tensors built as linear combination of supplied basis tensors.

        Each basis tensor is described by a dict::
            
            T_u= (torch.Tensor, [
                ...,
                ({"meta": {"pg": "A_1"}, ...}, torch.Tensor),
                ...,
            ])

        where the value of "pg" (specified inside dict "meta") is either "A_1" or "A_2" for 
        trivalent tensors ``'T_u'``, ``'T_d'`` and "A" or "B" for bond tensors ``'B_a'``,
        ``'B_b'``, ``'B_c'``.

        Coefficients and basis tensors can be accessed in member dictionaries ``coeffs`` 
        and ``basis_t`` respectively.
        
        If ``SYM_UP_DOWN``, then `T_d` trivalent tensor is taken to be identical to `T_u`. 
        The choice of the contraction guarantees the same (direction of) chirality on up 
        and down triangles. 

        If ``SYM_BOND_S``, then `B_a` and `B_b` bond tensors are taken to be identical to `B_c`.

        The ``pgs`` specifies point-group irreps for elementary tensors, see :class:`IPESS_KAGOME_PG`.
        Only basis tensors of selected point-group irrep are used to construct ipess tensors. 
        """
        self.SYM_UP_DOWN= SYM_UP_DOWN
        self.SYM_BOND_S= SYM_BOND_S
        
        # default setup
        self.coeffs= OrderedDict({'T_u': T_u[0],'B_c': B_c[0]})
        self.basis_t= OrderedDict({'T_u': T_u[1],'B_c': B_c[1]})
        # ipess_tensors= OrderedDict({'T_u': T_u[1], 'T_d': T_u[1],\
        #     'B_c': B_c[1], 'B_a': B_c[1], 'B_b': B_c[1]})
        if not SYM_UP_DOWN:
            assert isinstance(T_d[0], torch.Tensor),\
                "coefficients and basis tensors for T_d have to be provided"
            self.coeffs['T_d']= T_d[0]  
            self.basis_t['T_d']= T_d[1]
        if not SYM_BOND_S:
            assert isinstance(B_a[0],torch.Tensor) and isinstance(B_b[0],torch.Tensor),\
                "coefficients and basis tensors for bond 1 and bond 2 must be provided"
            self.coeffs['B_a']= B_a[0]
            self.coeffs['B_b']= B_b[0]
            self.basis_t['B_a']= B_a[1]
            self.basis_t['B_b']= B_b[1]

        # PGs
        if pgs==None: pgs=dict()
        assert isinstance(pgs,dict) and set(list(pgs.keys()))<=set(['T_u','T_d','B_a','B_b','B_c']),\
            "Invalid point-group specification "+str(pgs)
        self.pgs= pgs
        
        elem_tensors= self.build_elem_tensors()
        super().__init__(**elem_tensors,\
            SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S, pgs=pgs,\
            pg_symmetrize=False, peps_args=peps_args, global_args=global_args)

    def __str__(self):
        for k in self.coeffs.keys():
            print(f"{k}")
            for m_t in self.basis_t[k]:
                print(f"{m_t[0]}")
        super().__str__()
        return ""

    def get_parameters(self):
        r"""
        :return: variational parameters of IPESS_KAGOME_PG_LC
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        In this case member ``coeffs``.
        """
        return self.coeffs.values()

    def get_checkpoint(self):
        r"""
        :return: all data necessary to reconstruct the state. In this case dict containing 
                 members ``coeffs`` and ``basis_t``
        :rtype: dict[str: dict[str: torch.Tensor], str: dict[str: list(tuple(dict,torch.Tensor))]]
        
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return dict(coeffs= self.coeffs, basis_t= self.basis_t)

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.coeffs= checkpoint["parameters"]["coeffs"]
        self.basis_t= checkpoint["parameters"]["basis_t"]
        self.update_()

    @staticmethod
    def create_from_checkpoint(checkpoint_file, SYM_UP_DOWN=True, SYM_BOND_S=True,\
            pgs=None, peps_args=cfg.peps_args, global_args=cfg.global_args):
        checkpoint= torch.load(checkpoint_file, map_location=global_args.device, weights_only=False)
        coeffs= checkpoint["parameters"]["coeffs"]
        basis_t= checkpoint["parameters"]["basis_t"]
        c_b= { ind: (coeffs[ind], basis_t[ind]) for ind in coeffs.keys() }
        return IPESS_KAGOME_PG_LC( c_b['T_u'], c_b['B_c'],\
            T_d=c_b['T_d'] if 'T_d' in c_b else None,\
            B_a=c_b['B_a'] if 'B_a' in c_b else None,\
            B_b=c_b['B_b'] if 'B_b' in c_b else None,\
            SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S, pgs=pgs,\
            peps_args=peps_args, global_args=global_args)

    def build_elem_tensors(self):
        r"""
        :return: elementary tensors
        :rtype: dict[str: torch.Tensor]

        Construct elementary tensors ``'T_u'``, ``'B_c'`` and optionally ``'T_d'``, ``'B_a'``, ``'B_b'``
        as linear combinations of basis tensors with real coefficients.
        """
        elem_tensors=dict()
        for k in self.coeffs:
            if k in ['T_u', 'T_d']:
                if k in self.pgs.keys() and  self.pgs[k]=="A_1":
                    sym_t_A1= list(filter(lambda x: x[0]["meta"]["pg"]=="A_1", self.basis_t[k]))
                    sym_t_A2= list(filter(lambda x: x[0]["meta"]["pg"]=="A_2", self.basis_t[k]))
                    ts= torch.stack( [t for m,t in sym_t_A1] + [ 1.0j*t for m,t in sym_t_A2] )
                elif k in self.pgs.keys() and  self.pgs[k]=="A_2":
                    sym_t_A1= list(filter(lambda x: x[0]["meta"]["pg"]=="A_1", self.basis_t[k]))
                    sym_t_A2= list(filter(lambda x: x[0]["meta"]["pg"]=="A_2", self.basis_t[k]))
                    ts= torch.stack( [t for m,t in sym_t_A2] + [ 1.0j*t for m,t in sym_t_A1] )
                else:
                    ts= torch.stack( [t for m,t in self.basis_t[k]] )

            elif k in ['B_a', 'B_b', 'B_c']:
                if k in self.pgs.keys() and  self.pgs[k]=="A":
                    sym_t_A= list(filter(lambda x: x[0]["meta"]["pg"]=="A", self.basis_t[k]))
                    sym_t_B= list(filter(lambda x: x[0]["meta"]["pg"]=="B", self.basis_t[k]))
                    ts= torch.stack( [t for m,t in sym_t_A] + [ 1.0j*t for m,t in sym_t_B] )
                elif k in self.pgs.keys() and  self.pgs[k]=="B":
                    sym_t_A= list(filter(lambda x: x[0]["meta"]["pg"]=="A", self.basis_t[k]))
                    sym_t_B= list(filter(lambda x: x[0]["meta"]["pg"]=="B", self.basis_t[k]))
                    ts= torch.stack( [t for m,t in sym_t_B] + [ 1.0j*t for m,t in sym_t_A] )
                else:
                    ts= torch.stack( [t for m,t in self.basis_t[k]] )

            c= self.coeffs[k].clone()
            if ts.is_complex(): c= c*(1.0+0.j)
            elem_tensors[k]= torch.einsum('i,iabc->abc',c,ts)
        return elem_tensors

    def update_(self):
        r"""
        Update parent classes :class:`IPESS_KAGOME_PG`, :class:`IPESS_KAGOME_GENERIC`, 
        and :class:`IPEPS_KAGOME`. First, invoking reconstruction of elementary tensors 
        by :meth:`build_elem_tensors` and then construct rank-5 iPEPS by :meth:`build_onsite_tensors`.
        """
        self.elem_tensors= self.build_elem_tensors()
        self.ipess_tensors= {'T_u': self.elem_tensors['T_u'], 'T_d': self.elem_tensors['T_u'],\
            'B_a': self.elem_tensors['B_c'], 'B_b': self.elem_tensors['B_c'],\
            'B_c': self.elem_tensors['B_c']}
        if not self.SYM_UP_DOWN:
            self.ipess_tensors['T_d']= self.elem_tensors['T_d']
        if not self.SYM_BOND_S:
            self.ipess_tensors['B_b']= self.elem_tensors['B_b']
            self.ipess_tensors['B_a']= self.elem_tensors['B_a']

        self.sites = self.build_onsite_tensors()

    def add_noise(self, noise):
        r"""
        :param noise: magnitude of noise
        :type noise: float

        Add uniform random noise to coefficients of linear combinations.
        """
        for k in self.coeffs:
            rand_t= torch.rand_like( self.coeffs[k] )
            self.coeffs[k]= self.coeffs[k] + noise * (rand_t-1.0)
        self.update_()

    def extend_bond_dim(self, new_d):
        raise NotImplementedError("")

    def write_to_file(self, outputfile, tol=1.0e-14, normalize=False):
        r"""
        See :meth:`write_ipess_kagome_pg_lc`.
        """
        write_ipess_kagome_pg_lc(self, outputfile, tol=tol, normalize=normalize)

def write_ipess_kagome_pg_lc(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPESS_KAGOME_PG_LC
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool

    Write state to file.
    """
    #TODO implement cutoff on elements with magnitude below tol
    json_state=dict({"pgs": state.pgs , "basis_t": {}, "coeffs": {}, \
        "SYM_UP_DOWN": state.SYM_UP_DOWN, "SYM_BOND_S": state.SYM_BOND_S})
    for k in state.basis_t.keys():
        json_state["basis_t"][k]= []
        for m_t in state.basis_t[k]:
            json_state["basis_t"][k].append(
                serialize_basis_t(*m_t)
                )
    for k in state.coeffs.keys():
        tmp_t= state.coeffs[k]
        if normalize:
            tmp_t= tmp_t/tmp_t.abs().max()
        json_state["coeffs"][k]= serialize_basis_t(None, tmp_t)

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

def read_ipess_kagome_pg_lc(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPESS_KAGOME_PG_LC in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_PG_LC

    Read IPESS_KAGOME_PG_LC state from file.
    """
    with open(jsonfile) as j:
        raw_state = json.load(j)

        SYM_UP_DOWN= True
        if "SYM_UP_DOWN" in raw_state.keys(): 
            SYM_UP_DOWN= raw_state["SYM_UP_DOWN"]
        SYM_BOND_S= True
        if "SYM_BOND_S" in raw_state.keys(): 
            SYM_BOND_S= raw_state["SYM_BOND_S"] 

        pgs=None
        if "pgs" in raw_state.keys():
            # legacy
            if not isinstance(raw_state["pgs"],dict):
                pgs= tuple( raw_state["pgs"] )
                if pgs==(None,None,None): pgs=None
                elif pgs==("A_2","A_2","B"):
                    pgs= {"T_u": "A_2", "T_d": "A_2", "B_c": "B", "B_a": "B", "B_b": "B"}
            else:
                pgs= raw_state["pgs"]

        basis_t= dict()
        for k in raw_state["basis_t"].keys():
            basis_t[k]= []
            for b_t in raw_state["basis_t"][k]:
                basis_t[k].append( read_basis_t(b_t, device=global_args.device) )
        coeffs= dict()
        for k in raw_state["coeffs"].keys():
            _, coeffs[k]= read_basis_t(raw_state["coeffs"][k], device=global_args.device)

    pg_lc_tensors= { k: (coeffs[k],basis_t[k]) for k in coeffs.keys() }
    state= IPESS_KAGOME_PG_LC( **pg_lc_tensors,\
                SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S, pgs=pgs,\
                peps_args=peps_args, global_args=global_args)

    return state