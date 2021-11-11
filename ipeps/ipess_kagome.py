import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from ipeps.tensor_io import *

class IPESS_KAGOME_GENERIC(ipeps.IPEPS):
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

        iPESS ansatz for Kagome composes five tensors T_u, T_d, B_a, B_b, and B_c within 
        elementary unit cell into regular iPEPS. The B_* tensors hold physical degrees of 
        freedom, which reside on corners shared between different triangles, described by
        tensors T_u or T_d for up and down triangles respectively. 
        These tensors are passed in a dictionary with corresponding keys "T_u", "T_d",...
        The on-site tensors of corresponding iPEPS is obtained by the following contraction::
                                                     
                 2(d)            2(c)                    a
                  \             /          rot. pi       |
             0(w)==B_a         B_b==0(v)   clockwise  b--\                     
                    \         /             =>            \
                    1(l)     1(k)                         s0--s2--d
                     2(l)   1(k)                           | / 
                       \   /                               |/   <- DOWN_T
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
        return self.ipess_tensors.values()

    def get_checkpoint(self):
        return self.ipess_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.ipess_tensors= checkpoint["parameters"]
        for t in self.ipess_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()

    def build_onsite_tensors(self):
        r"""
        Build on-site tensor of corresponding iPEPS
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
        current_auxd= self.get_aux_bond_dims()
        assert new_d>=current_auxd, "Desired dimension is smaller than current aux dimension"
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
    :param jsonfile: input file describing iPEPS in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice
                         into a vertex within elementary unit cell
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS


    A simple PBC ``vertexToSite`` function is used by default

    Parameter ``aux_seq`` defines the expected order of auxiliary indices
    in input file relative to the convention fixed in tn-torch::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3
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
                ipess_tensors[keymap[key]]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                    .to(global_args.device)
        # default
        elif "ipess_tensors" in raw_state.keys(): 
            assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
                "missing ipess tensors"
            for key,t in raw_state["ipess_tensors"].items():
                ipess_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                    .to(global_args.device)
        else:
            raise RuntimeError("Not a valid IPESS_KAGOME_GENERIC state.")

        state = IPESS_KAGOME_GENERIC(ipess_tensors, peps_args=peps_args, \
            global_args=global_args)
    return state

def write_ipess_kagome_generic(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    Parameter ``aux_seq`` defines the order of auxiliary indices relative to the convention
    fixed in tn-torch in which the tensor elements are written out::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3

    TODO drop constrain for aux bond dimension to be identical on
    all bond indices

    TODO implement cutoff on elements with magnitude below tol
    """
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "ipess_tensors": {}})

    # write list of considered elementary tensors
    for key, t in state.ipess_tensors.items():
        tmp_t= t/t.abs().max() if normalize else t
        json_state["ipess_tensors"][key]= serialize_bare_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))


class IPESS_KAGOME_PG(IPESS_KAGOME_GENERIC):
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

        iPESS ansatz for Kagome lattice with additional spatial symmetries.

               2(d)            2(c)                      a
                  \             /          rot. pi       |
             0(w)==B_a         B_b==0(v)  clockwise   b--\                     
                    \         /             =>            \
                    1(l)     1(k)                         s0--s2--d
                     2(l)   1(k)                           | / 
                       \   /                               |/   <- DOWN_T
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

        where ``B_c`` holds ``s0`` DoF and ``B_a`` and ``B_b`` hold ``s2`` and ``s1`` DoFs
        respectively.

        In case T_u==T_d, the choice of contraction guarantees the same (direction of) chirality on
        up and down triangles. 

        Argument ``pgs`` is assumed to be dictionary, with keys equivalent to names of elementary 
        tensors, i.e. ``{'T_u': 'A2', 'B_a': 'A'}``.
        """
        self.SYM_UP_DOWN= SYM_UP_DOWN
        self.SYM_BOND_S= SYM_BOND_S
        
        # default setup
        self.elem_tensors= OrderedDict({'T_u': T_u,'B_c': B_c})
        ipess_tensors= OrderedDict({'T_u': T_u, 'T_d': T_u,\
            'B_c': B_c, 'B_a': B_c, 'B_b': B_c})
        if not SYM_UP_DOWN:
            assert isinstance(T_d,torch.Tensor),\
                "rank-3 tensor for down triangle must be provided"
            self.elem_tensors['T_d']=ipess_tensors['T_d'] = T_d
        if not SYM_BOND_S:
            assert isinstance(B_a,torch.Tensor) and isinstance(B_b,torch.Tensor),\
                "rank-3 tensor for bond 1 and bond 2 must be provided"
            self.elem_tensors['B_a']=ipess_tensors['B_a']= B_a
            self.elem_tensors['B_b']=ipess_tensors['B_b']= B_b

        # PGs
        if pgs==None: pgs=dict()
        assert isinstance(pgs,dict) and set(list(pgs.keys()))<=set(['T_u','T_d','B_a','B_b','B_c']),\
            "Invalid point-group specification "+str(pgs)
        self.pgs= pgs
        if pg_symmetrize:
            self.elem_tensors= _to_PG_symmetric(self.pgs, self.elem_tensors)
        
        super().__init__(ipess_tensors, peps_args=peps_args,
                         global_args=global_args)

    def __str__(self):
        print(f"Equivalent up and down triangle: {self.SYM_UP_DOWN}")
        print(f"Equivalent bond tensors: {self.SYM_BOND_S}")
        print(f"Point groups irreps: {self.pgs}")
        super().__str__()
        return ""

    def get_parameters(self):
        return self.elem_tensors.values()

    def get_checkpoint(self):
        return self.elem_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
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

    def write_to_file(self, outputfile, aux_seq=None, tol=1.0e-14, normalize=False):
        write_ipess_kagome_pg(self, outputfile, tol=tol, normalize=normalize)

    def extend_bond_dim(self, new_d):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type state: IPESS_KAGOME_PG
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPESS_KAGOME_PG

        Take IPESS_KAGOME_PG and enlarge all auxiliary bond dimensions of T_u, [T_d,] B_a, 
        to the new size ``new_d``
        """
        current_auxd= self.get_aux_bond_dims()
        assert new_d>=current_auxd, "Desired dimension is smaller than current aux dimension"
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
    pg_elem_ts= OrderedDict({})
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
            # A_2 + iA_1
            if pg=="A_2":
                tmp_t= (1./3)*(elem_ts[t_id]\
                    + elem_ts[t_id].permute(1,2,0)\
                    + elem_ts[t_id].permute(2,0,1))
                tmp_t= 0.5*(tmp_t - tmp_t.permute(0,2,1).conj())
                pg_elem_ts[t_id]= tmp_t
            else:
                raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
    return pg_elem_ts

def to_PG_symmetric(state, SYM_UP_DOWN=None, SYM_BOND_S=None, pgs=None):
    assert type(state)==IPESS_KAGOME_PG, "Expected IPESS_KAGOME_PG instance"
    if SYM_UP_DOWN is None: SYM_UP_DOWN= state.SYM_UP_DOWN
    if SYM_BOND_S is None: SYM_BOND_S= state.SYM_BOND_S
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
    :param jsonfile: input file describing iPEPS in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice
                         into a vertex within elementary unit cell
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS


    A simple PBC ``vertexToSite`` function is used by default

    Parameter ``aux_seq`` defines the expected order of auxiliary indices
    in input file relative to the convention fixed in tn-torch::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3
    """
    dtype = global_args.torch_dtype

    with open(jsonfile) as j:
        raw_state = json.load(j)

        SYM_UP_DOWN= raw_state["SYM_UP_DOWN"]
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
                .to(global_args.device)

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

        import pdb; pdb.set_trace()

        state = IPESS_KAGOME_PG(elem_tensors['T_u'], elem_tensors['B_c'], \
            T_d=elem_tensors['T_d'], B_a= elem_tensors['B_a'],\
            B_b=elem_tensors['B_b'], SYM_UP_DOWN=SYM_UP_DOWN, SYM_BOND_S=SYM_BOND_S,\
            pgs= pgs, peps_args=peps_args, global_args=global_args)
    return state

def write_ipess_kagome_pg(state, outputfile, tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS
    :type ouputfile: str or Path object
    :type aux_seq: list[int]
    :type tol: float
    :type normalize: bool

    Parameter ``aux_seq`` defines the order of auxiliary indices relative to the convention
    fixed in tn-torch in which the tensor elements are written out::

         0
        1A3 <=> [up, left, down, right]: aux_seq=[0,1,2,3]
         2

        for alternative order, eg.

         1
        0A2 <=> [left, up, right, down]: aux_seq=[1,0,3,2]
         3

    TODO drop constrain for aux bond dimension to be identical on
    all bond indices

    TODO implement cutoff on elements with magnitude below tol
    """
    json_state = dict({"elem_tensors": {}, "SYM_UP_DOWN": state.SYM_UP_DOWN, \
        "SYM_BOND_S": state.SYM_UP_DOWN, "pgs": state.pgs})

    # write list of considered elementary tensors
    for key, t in state.elem_tensors.items():
        tmp_t= t/t.abs().max() if normalize else t
        json_state["elem_tensors"][key]= serialize_bare_tensor_legacy(tmp_t)

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
            kagome_tensors[coord_kagome] = X.to(device=global_args.device)
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
