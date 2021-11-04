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
        assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
            "missing elementary tensors"
        for key,t in raw_state["ipess_tensors"].items():
            ipess_tensors[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                .to(global_args.device)

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
    def __init__(self, triangle_up, bond_site, triangle_down=None, 
                 SYM_UP_DOWN=True, pgs=None,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param triangle_up: rank-3 tensor
        :param bond_site: rank-3 tensor containing physical degree of freedom
        :param triangle_down: rank-3 tensor
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type triangle_up: torch.tensor
        :type triangle_down: torch.tensor
        :type triangle_up: torch.tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        iPESS ansatz for Kagome lattice with additional spatial symmetries.

               2(d)            2(c)                      a
                  \             /          rot. pi       |
             0(w)==B           B==0(v)   clockwise    b--\                     
                    \         /             =>            \
                    1(l)     1(k)                         s0--s2--d
                     2(l)   1(k)                           | / 
                       \   /                               |/   <- DOWN_T
                        T_d                               s1
                         |                                 |
                         0(j)                              c
                         1(j)                               
                         |                 
                         B==0(u)        
                         |
                         2(i)
                         0(i)  
                         |
                        T_u
                       /   \ 
                     1(a)   2(b)

        In case T_u==T_d, the choice of contraction guarantees same (direction of) chirality on
        up and down triangles.
        """
        self.SYM_UP_DOWN= SYM_UP_DOWN
        if pgs==None: pgs= (None,None,None)
        assert isinstance(pgs,tuple) and len(pgs)==3,"Invalid point-group symmetries"
        self.pgs= pgs
        self.elem_tensors= OrderedDict({'T_u': triangle_up, 'B_a': bond_site})
        if not SYM_UP_DOWN:
            assert isinstance(triangle_down,torch.Tensor),\
                "rank-3 tensor for down triangle must be provided"
            self.elem_tensors['T_d']= triangle_down
        else:
            self.elem_tensors['T_d']= triangle_up

        # TODO? self.to_PG_symmetric_()
        ipess_tensors= {'T_u': triangle_up, 'T_d': self.elem_tensors['T_d'], \
            'B_a': bond_site, 'B_b': bond_site, 'B_c': bond_site}
        super().__init__(ipess_tensors, peps_args=peps_args,
                         global_args=global_args)

    def __str__(self):
        print(f"Symmetric up and down triangle: {self.SYM_UP_DOWN}")
        print(f"Point groups irreps of (T_u, T_d, B_a): {self.pgs}")
        super().__str__()
        return ""

    def get_parameters(self):
        if self.SYM_UP_DOWN:
            return [self.elem_tensors['T_u'], self.elem_tensors['B_a']]
        else:
            return self.elem_tensors

    def get_checkpoint(self):
        return self.elem_tensors

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        elem_t= checkpoint["parameters"]

        # legacy handling
        if "BOND_S" in elem_t.keys() and "UP_T" in elem_t.keys():
            self.elem_tensors= {'T_u': elem_t["UP_T"], 'B_a': elem_t["BOND_S"]}
            if "DOWN_T" in elem_t.keys(): self.elem_tensors['T_d']= elem_t["DOWN_T"]
        else:
            self.elem_tensors= elem_t

        if self.SYM_UP_DOWN:
            self.elem_tensors['T_d']= self.elem_tensors['T_u']
        for t in self.elem_tensors.values(): t.requires_grad_(False)

        ipess_tensors= {'T_u': self.elem_tensors['T_u'], 'T_d': self.elem_tensors['T_d'], \
            'B_a': self.elem_tensors['B_a'], 'B_b': self.elem_tensors['B_b'], \
            'B_c': self.elem_tensors['B_c']}

        self.ipess_tensors= ipess_tensors
        self.sites = self.build_onsite_tensors()

    def add_noise(self, noise):
        for k in self.elem_tensors:
            rand_t= torch.rand( self.elem_tensors[k].size(), dtype=self.dtype, device=self.device)
            self.elem_tensors[k]= self.elem_tensors[k] + noise * (rand_t-1.0)
        if self.SYM_UP_DOWN:
            self.elem_tensors['T_d']= self.elem_tensors['T_u']
        self.elem_tensors= _to_PG_symmetric(self.pgs, self.elem_tensors)
        # update parent generic kagome iPESS and invoke reconstruction of on-site tensor
        self.ipess_tensors= {'T_u': self.elem_tensors['T_u'], 'T_d': self.elem_tensors['T_d'],\
            'B_a': self.elem_tensors['B_a'], 'B_b': self.elem_tensors['B_a'],\
            'B_c': self.elem_tensors['B_a']}
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
        new_elem_tensors['T_d']= torch.zeros(new_d,new_d,new_d, dtype=self.dtype, device=self.device)
        new_elem_tensors['T_d'][:ad,:ad,:ad]= self.elem_tensors['T_d']
        new_elem_tensors['B_a']= torch.zeros(self.elem_tensors['B_a'].size(0),new_d,new_d,\
            dtype=self.dtype, device=self.device)
        new_elem_tensors['B_a'][:,:ad,:ad]= self.elem_tensors['B_a']

        new_state= self.__class__(new_elem_tensors['T_u'], new_elem_tensors['B_a'],\
            triangle_down=None if self.SYM_UP_DOWN else new_elem_tensors['T_d'],\
            SYM_UP_DOWN=self.SYM_UP_DOWN, pgs= self.pgs,\
            peps_args=cfg.peps_args, global_args=cfg.global_args)

        return new_state

def _to_PG_symmetric(pgs, elem_t):
    if pgs==(None, None, None): return elem_t

    # A + iB
    if pgs[2]=="A":
        elem_t["B_a"]= 0.5*(elem_t["B_a"]\
            + elem_t["B_a"].permute(0,2,1).conj())
    # B + iA
    if pgs[2]=="B": 
        elem_t["B_a"]= 0.5*(elem_t["B_a"]\
            - elem_t["B_a"].permute(0,2,1).conj())
    else:
        raise RuntimeError("Unsupported point-group "+pgs[2])

    # trivalent tensor "up" and "down" A_2 + iA_1
    for pg, elem_t_id in zip( pgs[0:2], ("T_u", "T_d") ):
        if pg=="A_2":
            elem_t[elem_t_id]= (1./3)*(elem_t[elem_t_id]\
                + elem_t[elem_t_id].permute(1,2,0)\
                + elem_t[elem_t_id].permute(2,0,1))
            elem_t[elem_t_id]= \
                0.5*(elem_t[elem_t_id] - elem_t[elem_t_id].permute(0,2,1).conj())
        else:
            raise RuntimeError("Unsupported point-group "+pgs[1])
    return elem_t

def to_PG_symmetric(state, SYM_UP_DOWN=False, pgs=(None,None,None)):
    assert type(state)==IPESS_KAGOME_PG, "Expected IPESS_KAGOME_PG instance"
    
    symm_elem_t= _to_PG_symmetric(pgs, state.elem_tensors)

    symm_state= IPESS_KAGOME_PG(symm_elem_t["T_u"], symm_elem_t["B_a"], \
        triangle_down=None if SYM_UP_DOWN else symm_elem_t["T_d"], \
        SYM_UP_DOWN= SYM_UP_DOWN, pgs=pgs,
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

        pgs=None
        if "pgs" in raw_state.keys():
            pgs= tuple( raw_state["pgs"] )

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        elem_t= OrderedDict()
        for key,t in raw_state["elem_tensors"].items():
            elem_t[key]= torch.from_numpy(read_bare_json_tensor_np_legacy(t))\
                .to(global_args.device)
        if "UP_T" in elem_t.keys() and "BOND_S" in elem_t.keys():
            elem_tensors= {'T_u': elem_t["UP_T"], 'B_a': elem_t["BOND_S"]}
            if "DOWN_T" in elem_t.keys(): elem_tensors['T_d']= elem_t["DOWN_T"]
        else:
            elem_tensors= elem_t
        assert set(('T_u', 'B_a', 'T_d'))==set(list(elem_tensors.keys())),\
            "missing elementary tensors"

        if SYM_UP_DOWN: elem_tensors['T_d']=None

        state = IPESS_KAGOME_PG(elem_tensors['T_u'], elem_tensors['B_a'], \
            triangle_down=elem_tensors['T_d'], SYM_UP_DOWN=SYM_UP_DOWN, \
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
        "pgs": list(state.pgs)})

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
                SYM_UP_DOWN=True, peps_args=peps_args, global_args=global_args)            
        
    return state
