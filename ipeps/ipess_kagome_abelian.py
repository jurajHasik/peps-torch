import warnings
from collections import OrderedDict
from itertools import chain
import json
import math
import torch
import config as cfg
try:
    import torch
    from ipeps.ipess_kagome import IPESS_KAGOME_GENERIC
except ImportError as e:
    warnings.warn("torch not available", Warning)
import yastn.yastn as yastn
import ipeps.ipeps_kagome_abelian as ipeps_kagome
from ipeps.tensor_io import *

class IPESS_KAGOME_GENERIC_ABELIAN(ipeps_kagome.IPEPS_KAGOME_ABELIAN):
    def __init__(self, settings, ipess_tensors, build_sites=True,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param ipess_tensors: dictionary of five tensors, which make up Kagome iPESS
                              ansatz 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type ipess_tensors: dict(str, yastn.Tensor)
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        iPESS ansatz for Kagome lattice composes five tensors, specified
        by dictionary::

            ipess_tensors = {'T_u': yastn.Tensor, 'T_d': ..., 'B_a': ..., 'B_b': ..., 'B_c': ...}

        into a single rank-5 on-site tensor of parent IPEPS. These iPESS tensors can
        be accessed through member ``ipess_tensors``. 

        The ``'B_*'`` are rank-3 tensors, with index structure [p,i,j] where the first index `p` 
        is for physical degree of freedom, while indices `i` and `j` are auxiliary.
        These bond tensors reside on corners shared between different triangles of Kagome lattice. 
        Bond tensors are connected by rank-3 trivalent tensors ``'T_u'``, ``'T_d'`` on up and down triangles 
        respectively. Trivalent tensors have only auxiliary indices.
    
        The on-site tensors of corresponding iPEPS is obtained by the following contraction::
                                                     
                 2(d)            2(c)                    (-)a
                  \             /          rot. pi          |
             0(w)==B_a         B_b==0(v)   clockwise  (-)b--\                     
                    \         /             =>               \
                    1(l)     1(k)                            s0--s2--d(+)
                     2(l)   1(k)                              | / 
                       \   /                                  |/   <- DOWN_T
                        T_d                                  s1
                         |                                    |
                         0(j)                                 c(+)
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

        where the signature convetions of trivalent and bond tensors are as follows::

            (-)\ /(-)     (-)|             |(-)
                T_d   ,     T_u     , (+)--B--(+)
                |(-)    (-)/   \(-)
        
        This choice guarantees that the signature of on-site tensor of equivalent single-site 
        IPEPS_KAGOME_ABELIAN is compatible. 

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
        assert ipess_tensors['T_d'].get_signature()==(-1,-1,-1),"Unexpected signature"
        assert ipess_tensors['T_u'].get_signature()==(-1,-1,-1),"Unexpected signature"
        assert ipess_tensors['B_a'].get_signature()==(-1,1,1),"Unexpected signature"
        assert ipess_tensors['B_b'].get_signature()==(-1,1,1),"Unexpected signature"
        assert ipess_tensors['B_c'].get_signature()==(-1,1,1),"Unexpected signature"
        self.ipess_tensors= ipess_tensors
        if build_sites:
            sites= self.build_onsite_tensors()
        else:
            sites= {}

        super().__init__(settings, sites, lX=1, lY=1, peps_args=peps_args,
                         global_args=global_args)

    def get_parameters(self):
        r"""
        :return: variational parameters of IPESS_KAGOME_GENERIC_ABELIAN
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        In this case member ``ipess_tensors``.
        """
        return list(self.ipess_tensors[ind].data for ind in self.ipess_tensors)

    def get_checkpoint(self):
        r"""
        :return: serializable representation of IPESS_KAGOME_GENERIC_ABELIAN state
        :rtype: dict

        Return dict containing serialized ``ipess_tensors`` (block-sparse) tensors. 
        The individual blocks are serialized into Numpy ndarrays. 
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return {ind: self.ipess_tensors[ind].save_to_dict() for ind in self.ipess_tensors}

    def load_checkpoint(self, checkpoint_file):
        r"""
        :param checkpoint_file: path to checkpoint file
        :type checkpoint_file: str or file object 

        Initializes IPESS_KAGOME_GENERIC_ABELIAN from checkpoint file. 
        """
        checkpoint= torch.load(checkpoint_file, map_location=self.device, weights_only=False)
        self.ipess_tensors= {ind: yastn.load_from_dict(config= self.engine, d=t_dict_repr) \
            for ind,t_dict_repr in checkpoint["parameters"].items()}
        for t in self.ipess_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()
        self.sync_precomputed()

    def to_dense(self, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :return: returns equivalent dense state with all on-site tensors in their dense 
                 representation on PyTorch backend.
        :rtype: IPESS_KAGOME_GENERIC_ABELIAN

        Create an IPESS_KAGOME_GENERIC state with all ``ipess_tensors`` as dense possesing 
        no explicit block structure (symmetry). This operations preserves gradients 
        on the returned dense state.
        """
        ipess_tensors_dense= {t_id: t.to_dense() for t_id,t in self.ipess_tensors.items()}
        state_dense= IPESS_KAGOME_GENERIC(ipess_tensors_dense,peps_args=peps_args,\
            global_args=global_args)
        return state_dense

    def build_onsite_tensors(self):
        r"""
        :return: elementary unit cell of underlying IPEPS
        :rtype: dict[tuple(int,int): yastn.Tensor]

        Build on-site tensor of corresponding IPEPS_KAGOME_ABELIAN be 
        performing following contraction of ``ipess_tensors``::

                 2(-7)          2(-6)                    (-)(-4)
                  \             /          rot. pi             |
             0(-3)==B_a        B_b==0(-2)  clockwise  (-)(-5)--\                     
                    \         /             =>                  \
                    1(3)     1(2)                                s0(-1)--s2(-3)--(-7)(+)
                     2(3)   1(2)                                 | / 
                       \   /                                     |/   <- DOWN_T
                        T_d                                      s1(-2)
                         |                                       |
                         0(1)                                   (-6)(+)
                         1(1)                       
                         |                 
                         B_c==0(-1)        
                         |
                         2(0)
                         0(0)  
                         |
                        T_u
                       /   \ 
                     1(-4)  2(-5)
        """
        A= yastn.ncon([self.ipess_tensors['T_u'], self.ipess_tensors['B_c'],\
            self.ipess_tensors['T_d'], self.ipess_tensors['B_b'], self.ipess_tensors['B_a']],\
            [[1,-3,-4], [-0,2,1], [2,3,4], [-1,3,-5], [-2,4,-6]])
        #    [[0,-4,-5], [-1,1,0], [1,2,3], [-2,2,-6], [-3,3,-7]]
        A= A.fuse_legs(axes=((0,1,2),3,4,5,6))
        A= A/A.norm(p='inf')
        sites= {(0, 0): A}
        return sites

    def add_noise(self, noise=0, peps_args=cfg.peps_args):
        r"""
        :param noise: magnitude of the noise
        :type noise: float
        :return: a copy of state with noisy ``ipess_tensors``. For default value of 
                 ``noise`` being zero ``self`` is returned. 
        :rtype: IPESS_KAGOME_GENERIC_ABELIAN

        Create a new state by adding random uniform noise with magnitude ``noise`` to all 
        copies of ``ipess_tensors``. The noise is added to all allowed blocks making up 
        the individual tensors. If noise is 0, returns ``self``.
        """
        if noise==0: return self
        ipess_tensors= {}
        for ind,t in self.ipess_tensors.items():
            ts, Ds= (l.t for l in t.get_legs(native=True)), (l.D for l in t.get_legs(native=True))
            t_noise= yastn.rand(config= t.config, s=t.s, n=t.n, t=ts, D=Ds, isdiag=t.isdiag)
            ipess_tensors[ind]= t + noise*t_noise
        return IPESS_KAGOME_GENERIC_ABELIAN(self.engine, ipess_tensors,\
                 peps_args=peps_args)

    def generate_weights(self):
        r"""
        :return: dictionary of weights on non-equivalent bonds of the iPESS ansatz
        :rtype: dict( str: yastn.Tensor )
         
        Generate a set of identity tensors, weights W, for each non-equivalent link 
        in the iPESS ansatz::

                2(d)            2(c)                    (-)a
                 \             /          rot. pi          |
            0(w)==B_a         B_b==0(v)   clockwise  (-)b--\                     
                   \         /             =>               \
                   1(l)     1(k)                            s0--s2--d(+)
                 W_down_a  W_down_b                          |  /
                    2(l)   1(k)                              | / 
                      \   /                                  |/   <- DOWN_T
                       T_d                                  s1
                        |                                    |
                        0(j)                                 c(+)
                      W_down_c
                        1(j)
                        |                 
                        B_c==0(u)        
                        |
                        2(i)
                       W_up_c
                        0(i)  
                        |
                       T_u
                      /   \ 
                    1(a)   2(b)
                  W_up_b   W_up_a
        """
        weights=dict()
        for w_id, ind1, ind2 in [ 
                ('lambda_up_c', ('T_u',0), ('B_c',2)), 
                ('lambda_up_a', ('T_u',2), ('B_a',2)),
                ('lambda_up_b', ('T_u',1), ('B_b',2)),
                ('lambda_dn_c', ('T_d',0), ('B_c',1)),
                ('lambda_dn_a', ('T_d',2), ('B_a',1)),
                ('lambda_up_b', ('T_u',1), ('B_b',1))
            ]:
            W= yastn.match_legs( tensors=[self.ipess_tensors[ind1[0]], self.ipess_tensors[ind2[0]]],\
                legs=[ ind1[1] , ind2[1] ], isdiag=True ).flip_signature()
            weights[w_id]= W
        return weights

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for t_id,t in self.ipess_tensors.items():
            print(f"{t_id}")
            print(f"{t}")
        print("")    
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"a{nid} {coord}: {site}")
        
        # show tiling of a square lattice
        coord_list = list(self.sites.keys())
        mx, my = 3*self.lX, 3*self.lY
        label_spacing = 1+int(math.log10(len(self.sites.keys())))
        for y in range(-my,my):
            if y == -my:
                print("y\\x ", end="")
                for x in range(-mx,mx):
                    print(str(x)+label_spacing*" "+" ", end="")
                print("")
            print(f"{y:+} ", end="")
            for x in range(-mx,mx):
                print(f"a{coord_list.index(self.vertexToSite((x,y)))} ", end="")
            print("")
        
        return ""

    def write_to_file(self, outputfile, tol=None, normalize=False):
        write_ipess_kagome_generic(self, outputfile, tol=tol, normalize=normalize)

# TODO verify global_args.device is compatible with device in settings
def read_ipess_kagome_generic(jsonfile, settings, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPESS_KAGOME_GENERIC_ABELIAN in json format
    :param settings: YAST configuration
    :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_GENERIC_ABELIAN

    Read IPESS_KAGOME_GENERIC_ABELIAN from file.
    """
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        ipess_tensors= OrderedDict()
        assert "ipess_tensors" in raw_state.keys(),"Missing \"ipess_tensors\"" 
        assert set(('T_u','T_d','B_a','B_b','B_c'))==set(list(raw_state["ipess_tensors"].keys())),\
            "missing some ipess tensors"
        for key,t in raw_state["ipess_tensors"].items():
            # depending on the "format", read the bare tensor
            assert "format" in t.keys(), "\"format\" not specified"
            if t["format"]=="abelian":
                X= read_json_abelian_tensor_legacy(t, settings)
                # move all tensors to desired dtype and device
                # TODO improve
                if X.is_complex() and X.yast_dtype in ["float64", "float32"]:
                    warnings.warn("Downcasting complex tensor to real", RuntimeWarning)
                ipess_tensors[key]= X.to(dtype=global_args.dtype, device=global_args.device)
            else:
                raise Exception("Unsupported format "+t["format"])

    state = IPESS_KAGOME_GENERIC_ABELIAN(settings, ipess_tensors, peps_args=peps_args, \
            global_args=global_args)
    return state

def write_ipess_kagome_generic(state, outputfile, tol=None, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction to write out in JSON format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, ipess tensors are normalized before writing
    :type state: IPESS_KAGOME_GENERIC_ABELIAN
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool

    Write IPESS_KAGOME_GENERIC_ABELIAN to file.
    """
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "ipess_tensors": {}})

    # write list of considered elementary tensors
    for key, t in state.ipess_tensors.items():
        tmp_t= t/t.norm(p='inf') if normalize else t
        json_state["ipess_tensors"][key]= serialize_abelian_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '), cls=NumPy_Encoder)