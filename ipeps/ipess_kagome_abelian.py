from collections import OrderedDict
from itertools import chain
import json
import math
import config as cfg
import yamps.yast as yast
import ipeps.ipeps_kagome_abelian as ipeps_kagome
from ipeps.tensor_io import *

class IPESS_KAGOME_GENERIC_ABELIAN(ipeps_kagome.IPEPS_KAGOME_ABELIAN):
    def __init__(self, settings, ipess_tensors,
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: yast symmetry module
        :param ipess_tensors: dictionary of five tensors, which make up Kagome iPESS
                              ansatz 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type ipess_tensors: dict(str, yast.Tensor)
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        iPESS ansatz for Kagome composes five tensors T_u, T_d, B_a, B_b, and B_c within 
        elementary unit cell into regular iPEPS. The B_* tensors hold physical degrees of 
        freedom, which reside on corners shared between different triangles, described by
        tensors T_u or T_d for up and down triangles respectively. 
        These tensors are passed in a dictionary with corresponding keys "T_u", "T_d",...
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
        iPEPS is compatible. 

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
        sites = self.build_onsite_tensors()

        super().__init__(settings, sites, lX=1, lY=1, peps_args=peps_args,
                         global_args=global_args)

    def get_parameters(self):
        return list(chain( *(self.ipess_tensors[ind].A.values() for ind in self.ipess_tensors)))

    def get_checkpoint(self):
        return {ind: self.ipess_tensors[ind].export_to_dict() for ind in self.ipess_tensors}

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, map_location=self.device)
        self.ipess_tensors= {ind: yast.import_from_dict(config= self.engine, d=t_dict_repr) \
            for ind,t_dict_repr in checkpoint["parameters"].items()}
        for t in self.ipess_tensors.values(): t.requires_grad_(False)
        self.sites = self.build_onsite_tensors()

    def build_onsite_tensors(self):
        r"""
        Build on-site tensor of corresponding iPEPS::

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
        A= yast.ncon([self.ipess_tensors['T_u'], self.ipess_tensors['B_c'],\
            self.ipess_tensors['T_d'], self.ipess_tensors['B_b'], self.ipess_tensors['B_a']],\
            [[1,-3,-4], [-0,2,1], [2,3,4], [-1,3,-5], [-2,4,-6]])
        #    [[0,-4,-5], [-1,1,0], [1,2,3], [-2,2,-6], [-3,3,-7]]
        #print(A)
        A= A.fuse_legs(axes=((0,1,2),3,4,5,6))
        A= A/A.norm(p='inf')
        sites= {(0, 0): A}
        return sites

    def add_noise(self, noise=0):
        r"""
        :param noise: magnitude of the noise
        :type noise: float
        :return: a copy of state with noisy ipess tensors. For default value of 
                 ``noise`` being zero ``self`` is returned. 
        :rtype: IPESS_KAGOME_GENERIC_ABELIAN

        Create a new state by adding random uniform noise with magnitude ``noise`` to all 
        copies of ipess tensors. The noise is added to all allowed blocks making up 
        the individual tensors.
        """
        if noise==0: return
        for ind,t in self.ipess_tensors.items():
            ts, Ds= t.get_leg_charges_and_dims(native=True)
            t_noise= yast.rand(config= t.config, s=t.s, n=t.n, t=ts, D=Ds, isdiag=t.isdiag)
            self.ipess_tensors[ind]= t + noise*t_noise
        self.sites = self.build_onsite_tensors()

    def write_to_file(self, outputfile, tol=None, normalize=False):
        write_ipess_kagome_generic(self, outputfile, tol=tol, normalize=normalize)

# TODO verify global_args.device is compatible with device in settings
def read_ipess_kagome_generic(jsonfile, settings, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPESS_KAGOME_GENERIC_ABELIAN in json format
    :param settings: yast symmetry module
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPESS_KAGOME_GENERIC_ABELIAN
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
                ipess_tensors[key]= X.to(global_args.device)
            else:
                raise Exception("Unsupported format "+t["format"])

    state = IPESS_KAGOME_GENERIC_ABELIAN(settings, ipess_tensors, peps_args=peps_args, \
            global_args=global_args)

    # check dtypes of all on-site tensors for newly created state
    assert (False not in [state.dtype==s.unique_dtype() for s in ipess_tensors.values()]),\
        "incompatible dtype among state and ipess tensors"
    
    return state

def write_ipess_kagome_generic(state, outputfile, tol=None, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, ipess tensors are normalized before writing
    :type state: IPESS_KAGOME_GENERIC_ABELIAN
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    """
    json_state = dict({"lX": state.lX, "lY": state.lY, \
        "ipess_tensors": {}})

    # write list of considered elementary tensors
    for key, t in state.ipess_tensors.items():
        tmp_t= t/t.norm(p='inf') if normalize else t
        json_state["ipess_tensors"][key]= serialize_abelian_tensor_legacy(tmp_t)

    with open(outputfile, 'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '), cls=NumPy_Encoder)