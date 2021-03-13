import torch
import ipeps.ipeps as ipeps
from groups.pg import make_c4v_symm 
import config as cfg

class IPEPS_C4V(ipeps.IPEPS):
    def __init__(self, site=None, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param site: on-site tensor
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type site: torch.tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        The index-position convetion for on-site tensor is defined as follows::

               u s 
               |/ 
            l--A--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        """
        if site is not None:
            assert isinstance(site,torch.Tensor), "site is not a torch.Tensor"
            sites= {(0,0): site}
        else:
            sites= dict()
        super().__init__(sites, lX=1, lY=1, peps_args=peps_args,\
            global_args=global_args)

    def site(self,coord=None):
        return self.sites[(0,0)]

    def add_noise(self,noise,symmetrize=False):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to on-site tensor
        """
        rand_t = torch.rand( self.site().size(), dtype=self.dtype, device=self.device)
        self.sites[(0,0)]= self.site() + noise * rand_t
        if symmetrize:
            self.sites[(0,0)]= make_c4v_symm(self.site())

    def write_to_file(self,outputfile,symmetrize=True,**kwargs):
        # symmetrize before writing out
        tmp_state= to_ipeps_c4v(self) if symmetrize else self
        ipeps.write_ipeps(tmp_state, outputfile,**kwargs)

def extend_bond_dim(state, new_d):
    return ipeps.extend_bond_dim(state, new_d)

def to_ipeps_c4v(state, normalize=False):
    #TODO other classes of C4v-symmetric ansatz ?
    # we choose A1 irrep, in principle, other choices are possible (A2, B1, ...)
    assert len(state.sites.items())==1, "state has more than a single on-site tensor"
    A= next(iter(state.sites.values()))
    
    if A.is_complex():
        A= make_c4v_symm(A.real) + make_c4v_symm(A.imag, irreps=["A2"]) * 1.0j
    else:
        A= make_c4v_symm(A)
    
    if normalize: A= A/A.norm()
    return IPEPS_C4V(A)

def read_ipeps_c4v(jsonfile, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_C4V in json format
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_C4V
    
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
    state= ipeps.read_ipeps(jsonfile, aux_seq=aux_seq, peps_args=peps_args,\
        global_args=global_args)
    assert len(state.sites.items())==1, "state has more than a single on-site tensor"
    return IPEPS_C4V(next(iter(state.sites.values())))