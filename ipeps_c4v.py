import torch
import ipeps
from groups.c4v import make_c4v_symm 
import config as cfg

class IPEPS_C4V(ipeps.IPEPS):
    def __init__(self, site, peps_args=cfg.peps_args, global_args=cfg.global_args):
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
            l--A--r  <=> A[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        """
        assert isinstance(site,torch.Tensor), "site is not a torch.Tensor"
        super().__init__(dict({(0,0): site}),peps_args=peps_args,\
            global_args=global_args)

    def add_noise(self,noise):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        for coord in self.sites.keys():
            rand_t = torch.rand( self.sites[coord].size(), dtype=self.dtype, device=self.device)
            self.sites[coord] = make_c4v_symm(self.sites[coord] + noise * rand_t)

    def write_to_file(self,outputfile,**kwargs):
        # symmetrize before writing out
        tmp_t= make_c4v_symm(self.site((0,0)))
        tmp_state= IPEPS_C4V(tmp_t)
        ipeps.write_ipeps(tmp_state,outputfile,**kwargs)

def to_ipeps_c4v(state):
    assert len(state.sites.items())==1, "state has more than a single on-site tensor"
    return IPEPS_C4V(next(iter(state.sites.values())))

def read_ipeps_c4v(jsonfile, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    state= ipeps.read_ipeps(jsonfile, aux_seq=aux_seq, peps_args=peps_args,\
        global_args=global_args)
    return to_ipeps_c4v(state)
