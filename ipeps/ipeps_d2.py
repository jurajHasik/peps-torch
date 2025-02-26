import torch
import ipeps.ipeps as ipeps
from groups.pg import make_d2_symm
import config as cfg

class IPEPS_D2SYM(ipeps.IPEPS):
    def __init__(self, site=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param site: on-site tensor
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type site: torch.tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS
        
        We use a single tensor A, with three types of auxilliary indices::
        
                 s         s
               :/        :/
            l--A--r = r--A--l
               |         |
        
        The "up" index is assumed to form "weak" bonds on the coupled ladders,
        the "down" index on the other hand forms the strong bonds on the rungs of the ladder.
        Finally, the horizontal indices are assumed to be the same A[s,u,l,d,r]=A[s,u,r,d,l].
        
        The 1x2 unit cell is effectively given by::

              :
            --A--
              |    <- symmetric under reflection
            --A--
              :

        The bipartite (AFM) nature is added at the level of observables by
        applying rotation on the physical indices on every (say) sublattice-B site.
        """
        if site is not None:
            self.parent_site= site
            sites= self.build_onsite_tensors()
        else: 
            sites= dict()

        def vertexToSite(coord):
            x = coord[0]
            y = coord[1]
            return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
        
        super().__init__(sites, vertexToSite=vertexToSite, lX=1, lY=2, 
            peps_args=peps_args, global_args=global_args)

    def get_parameters(self):
        return (self.parent_site,)

    def get_checkpoint(self):
        return self.parent_site

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file, weights_only=False)
        self.parent_site= checkpoint["parameters"]
        self.parent_site.requires_grad_(False)
        self.sites= self.build_onsite_tensors()

    def build_onsite_tensors(self):
        sites=dict()
        sites[(0,0)]=self.parent_site
        sites[(0,1)]=sites[(0,0)].permute(0,3,2,1,4).contiguous()
        #sites[(0,1)]=sites[(0,0)].permute(0,3,4,1,2).contiguous()
        return sites

    def add_noise(self,noise):
        rand_t= torch.rand( self.parent_site.size(), dtype=self.dtype, device=self.device)
        temp_t= self.parent_site + noise * rand_t
        temp_t= make_d2_symm(temp_t) 
        self.parent_site= temp_t/torch.max(torch.abs(temp_t))
        self.sites= self.build_onsite_tensors()

    def get_aux_bond_dims(self):
        return self.parent_site.size()[1:]

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_d2(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

def extend_bond_dim(state, new_d):
    r"""
    :param state: wavefunction to modify
    :param new_d: new enlarged auxiliary bond dimension
    :type state: IPEPS_D2SYM
    :type new_d: int
    :return: wavefunction with enlarged auxiliary bond dimensions
    :rtype: IPEPS_D2SYM

    Take IPEPS_D2SYM and enlarge all auxiliary bond dimensions of parent tensor up to 
    size ``new_d``
    """
    dims = state.parent_site.size()
    size_check = [new_d >= d for d in dims[1:]]
    if False in size_check:
        raise ValueError("Desired dimension is smaller than following aux dimensions: "+str(size_check))

    new_t = torch.zeros((dims[0],new_d,new_d,new_d,new_d), dtype=state.dtype, device=state.device)
    new_t[:,:dims[1],:dims[2],:dims[3],:dims[4]] = state.parent_site
    new_state= IPEPS_D2SYM(new_t)
    return new_state

def read_ipeps_d2(jsonfile, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    state= ipeps.read_ipeps(jsonfile, aux_seq=aux_seq, peps_args=peps_args,\
        global_args=global_args)
    # assume two-site ipeps state and take site at (0,0) as the parent_site 
    assert len(state.sites.items())==2 and state.lX==1 and state.lY==2, \
        "Not a valid IPEPS_D2SYM"
    return IPEPS_D2SYM(state.site((0,0)),peps_args=peps_args,global_args=global_args)

def write_ipeps_d2(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    # force building of on-site tensors
    tmp_state= IPEPS_D2SYM(state.parent_site)
    ipeps.write_ipeps(tmp_state, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)
