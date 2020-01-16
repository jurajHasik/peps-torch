import torch
import json
import itertools
import math
import config as cfg
from groups.pg import make_d2_symm
from ipeps import IPEPS
from ipeps import add_random_noise, read_ipeps, write_ipeps

class IPEPS_D2SYM(IPEPS):
    def __init__(self, parent_tensors, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--A--r  <=> A[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up
        Member ``vertexToSite`` is a mapping function from vertex on a square lattice
        passed in as tuple(x,y) to a corresponding tuple(x,y) within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            A_00= wfc.site(coord)

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
        applying rotating 
        """
        super(IPEPS, self).__init__()
        self.dtype = global_args.dtype
        self.device = global_args.device

        self.parent_tensors = parent_tensors
        self.sites = self.build_onsite_tensors()
        
        # infer the size of the cluster
        self.lX = 1
        self.lY = 2

        def vertexToSite(coord):
            x = coord[0]
            y = coord[1]
            return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
        self.vertexToSite = vertexToSite

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"A{nid} {coord}: {site.size()}")
        
        # show tiling of a square lattice
        # create dummy sites dict explicitly listing reflected site
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
                print(f"A{coord_list.index(self.vertexToSite((x,y)))} ", end="")
            print("")
        
        return ""

    def get_parameters(self):
        return [self.parent_tensors[k] for k in self.parent_tensors.keys()]

    def build_onsite_tensors(self):
        sites=dict()
        sites[(0,0)]=next(iter(self.parent_tensors.values()))
        sites[(0,1)]=sites[(0,0)].permute(0,3,2,1,4).contiguous()
        #sites[(0,1)]=sites[(0,0)].permute(0,3,4,1,2).contiguous()
        return sites

    def add_noise(self,noise):
        for coord in self.parent_tensors.keys():
            rand_t= torch.rand( self.parent_tensors[coord].size(), dtype=self.dtype, device=self.device)
            temp_t= self.parent_tensors[coord] + noise * rand_t
            temp_t= make_d2_symm(temp_t) 
            self.parent_tensors[coord] = temp_t/torch.max(torch.abs(temp_t))

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[1:]]

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_d2(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

def extend_bond_dim(state, new_d):
    r"""
    :param state: wavefunction to modify
    :param new_d: new enlarged auxiliary bond dimension
    :type state: IPEPS
    :type new_d: int
    :return: wavefunction with enlarged auxiliary bond dimensions
    :rtype: IPEPS

    Take IPEPS and enlarge all auxiliary bond dimensions of all on-site tensors up to 
    size ``new_d``
    """
    new_state = state
    for coord,t in new_state.parent_tensors.items():
        dims = t.size()
        size_check = [new_d >= d for d in dims[1:]]
        if False in size_check:
            raise ValueError("Desired dimension is smaller than following aux dimensions: "+str(size_check))

        new_t = torch.zeros((dims[0],new_d,new_d,new_d,new_d), dtype=state.dtype, device=state.device)
        new_t[:,:dims[1],:dims[2],:dims[3],:dims[4]] = t
        new_state.parent_tensors[coord] = new_t
    new_state.sites= new_state.build_onsite_tensors()
    return new_state

def read_ipeps_d2(jsonfile, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    
    state= read_ipeps(jsonfile, aux_seq=aux_seq, peps_args=peps_args,\
        global_args=global_args)

    if len(state.sites.items())>1:
        raise ValueError("Not a valid 1-site D2 symmetric iPEPS")

    return IPEPS_D2SYM(state.sites, peps_args=peps_args,global_args=global_args)

def write_ipeps_d2(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    temp_ipeps = IPEPS(state.parent_tensors)
    write_ipeps(temp_ipeps, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)