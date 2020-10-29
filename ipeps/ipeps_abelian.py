from collections import OrderedDict
from itertools import chain
import json
import itertools
import math
import warnings
try:
    import torch
except ImportError as e:
    warnings.warn("torch not available", Warning)
import config as cfg
import yamps.tensor as TA
from ipeps.tensor_io import *

class IPEPS_ABELIAN():
    
    _REF_S_DIRS=(-1,-1,-1,1,1)

    def __init__(self, settings, sites, vertexToSite=None, lX=None, lY=None, 
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: TODO
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type settings: TODO
        :type sites: dict[tuple(int,int) : yamps.tensor.Tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

           (-1)u (-1)s 
               |/ 
        (-1)l--a--(+1)r  <=> a[s,u,l,d,r] with reference symmetry signature [-1,-1,-1,1,1]
               |
           (+1)d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS_ABELIAN object ``wfc`` at vertex (x,y) is conveniently accessed 
        through the member function ``site``, which internally uses ``vertexToSite`` mapping::
            
            coord= (0,0)
            a_00= wfc.site(coord)

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites``, various tilings of a square lattice can be achieved:: 
            
            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): a}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   a  a a a a
            # -1   a  a a a a
            #  0   a  a a a a
            #  1   a  a a a a
            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): a, (1,0): b}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.
        """
        self.engine= settings
        self.backend= settings.back
        assert global_args.dtype==settings.dtype, "global_args.dtype "+global_args.dtype\
            +" settings.dtype "+settings.dtype
        self.dtype= settings.dtype
        self.device= global_args.device
        self.nsym = settings.nsym
        self.sym= settings.sym

        self.sites= OrderedDict(sites)
        
        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if lX is None or lY is None:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        else:
            self.lX = lX
            self.lY = lY

        if vertexToSite is not None:
            self.vertexToSite = vertexToSite
        else:
            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*self.lX)%self.lX, (y + abs(y)*self.lY)%self.lY )
            self.vertexToSite = vertexToSite

    def site(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: on-site tensor corresponding to the vertex (x,y)
        :rtype: yamps.tensor.Tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def to(self, device):
        r"""
        :param device: device identifier
        :type device: str
        :return: returns a copy of the state on ``device``. If the state
                 already resides on `device` returns ``self``.
        :rtype: IPEPS_ABELIAN

        Move the entire state to ``device``.        
        """
        if device==self.device: return self
        sites= {ind: t.to(device) for ind,t in self.sites.items()}
        state= IPEPS_ABELIAN(self.engine, sites, self.vertexToSite, 
            lX=self.lX, lY=self.lY)
        return state

    def to_dense(self):
        r"""
        :return: returns a copy of the state with all on-site tensors in their dense 
                 representation. If the state already has just dense on-site tensors 
                 returns ``self``.
        :rtype: IPEPS_ABELIAN

        Create a copy of state with all on-site tensors as dense possesing no explicit
        block structure (symmetry). This operations preserves gradients on returned
        dense state.
        """
        if sites.nsym==0: return self
        sites_dense= {ind: t.to_dense() for ind,t in self.sites.items()}
        settings_dense= next(iter(sites.values())).conf
        state_dense= IPEPS_ABELIAN(settings_dense, sites_dense, \
            vertexToSite=self.vertexToSite, lX=self.lX, lY=self.lY)
        return state_dense

    def get_parameters(self):
        return list(chain( *(self.sites[ind].A.values() for ind in self.sites)))

    def get_checkpoint(self):
        r"""
        :return: serializable (pickle-able) representation of IPEPS_ABELIAN state
        :rtype: dict

        Return dict containing serialized on-site (block-sparse) tensors. The individual
        blocks are serialized into Numpy ndarrays
        """
        return {ind: self.sites[ind].to_dict() for ind in self.sites}

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.sites= {ind: TA.from_dict(settings= self.engine, d=t_dict_repr) \
            for ind,t_dict_repr in checkpoint["parameters"].items()}

    def write_to_file(self, outputfile, tol=None, normalize=False):
        write_ipeps(self, outputfile, tol=tol, normalize=normalize)

    # NOTE what about non-initialized blocks, which are however allowed by the symmetry ?
    def add_noise(self, noise=0):
        r"""
        :param noise: magnitude of the noise
        :type noise: float
        :return: a copy of state with noisy on-site tensors. For default value of 
                 ``noise`` being zero ``self`` is returned. 
        :rtype: IPEPS_ABELIAN

        Create a new state by adding random uniform noise with magnitude ``noise`` to all 
        copies of on-site tensors. The noise is added to all blocks making up the individual 
        on-site tensors.
        """
        if noise==0: return self
        sites= {}
        for ind,t in self.sites.items():
            t_data, D_data= t.get_tD()
            sites[ind]= t + noise * t.rand(s=t.s, n=t.n, t=t_data, D=D_data, isdiag=t.isdiag)
        state= IPEPS_ABELIAN(self.engine, sites, self.vertexToSite, 
            lX=self.lX, lY=self.lY)
        return state

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
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

def read_ipeps(jsonfile, settings, vertexToSite=None, \
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_ABELIAN in json format
    :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                         into a vertex within elementary unit cell
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_ABELIAN
    

    A simple PBC ``vertexToSite`` function is used by default
    """
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # Loop over non-equivalent tensor,site pairs in the unit cell
        for ts in raw_state["map"]:
            coord = (ts["x"],ts["y"])

            # find the corresponding tensor (and its elements) 
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["sites"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"") 

            # depending on the "format", read the bare tensor
            if "format" in t.keys():
                if t["format"]=="abelian":
                    X= read_json_abelian_tensor_legacy(t, settings)
                else:
                    raise Exception("Unsupported format "+t["format"])
            else:
                raise Exception(f"Missing key \"format\"")

            sites[coord]= X

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        # check for legacy keys
        lX = raw_state["sizeM"] if "sizeM" in raw_state else raw_state["lX"]
        lY = raw_state["sizeN"] if "sizeN" in raw_state else raw_state["lY"]

    if vertexToSite == None:
        def vertexToSite(coord):
            x = coord[0]
            y = coord[1]
            return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

        state = IPEPS_ABELIAN(settings, sites, vertexToSite, lX=lX, lY=lY, \
            peps_args=peps_args, global_args=global_args)
    else:
        state = IPEPS_ABELIAN(settings, sites, vertexToSite, lX=lX, lY=lY, \
            peps_args=peps_args, global_args=global_args)

    # check dtypes of all on-site tensors for newly created state
    assert (False not in [state.dtype==s.dtype for s in sites.values()]), \
        "incompatible dtype among state and on-site tensors"

    # move to desired device and return
    return state.to(global_args.device)

def write_ipeps(state, outputfile, tol=None, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS_ABELIAN
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    """
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": []})
    
    site_ids=[]
    site_map=[]
    for nid,coord,site in [(t[0], *t[1]) for t in enumerate(state.sites.items())]:
        if normalize:
            site= site/site.max_abs()
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        
        json_tensor= serialize_abelian_tensor_legacy(site)

        json_tensor["siteId"]=site_ids[-1]
        json_state["sites"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '), cls=NumPy_Encoder)