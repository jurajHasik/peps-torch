from collections import OrderedDict
from itertools import chain
import json
import math
import warnings
try:
    import torch
    from ipeps.ipeps import IPEPS
except ImportError as e:
    warnings.warn("torch not available", Warning)
import config as cfg
import yastn.yastn as yastn
from ipeps.tensor_io import *
import logging
log = logging.getLogger(__name__)


def _fused_open_dl_site(a, fusion_level="full"):
    r"""
    Convenience function to construct open double-layer on-site tensor by taking outer 
    product with its complex conjugate. The outer product is followed by fusion of auxiliary 
    indices

         0 1 
          \|                      (16)(05)           1  0
        2--a--4                    || //             | /
           |     1->6      = (27)==aa*==(49) =>  2--(aa*)--4
           3     |                 ||                |
              2--a*--4->9         (38)               3
            ->7  |\0->5
                 3->8

    Such tensors serve as building blocks of reduced density matrices together with environment tensors.
    The auxiliary indices of `ket` and `bra` layers are fused into double-layer auxiliary in that order.
    If the physical indices are fused, the `ket` physical index precedes `bra`.

    Parameters
    ----------
    a: Tensor
        rank-5 on-site tensor with legs ordered as physical and auxiliary up, left, down, right

    fusion_level: str
        'full' `bra` and `ket` physical indices are fused together resulting in rank-5 tensor
        'basic' `bra` and `ket` physical indices are left unfused resulting in rank-6 tensor
        with index order physical `ket`, physical `bra`, auxiliary up, left, down, right     

    Returns
    -------
    tensor : Tensor
    """
    A= a.tensordot(a, axes=([],[]), conj=(0,1))
    if fusion_level=="full":
        A= A.fuse_legs( axes=((0,5),(1,6),(2,7),(3,8),(4,9)) )
    elif fusion_level=="basic":
        A= A.fuse_legs( axes=(0,5,(1,6),(2,7),(3,8),(4,9)) )
    else:
        raise RuntimeError("Unsupported fusion_level option "+fusion_level)
    return A

def _fused_dl_site(a, a_bra=None):
    if a_bra is None:
        A= a.tensordot(a, axes=([0],[0]), conj=(0,1))
    else:
        A= a.tensordot(a_bra, axes=([0],[0]))
    A= A.fuse_legs( axes=((0,4),(1,5),(2,6),(3,7)) )
    return A


class IPEPS_ABELIAN():
    
    _REF_S_DIRS=(-1,-1,-1,1,1)

    def __init__(self, settings, sites, vertexToSite=None, lX=None, lY=None, 
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param build_open_dl: build complementary :class:`IPEPS_ABELIAN` with with 
                              open double-layer on-site tensors
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :type sites: dict[tuple(int,int) : yastn.Tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type build_open_dl: bool
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
            wfc= IPEPS_ABELIAN(sites,vertexToSite)
        
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
            wfc= IPEPS_ABELIAN(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  b a b a
            # -1   B  a b a b
            #  0   A  b a b a
            #  1   B  a b a b
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): a, (1,0): b, (2,0): c, (0,1): d, (1,1): e, (2,1): f}
            wfc= IPEPS_ABELIAN(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   b  c a b c
            # -1   e  f d e f
            #  0   b  c a b c
            #  1   e  f d e f

        where in the last example a default setting for ``vertexToSite`` is used, which
        maps square lattice into elementary unit cell of size ``lX`` x ``lY`` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.

        Performance-wise, it is favourable to construct complementary iPEPS formed by open
        double-layer on-site tensors. Such tensors are repeatedly used as building blocks 
        of reduced density matrices when computing observables. If `build_open_dl=True` they 
        are pre-computed and accessible through member `sites_dl_open` or `site_dl_open(coord)`
        convenience function. 

        .. note::

            in case of differentiation through reduced density matrix construction, the 
            `sites_dl_open` computation must be a member of computation graph for correct gradients.
            It can be explicitly recomputed by invoking `build_sites_dl_open()`.

        """
        self.engine= settings
        assert global_args.dtype==settings.default_dtype, "global_args.dtype "+global_args.dtype\
            +" settings.default_dtype "+settings.default_dtype
        self.dtype= settings.default_dtype
        self.device= global_args.device
        self.nsym = settings.sym.NSYM
        self.sym= settings.sym.SYM_ID

        self.sites= OrderedDict(sites)
        # precomputation of (fused) double-layer tensors
        self.build_dl= peps_args.build_dl
        self.build_dl_open= peps_args.build_dl_open
        self.sites_dl= None
        self.sites_dl_open= None
        self.sync_precomputed()
        
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

    def build_sites_dl_open(self,fusion_level="full"):
        """
        If :py:attr:`config.PEPSARGS.build_dl`, build complementary open on-site double-layer iPEPS.

        :param fusion_level: see `_fused_open_dl_site()`
        :type fusion_level: str
        """
        self.sites_dl_open= OrderedDict(
            { coord: _fused_open_dl_site(site_t, fusion_level=fusion_level) for coord, site_t in self.sites.items() }
        )

    def build_sites_dl(self):
        """
        If :py:attr:`config.PEPSARGS.build_dl_open`, build complementary on-site double-layer iPEPS.
        """
        self.sites_dl= OrderedDict(
            { coord: _fused_dl_site(site_t) for coord, site_t in self.sites.items() }
        )

    def sync_precomputed(self):
        r"""
        Force recomputation of double-layer and open double-layer on-site tensors
        if corresponding options :py:attr:`config.PEPSARGS.build_dl` and 
        :py:attr:`config.PEPSARGS.build_dl_open` are ``True``.

        .. note::
            In active autograd regions, it might be necessary to force recomputation
            if the corresponding double-layer tensors are to be part of computational
            graph and hence differentiated. 

        """
        if self.build_dl: self.build_sites_dl()
        if self.build_dl_open: self.build_sites_dl_open()

    def normalize_(self):
        for c in self.sites.keys():
            self.sites[c]= self.sites[c]/self.sites[c].norm(p='inf')
        self.sync_precomputed()

    def site(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: on-site tensor corresponding to the vertex (x,y)
        :rtype: yastn.Tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def site_dl(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: double-layer on-site tensor corresponding to the vertex (x,y)
        :rtype: yastn.Tensor

        If :py:attr:`config.PEPSARGS.build_dl`, then precomputed double-layer on-site
        tensor is returned. Otherwise, Exception is raised. 
        """
        assert not self.sites_dl is None, "sites_dl not initialized"
        return self.sites_dl[self.vertexToSite(coord)]

    def site_dl_open(self, coord):
        """
        :param coord: tuple (x,y) specifying vertex on a square lattice
        :type coord: tuple(int,int)
        :return: open double-layer on-site tensor corresponding to the vertex (x,y)
        :rtype: yastn.Tensor

        If :py:attr:`config.PEPSARGS.build_dl_open`, then precomputed open double-layer 
        on-site tensor is returned. Otherwise, Exception is raised.
        """
        assert not self.sites_dl_open is None, "sites_dl_open not initialized"
        return self.sites_dl_open[self.vertexToSite(coord)]

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

    def to_dense(self, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :return: returns equivalent dense state with all on-site tensors in their dense 
                 representation on PyTorch backend.
        :rtype: IPEPS

        Create an IPEPS state with all on-site tensors as dense possesing no explicit
        block structure (symmetry). This operations preserves gradients on returned
        dense state.
        """
        sites_dense= {ind: t.to_dense() for ind,t in self.sites.items()}
        state_dense= IPEPS(sites_dense, vertexToSite=self.vertexToSite, \
            lX=self.lX, lY=self.lY,peps_args=peps_args, global_args=global_args)
        return state_dense

    def get_parameters(self):
        r"""
        :return: variational parameters of iPEPS
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        """
        return list(self.sites[ind].data for ind in self.sites)

    def get_checkpoint(self):
        r"""
        :return: serializable representation of IPEPS_ABELIAN state
        :rtype: dict

        Return dict containing serialized on-site (block-sparse) tensors. The individual
        blocks are serialized into Numpy ndarrays. This function is called by optimizer 
        to create checkpoints during the optimization process.
        """
        return {ind: self.sites[ind].save_to_dict() for ind in self.sites}

    def load_checkpoint(self, checkpoint_file):
        r"""
        :param checkpoint_file: path to checkpoint file
        :type checkpoint_file: str or file object 

        Initializes IPEPS_ABELIAN from checkpoint file. 

        .. note:: 

            The `vertexToSite` mapping function is not a part of checkpoint and must 
            be provided either when instantiating IPEPS_ABELIAN or afterwards. 
        """
        checkpoint= torch.load(checkpoint_file, map_location=self.device, weights_only=False) 
        self.sites= {ind: yastn.load_from_dict(config= self.engine, d=t_dict_repr) \
            for ind,t_dict_repr in checkpoint["parameters"].items()}
        for site_t in self.sites.values(): site_t.requires_grad_(False)
        self.sync_precomputed()

    def write_to_file(self, outputfile, tol=None, normalize=False):
        """
        Writes state to file. See :meth:`write_ipeps`.
        """
        write_ipeps(self, outputfile, tol=tol, normalize=normalize)

    # TODO what about non-initialized blocks, which are however allowed by the symmetry ?
    def add_noise(self, noise=0, peps_args=cfg.peps_args):
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
            legs= t.get_legs(native=True)
            ts, Ds= (l.t for l in legs), (l.D for l in legs)
            t_noise= yastn.rand(config=t.config, s=t.s, n=t.n, t=ts, D=Ds, isdiag=t.isdiag)
            sites[ind]= t + noise * t_noise
        state= IPEPS_ABELIAN(self.engine, sites, self.vertexToSite, 
            lX=self.lX, lY=self.lY, peps_args=peps_args)
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
    :param settings: YAST configuration
    :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                         into a vertex within elementary unit cell
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
    :type vertexToSite: function(tuple(int,int))->tuple(int,int)
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_ABELIAN
    
    Read state in JSON format from file.

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
                warnings.warn("\"format\" not specified. Assuming dense tensor", Warning)
                t["charges"]=[]
                tmp_t= {"blocks": [t]}
                tmp_t["format"]="abelian"
                tmp_t["dtype"]= t["dtype"]
                tmp_t["nsym"]=0
                tmp_t["symmetry"]=[]
                tmp_t["signature"]= IPEPS_ABELIAN._REF_S_DIRS
                tmp_t["n"]=0
                tmp_t["isdiag"]=False
                tmp_t["rank"]= len(t["dims"])
                X= read_json_abelian_tensor_legacy(tmp_t, settings)

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
    assert (False not in [state.dtype==s.yast_dtype for s in sites.values()]), \
        "incompatible dtype among state and on-site tensors"

    # move to desired device and return
    return state.to(global_args.device)

def write_ipeps(state, outputfile, tol=None, normalize=False, metadata=None,\
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

    Write state to file.
    """
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": [], "metadata": metadata})
    
    site_ids=[]
    site_map=[]
    for nid,coord,site in [(t[0], *t[1]) for t in enumerate(state.sites.items())]:
        if normalize:
            site= site/site.norm(p='inf')
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        
        json_tensor= serialize_abelian_tensor_legacy(site)

        json_tensor["siteId"]=site_ids[-1]
        json_state["sites"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '), cls=NumPy_Encoder)


class IPEPS_ABELIAN_WEIGHTED(IPEPS_ABELIAN):

    # TODO validate weights
    def __init__(self, state=None, config=None, sites=None, weights=None, \
        vertexToSite=None, lX=None, lY=None, 
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param config: YAST configuration
        :param sites: map from elementary unit cell to on-site tensors
        :param weights: map from edges within unit cell to weight tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type config: NamedTuple or SimpleNamespace (TODO link to definition)
        :type sites: dict[tuple(int,int) : yastn.Tensor]
        :type weights: dict[tuple(tuple(int,int), tuple(int,int)) : yastn.Tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        IPEPS_ABELIAN_WEIGHTED augments basic IPEPS_ABELIAN with a tensor on each bond
        within elementary unit cell. In case of diagonal and positive semi-definite tensors, these
        are called weights. Such augmented ansatz provides basic structure for iTEBD algorithms
        such as Simple Update.

        The keys of `weights` dictionary index tensors by tuple of `(coord, dxy)` where
        `coord` specifies site within elementary unit cell and `(dxy)` is a directional vector specifying
        up, left, down, or right bond of that site as `(0,-1)`, `(-1,0)`, `(0,1)` or `(1,0)` respectively. 
        Thus the `weights` is not injective dictionary, instead keys (coord,dxy) and (coord+dxy,-dxy)
        should index identical tensor.
        """
        if state:
            config=state.engine
            sites=state.sites
            vertexToSite=state.vertexToSite
            lX=state.lX
            lY=state.lY
        elif sites:
            assert vertexToSite or (lX and lY),"vertexToSite or lX,lY has to be provided"
            assert len(sites)>0 or config,"Either non-empty sites or config has to be provided"
            config= next(iter(sites.values())).config if config is None else config    
        else:
            raise RuntimeError("Either state or sites have to be provided")
        super().__init__(config, sites, vertexToSite=vertexToSite, lX=lX, lY=lY, 
            peps_args=peps_args, global_args=global_args)
        self.weights= OrderedDict(weights) if weights else self.generate_weights()

    def absorb_weights(self, peps_args=cfg.peps_args, 
        global_args=cfg.global_args):
        r"""
        :return: regular IPEPS_ABELIAN obtained by symmetricaly absorbing weights of 
                 IPEPS_ABELIAN_WEIGHTED into its on-site tensors
        :rtype: IPEPS_ABELIAN

        Reduce weighted iPEPS to regular iPEPS by splitting its weights symmetrically 
        as `W = \sqrt(W)\sqrt(W)` and absorbing them into on-site tensors::

                    \sqrt(W)         
                      |/s             |/s
            \sqrt(W)--a--\sqrt(W) = --a'--
                      |               |
                    \sqrt(W)

        .. note:: 
            assumes weight tensors are diagonal and positive semi-definite
        """
        dxy_w_to_ind= OrderedDict({(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4})
        full_dxy=set(dxy_w_to_ind.keys())

        a_sites=dict()
        for coord in self.sites.keys():
            A= self.site(coord)
            # 0,[1--0,1->4],2->1,3->2,4->3
            # 0,[1--0,1->4],2->1,3->2,4->3
            # ...
            for dxy,ind in dxy_w_to_ind.items():
                w= self.weight((coord, dxy)).sqrt()
                _match_diag_signature= 1 if -w.get_signature()[1]==A.get_signature()[1] else 0
                A= A.tensordot(w, ([1],[_match_diag_signature]))
            a_sites[coord]= A

        return IPEPS_ABELIAN(self.engine, a_sites, vertexToSite=self.vertexToSite,\
            lX=self.lX, lY=self.lY, peps_args=peps_args, 
            global_args=global_args)

    def weight(self, weight_id):
        """
        :param weight_id: tuple with (x,y) coords specifying vertex on a square lattice
                          and tuple with (dx,dy) coords specifying on of the directions
                          (0,-1), (-1,0), (0,1), (1,0) corresponding to up, left, down, and
                          right respectively.
        :type weight_id: tuple(tuple(int,int), tuple(int,int))
        :return: diagonal weight tensor
        :rtype: yastn.Tensor
        """
        xy_site, dxy= weight_id
        assert dxy in [(0,-1), (-1,0), (0,1), (1,0)],"invalid direction"
        return self.weights[ (self.vertexToSite(xy_site), dxy) ]

    def generate_weights(self):
        #   
        #       w0         w2
        # w4--(0,0)--w5--(1,0)--[w4]
        #       w1         w3
        # w6--(0,1)--w7--(1,1)--[w6]
        #      [w0]       [w2]
        def neg_(dxy): return (-dxy[0],-dxy[1])
        def add_(coord,dxy): return (coord[0]+dxy[0],coord[1]+dxy[1])
        dxy_w_to_ind= dict({(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4})
        weights=dict()
        for coord in self.sites.keys():
            for dxy,ind in dxy_w_to_ind.items():
                # generate weight_id and reverse weight_id
                # (coord,dxy) identifies the same weight as (coord+dxy,-dxy) 
                w_id= (coord, dxy)
                w_rid= (self.vertexToSite(add_(coord,dxy)), neg_(dxy))

                if not w_id in weights.keys() and not w_rid in weights.keys():
                    W= yastn.eye( config=self.engine, legs=(
                        self.site(w_id[0]).get_legs(dxy_w_to_ind[w_id[1]]),
                        self.site(w_rid[0]).get_legs(dxy_w_to_ind[w_rid[1]])
                        ))
                    weights[w_id]= W
                    weights[w_rid]= W
        return weights

    def gauge(self,peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        """
        def neg_(dxy): return (-dxy[0],-dxy[1])
        def add_(coord,dxy): return (coord[0]+dxy[0],coord[1]+dxy[1])
        dxy_w_to_ind= dict({(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4})
        expr_ws= {(0,-1): 'um', (-1,0): 'ln', (0,1): 'do', (1,0): 'rp'}

        def _get_dl_gauges(coord,direction,sites,weights):
            coord= self.vertexToSite(coord)
            A= sites[coord]
            ds_to_contract= set(dxy_w_to_ind.keys())-set((direction,))
            
            #
            #         /w^2
            #  ------A^+--w^2
            #    w^2/|/   |
            #  -----|A-----
            #       /

            # get correct order for signatures to match
            #
            # A--? 0--w
            oi= { d: 0 if A.get_signature()[dxy_w_to_ind[d]]==-weights[(coord,d)].get_signature()[0] else 1 \
                for d in ds_to_contract }
            expr= 'suldr,smnop,'+','.join([expr_ws[d] if oi[d]==0 else expr_ws[d][::-1] for d in ds_to_contract])\
                +f'->{expr_ws[direction]}'
            order= ''.join([expr_ws[d][0] for d in ds_to_contract]+[expr_ws[d][1] for d in ds_to_contract])+'s'
            a= yastn.einsum(expr,\
                A,A.conj(),*( weights[(coord,d)]**2 for d in ds_to_contract),order=order)

            # diagonalize, since a is hermitian and positive. Force ordering in descending magnitude
            # n--    (-s)n(1)--U*
            #    a =           sqrt(D)^2
            # l--     (s)l(0)--U
            D,U = yastn.linalg.eigh((-1.)*a/a.norm(p='inf'), axes=(0,1), sU=-a.get_signature()[0])
            D= -1.*D
            #
            # (s)0--U--1(sU) (-sU)0--D--1(sU) = (s)0--X--1(sU=-s)
            X= U.tensordot(D.sqrt(),([1],[0]))
            D_invsqrt= D.rsqrt(cutoff=D.norm(p='inf')*1.0e-14)
            # (s)0--UD_invsqrt--1(sU) -> (sU)0--Xinv--1(s) -> (-sU=s)0--Xinv--1(-s)
            Xinv= (U.tensordot(D_invsqrt,([1],[0]))).transpose((1,0)).conj()
            return X, Xinv

        def _update_weights_and_sites(sites,weights,Xs):
            # associate pair X,Y to each unique weight and update it
            new_weights, Us= dict(), dict()
            for coord in sites.keys():
                for dxy,ind in dxy_w_to_ind.items():
                    # generate weight_id and reverse weight_id
                    # (coord,dxy) identifies the same weight as (coord+dxy,-dxy) 
                    w_id= (coord, dxy)
                    w_rid= (self.vertexToSite(add_(coord,dxy)), neg_(dxy))

                    if not w_id in new_weights.keys() and not w_rid in new_weights.keys():
                        #   
                        #       |                                              |
                        #   --(0,0)-X^{-1}-X[w_id]-w[w_id]-X'[w_rid]-X'^{-1}-(1,0)--
                        #       |                                              |
                        #   => 
                        #       |                               |
                        #   --(0,0)-X^{-1}--U--S--Vh--X'^{-1}-(1,0)--
                        #       |                               |
                        #
                        _match_diag_signature= 1 if -weights[w_id].get_signature()[1]==Xs[w_id][0].get_signature()[0] else 0
                        #
                        #  ... (s)--w[w_id]--?(-s) (s)0--X[w_id]--1(sU=-s) -> 0(sU=-s)--X[w_id]--w[w_id]-- ...
                        _sgn= -Xs[w_id][0].get_signature()[1] #-weights[w_id].get_signature()[0]
                        U,S,Vh= yastn.linalg.svd( Xs[w_id][0].tensordot(weights[w_id],([0],[_match_diag_signature]))\
                            .tensordot(Xs[w_rid][0],([1],[0])), sU=_sgn )
                        new_weights[w_id]= S #/S[0]
                        new_weights[w_rid]= S #/S[0]
                        #
                        # (sU)0--U--1(_sgn) -> (_sgn=s)0--U--1(sU)
                        Us[w_id]= U.transpose((1,0))
                        Us[w_rid]= Vh
            
            new_sites={}
            for coord in sites.keys():
                A= sites[self.vertexToSite(coord)]
                expr= 'smnop,'+','.join([expr_ws[d] for d in dxy_w_to_ind.keys()])+'->suldr'
                # (_sgn=s)0--U--1(sU) (-sU)0--Xinv--1(-s) = (_sgn=s)0--UXinv--1(-s)  <=> n,ln->l
                new_sites[coord]= yastn.einsum(expr,A,*(Us[(coord,d)].tensordot(Xs[(coord,d)][1],([1],[0]))\
                    for d in dxy_w_to_ind.keys()))
                # new_sites[coord]= A/A.abs().max()

            return new_sites, new_weights

        dist=[float('inf')]
        n_s, n_w= { c: t/t.norm(p='inf') for c,t in self.sites.items() }, self.weights
        while dist[-1]>peps_args.quasi_gauge_tol and len(dist)<peps_args.quasi_gauge_max_iter:
            # generate X, Xinv for site and bond
            Xs= { (coord,d): _get_dl_gauges(coord,d,n_s,n_w) \
                for  coord in n_s.keys() for d in [(0,-1),(-1,0),(0,1),(1,0)] }

            n_s, n_w1= _update_weights_and_sites(n_s,n_w,Xs)
            dist.append(sum([ yastn.norm( \
                n_w1[k] - n_w[k] \
                if n_w1[k].get_signature()[0]==n_w[k].get_signature()[0] else \
                n_w1[k] - n_w[k].transpose((1,0)) ).item() for k in n_w.keys() ])/len(n_s))
            n_w= n_w1

        log.info(f"gauge dist_legth: {len(dist)}, dist: {dist}")
        return type(self)(sites=n_s, weights=n_w, vertexToSite=self.vertexToSite, \
            lX=self.lX, lY=self.lY, peps_args=peps_args, global_args=global_args)  


def get_weighted_ipeps(state, weights, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: abelian-symmetric iPEPS
    :param weights: map from edges within unit cell to weight tensors
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type state: IPEPS_ABELIAN
    :type weights: dict[tuple(tuple(int,int), tuple(int,int)) : yamps.tensor.Tensor]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: iPEPS wavefunction augmented with weights
    :rtype: IPEPS_ABELIAN_WEIGHTED

    Create IPEPS_ABELIAN_WEIGHTED from regular IPEPS_ABELIAN by mapping weight tensor to each
    bond in the elementary unit cell.
    """
    return IPEPS_ABELIAN_WEIGHTED(state.engine, state.sites, weights,\
        vertexToSite=state.vertexToSite, lX=state.lX, lY=state.lY,\
        peps_args=peps_args, global_args=global_args)