import warnings
import torch
from collections import OrderedDict
import json
import itertools
import math
import config as cfg
from ipeps.tensor_io import *
import logging
log = logging.getLogger(__name__)

# TODO drop constrain for aux bond dimension to be identical on 
# all bond indices

class IPEPS():
    def __init__(self, sites=None, vertexToSite=None, lX=None, lY=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--a--r  <=> a[s,u,l,d,r]
               |
               d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        Member ``vertexToSite`` is a mapping function from any vertex (x,y) on a square lattice
        passed in as tuple(int,int) to a corresponding vertex within elementary unit cell.
        
        On-site tensor of an IPEPS object ``wfc`` at vertex (x,y) is conveniently accessed 
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
        if not sites:
            self.dtype= global_args.torch_dtype
            self.device= global_args.device
        else:
            assert len(set( tuple( site.dtype for site in sites.values() ) ))==1,"Mixed dtypes in sites"
            assert len(set( tuple( site.device for site in sites.values() ) ))==1,"Mixed devices in sites"
            self.dtype= next(iter(sites.values())).dtype
            self.device= next(iter(sites.values())).device
            self.sites= OrderedDict(sites)

        # TODO we infer the size of the cluster from the keys of sites. Is it OK?
        # infer the size of the cluster
        if (lX is None or lY is None) and sites:
            min_x = min([coord[0] for coord in sites.keys()])
            max_x = max([coord[0] for coord in sites.keys()])
            min_y = min([coord[1] for coord in sites.keys()])
            max_y = max([coord[1] for coord in sites.keys()])
            self.lX = max_x-min_x + 1
            self.lY = max_y-min_y + 1
        elif lX and lY:
            self.lX = lX
            self.lY = lY
        else:
            raise Exception("lX and lY has to set either directly or implicitly by sites")

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
        :rtype: torch.tensor
        """
        return self.sites[self.vertexToSite(coord)]

    def get_parameters(self):
        r"""
        :return: variational parameters of iPEPS
        :rtype: iterable
        
        This function is called by optimizer to access variational parameters of the state.
        """
        return self.sites.values()

    def get_checkpoint(self):
        r"""
        :return: all data necessary to reconstruct the state. In this case member ``sites`` 
        :rtype: dict[tuple(int,int): torch.tensor]
        
        This function is called by optimizer to create checkpoints during 
        the optimization process.
        """
        return self.sites

    def load_checkpoint(self,checkpoint_file):
        r"""
        :param checkpoint_file: path to checkpoint file 
        :type checkpoint_file: str
        
        Initializes the state according to the supplied checkpoint file.

        .. note:: 

            The `vertexToSite` mapping function is not a part of checkpoint and must 
            be provided either when instantiating IPEPS_ABELIAN or afterwards.
        """
        checkpoint= torch.load(checkpoint_file,map_location=self.device, weights_only=False)
        self.sites= checkpoint["parameters"]
        for site_t in self.sites.values(): site_t.requires_grad_(False)
        if True in [s.is_complex() for s in self.sites.values()]:
            self.dtype= torch.complex128

    def write_to_file(self,outputfile,aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        """
        Writes state to file. See :meth:`write_ipeps`.
        """
        write_ipeps(self,outputfile,aux_seq=aux_seq, tol=tol, normalize=normalize)

    def add_noise(self,noise,noise_f=None):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to all on-site tensors
        """
        for coord in self.sites.keys():
            if noise_f:
                rand_t = noise_f(self.sites[coord].size(), dtype=self.dtype, device=self.device)    
            else:
                rand_t = torch.rand(self.sites[coord].size(), dtype=self.dtype, device=self.device)-0.5
            self.sites[coord] = self.sites[coord] + noise * rand_t

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[1:]]

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"a{nid} {coord}: {site.size()}")
        
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

    def normalize_(self):
        for c in self.sites.keys():
            self.sites[c]= self.sites[c]/self.sites[c].abs().max()

def read_ipeps(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
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
    WARN_REAL_TO_COMPLEX=False
    asq = [x+1 for x in aux_seq]
    sites = OrderedDict()

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

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
                if t["format"]=="1D":
                    X= torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

            sites[coord]= X.permute((0, *asq)) 

            # allow promotion of real to complex dtype
            _typeT= torch.zeros(1,dtype=global_args.torch_dtype)
            if _typeT.is_complex() and not sites[coord].is_complex():
                sites[coord]= sites[coord] + 0.j 
                WARN_REAL_TO_COMPLEX= True

            # move to selected device
            sites[coord]= sites[coord].to(global_args.device)

        if WARN_REAL_TO_COMPLEX: warnings.warn("Some of the tensors were promoted from float to"\
            +" complex dtype", Warning)

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

            state = IPEPS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPEPS(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)

        # set the correct dtype for newly created state (might be different
        # default in cfg.global_args)
        # if True in [s.is_complex() for s in sites.values()]:
        #     state.dtype= torch.complex128
    return state

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
    for coord,site in new_state.sites.items():
        dims = site.size()
        size_check = [new_d >= d for d in dims[1:]]
        if False in size_check:
            raise ValueError("Desired dimension is smaller than following aux dimensions: "+str(size_check))

        new_site = torch.zeros((dims[0],new_d,new_d,new_d,new_d), dtype=state.dtype, device=state.device)
        new_site[:,:dims[1],:dims[2],:dims[3],:dims[4]] = site
        new_state.sites[coord] = new_site
    return new_state

def _write_ipeps_json(state, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "sites": []})
    
    site_ids=[]
    site_map=[]
    for nid,coord,site in [(t[0], *t[1]) for t in enumerate(state.sites.items())]:
        if normalize:
            site= site/site.abs().max()
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        
        if global_args.tensor_io_format=="legacy":
            json_tensor= serialize_bare_tensor_legacy(site)
            # json_tensor["physDim"]= site.size(0)
            # assuming all auxBondDim are identical
            # json_tensor["auxDim"]= site.size(1)
        elif global_args.tensor_io_format=="1D":
            json_tensor= serialize_bare_tensor_np(site)

        json_tensor["siteId"]=site_ids[-1]
        json_state["sites"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map
    return json_state
    
def write_ipeps(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
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

    """
    json_state= _write_ipeps_json(state, aux_seq=aux_seq, tol=tol, normalize=normalize,\
        peps_args=peps_args, global_args=global_args) 

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))


class IPEPS_WEIGHTED(IPEPS):

    # TODO validate weights
    def __init__(self, state=None, sites=None, weights=None, vertexToSite=None, \
        lX=None, lY=None, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param weights: map from edges within unit cell to weight tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.Tensor]
        :type weights: dict[tuple(tuple(int,int), tuple(int,int)) : torch.Tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        IPEPS_WEIGHTED augments basic IPEPS with a tensor on each bond
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
            sites=state.sites
            vertexToSite=state.vertexToSite
            lX=state.lX
            lY=state.lY
        elif sites:
            assert vertexToSite or (lX and lY),"vertexToSite or lX,lY has to be provided" 
        else:
            raise RuntimeError("Either state or sites have to be provided")
        super().__init__(sites, vertexToSite=vertexToSite, lX=lX, lY=lY, 
            peps_args=peps_args, global_args=global_args)
        self.weights= OrderedDict(weights) if weights else self.generate_weights()

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
                    assert self.site(w_id[0]).size(dxy_w_to_ind[w_id[1]])==\
                        self.site(w_rid[0]).size(dxy_w_to_ind[w_rid[1]]),"Bond dims do not match"
                    W= torch.eye(self.site(w_id[0]).size(dxy_w_to_ind[w_id[1]]),\
                        dtype=torch.float64, device=self.site(w_id[0]).device)
                    weights[w_id]= W
                    weights[w_rid]= W
        return weights

    def absorb_weights(self, peps_args=cfg.peps_args, 
        global_args=cfg.global_args):
        r"""
        :return: regular IPEPS obtained by symmetricaly absorbing weights of 
                 IPEPS_WEIGHTED into its on-site tensors
        :rtype: IPEPS

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
        expr_ws= {(0,-1): 'um', (-1,0): 'ln', (0,1): 'do', (1,0): 'rp'}
        full_dxy=set(dxy_w_to_ind.keys())

        a_sites=dict()
        for coord in self.sites.keys():
            A= self.site(coord)
            # 0,[1--0,1->4],2->1,3->2,4->3
            # 0,[1--0,1->4],2->1,3->2,4->3
            # ...
            expr='smnop,'+','.join([expr_ws[dxy] for dxy in dxy_w_to_ind])+'->suldr'
            a_sites[coord]= torch.einsum(expr,\
                A,*( self.weight((coord, dxy)).sqrt()*(1.0+0j) if A.is_complex() \
                    else self.weight((coord, dxy)).sqrt() for dxy in dxy_w_to_ind.keys() ) )
            #for dxy,ind in dxy_w_to_ind.items():
            #    w= self.weight((coord, dxy)).sqrt()
            #    A= torch.tensordot(A, w, ([1],[0]))
            # a_sites[coord]= A

        return IPEPS(a_sites, vertexToSite=self.vertexToSite,\
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
        :rtype: torch.Tensor
        """
        xy_site, dxy= weight_id
        assert dxy in [(0,-1), (-1,0), (0,1), (1,0)],"invalid direction"
        return self.weights[ (self.vertexToSite(xy_site), dxy) ]

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
            expr= 'suldr,smnop,'+','.join([expr_ws[d] for d in ds_to_contract])\
                +f'->{expr_ws[direction]}'
            a= torch.einsum(expr,\
                A,A.conj(),*( (weights[(coord,d)]**2)*(1.+0j) if A.is_complex() else weights[(coord,d)]**2 \
                    for d in ds_to_contract) ).contiguous()

            # diagonalize, since a is hermitian and positive. Force ordering in descending magnitude
            # l--    l(0)--U
            #    a =       sqrt(D)^2
            # n--    n(1)--U
            D,U = torch.linalg.eigh(-a/a.abs().max())
            D= -D
            X= U*D.sqrt()
            D_invsqrt= D.rsqrt()
            D_invsqrt[D/D[0]<1.0e-14]=0
            Xinv= (U*D_invsqrt).t().conj()
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
                        U,S,Vh= torch.linalg.svd(Xs[w_id][0].t()@( \
                            weights[w_id]*(1.0+0j) if Xs[w_id][0].is_complex() else weights[w_id] ) @Xs[w_rid][0])
                        new_weights[w_id]= torch.diag(S)#/S[0]
                        new_weights[w_rid]= torch.diag(S)#/S[0]
                        Us[w_id]= U.t()
                        Us[w_rid]= Vh

            
            new_sites={}
            for coord in sites.keys():
                A= sites[self.vertexToSite(coord)]
                expr= 'smnop,'+','.join([expr_ws[d] for d in dxy_w_to_ind.keys()])+'->suldr'
                new_sites[coord]= torch.einsum(expr,A,*(Us[(coord,d)]@Xs[(coord,d)][1]\
                    for d in dxy_w_to_ind.keys())).contiguous()
                # new_sites[coord]= A/A.abs().max()

            return new_sites, new_weights

        dist=[float('inf')]
        n_s, n_w= { c: t/t.abs().max() for c,t in self.sites.items() }, self.weights
        while dist[-1]>peps_args.quasi_gauge_tol and len(dist)<peps_args.quasi_gauge_max_iter:
            # generate X, Xinv for site and bond
            Xs= { (coord,d): _get_dl_gauges(coord,d,n_s,n_w) \
                for  coord in n_s.keys() for d in [(0,-1),(-1,0),(0,1),(1,0)] }

            n_s, n_w1= _update_weights_and_sites(n_s,n_w,Xs)
            dist.append(sum([ torch.dist(n_w1[k].diag(),n_w[k].diag()) \
                for k in n_w.keys() ]).item()/len(n_s))
            n_w= n_w1

        log.info(f"gauge dist_legth: {len(dist)}, dist: {dist}")
        return type(self)(sites=n_s, weights=n_w, vertexToSite=self.vertexToSite, \
            lX=self.lX, lY=self.lY, peps_args=peps_args, global_args=global_args)  


class IPEPO(IPEPS):
    def __init__(self, sites=None, vertexToSite=None, lX=None, lY=None, peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param sites: map from elementary unit cell to on-site tensors
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sites: dict[tuple(int,int) : torch.tensor]
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Member ``sites`` is a dictionary of non-equivalent on-site tensors
        indexed by tuple of coordinates (x,y) within the elementary unit cell.
        The index-position convetion for on-site tensors is defined as follows::

               u s
               |/
            l--A--r  <=> A[a,s,u,l,d,r]
              /|
             a d

        where a denotes ancilla index, s denotes physical index, and u,l,d,r label
        four principal directions up, left, down, right in anti-clockwise order
        starting from up

        """
        super().__init__(sites, vertexToSite=vertexToSite, lX=lX, lY=lY, peps_args=peps_args,\
            global_args=global_args)

    def get_aux_bond_dims(self):
        return [d for key in self.sites.keys() for d in self.sites[key].size()[2:]]

    def to_fused_ipeps(self):
        r"""
        Transform iPEPO into iPEPS defined by single rank-5 tensor with the
        physical and ancilla dimensions fused.

        Returns:
            IPEPSS: ipeps representaion of the ipepo
        """
        _sites= { c: t.view([t.size(0)*t.size(1)]+t.size()[2:]) \
            for c,t in self.sites.items() }
        return IPEPS(sites=_sites, vertexToSite=self.vertexToSite, lX=self.lX,\
            lY=self.lY)

    def to_nophys_ipeps(self):
        r"""
        Transform iPEPO into iPEPS defined by single rank-4 tensor with the
        physical and ancilla dimensions contracted over. Ancilla and physical space
        must be compatible.

        Returns:
            IPEPS: iPEPS representation with only aux indices
        """
        _sites= { c: torch.einsum('iiuldr->uldr',t).contiguous() \
            for c,t in self.sites.items() }
        return IPEPS(sites=_sites, vertexToSite=self.vertexToSite, lX=self.lX,\
            lY=self.lY)

def read_ipepo(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPO in json format
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
    :rtype: IPEPO
    

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
    WARN_REAL_TO_COMPLEX=False
    asq = [x+2 for x in aux_seq]
    sites = OrderedDict()

    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+2 for x in raw_state["aux_ind_seq"]]

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
                if t["format"]=="1D":
                    X= torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

            sites[coord]= X.permute((0,1, *asq)) 

            # allow promotion of real to complex dtype
            _typeT= torch.zeros(1,dtype=global_args.torch_dtype)
            if _typeT.is_complex() and not sites[coord].is_complex():
                sites[coord]= sites[coord] + 0.j 
                WARN_REAL_TO_COMPLEX= True

            # move to selected device
            sites[coord]= sites[coord].to(global_args.device)

        if WARN_REAL_TO_COMPLEX: warnings.warn("Some of the tensors were promoted from float to"\
            +" complex dtype", Warning)

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

            state = IPEPO(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPEPO(sites, vertexToSite, lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)

        # set the correct dtype for newly created state (might be different
        # default in cfg.global_args)
        # if True in [s.is_complex() for s in sites.values()]:
        #     state.dtype= torch.complex128
    return state

def write_ipepo(state, outputfile, tol=1.0e-14, normalize=False,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: operator to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPO
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

    """
    write_ipeps(state, outputfile, tol=tol, normalize=normalize,\
        peps_args=peps_args, global_args=global_args)
