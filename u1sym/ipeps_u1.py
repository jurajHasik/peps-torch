import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps

class IPEPS_U1SYM(ipeps.IPEPS):
    def __init__(self, sym_tensors, coeffs, vertexToSite=None, lX=None, lY=None, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param sym_tensors: list of selected symmetric tensors
        :param coeffs: map from elementary unit cell to vector of coefficients
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type sym_tensors: list[tuple(dict(str,str), torch.tensor)]
        :type coeffs: dict[tuple(int,int) : torch.tensor]
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

        By combining the appropriate ``vertexToSite`` mapping function with elementary unit 
        cell specified through ``sites`` various tilings of a square lattice can be achieved:: 

            # Example 1: 1-site translational iPEPS
            
            sites={(0,0): A}
            def vertexToSite(coord):
                return (0,0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  A A A A
            # -1   A  A A A A
            #  0   A  A A A A
            #  1   A  A A A A

            # Example 2: 2-site bipartite iPEPS
            
            sites={(0,0): A, (1,0): B}
            def vertexToSite(coord):
                x = (coord[0] + abs(coord[0]) * 2) % 2
                y = abs(coord[1])
                return ((x + y) % 2, 0)
            wfc= IPEPS(sites,vertexToSite)
        
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   A  B A B A
            # -1   B  A B A B
            #  0   A  B A B A
            #  1   B  A B A B
        
            # Example 3: iPEPS with 3x2 unit cell with PBC 
            
            sites={(0,0): A, (1,0): B, (2,0): C, (0,1): D, (1,1): E, (2,1): F}
            wfc= IPEPS(sites,lX=3,lY=2)
            
            # resulting tiling:
            # y\x -2 -1 0 1 2
            # -2   B  C A B C
            # -1   E  F D E F
            #  0   B  C A B C
            #  1   E  F D E F

        where in the last example we used default setting for ``vertexToSite``, which
        maps square lattice into elementary unit cell of size `lX` x `lY` assuming 
        periodic boundary conditions (PBC) along both X and Y directions.

        TODO we infer the size of the cluster from the keys of sites. Is it OK?
        """
        self.sym_tensors= sym_tensors
        self.coeffs= OrderedDict(coeffs)
        sites= self.build_onsite_tensors()

        super().__init__(sites, vertexToSite=vertexToSite, peps_args=peps_args,\
            global_args=global_args)

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.coeffs.items())]:
            print(f"A{nid} {coord}: {site.size()}")
        
        # show tiling of a square lattice
        coord_list = list(self.coeffs.keys())
        mx, my = 3*self.lX, 3*self.lY
        label_spacing = 1+int(math.log10(len(self.coeffs.keys())))
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
        
        # print meta-information of considered symmetric tensors
        for i,su2t in enumerate(self.sym_tensors):
            print(f"{i} {su2t[0]}")

        # print coefficients
        for nid,coord,c in [(t[0], *t[1]) for t in enumerate(self.coeffs.items())]:
            tdims = c.size()
            tlength = tdims[0]
            
            print(f"x: {coord[0]}, y: {coord[1]}")
            els=[f"{c[i]}" for i in range(tlength)]
            print(els)

        return ""

    def get_parameters(self):
        return self.coeffs.values()

    def get_checkpoint(self):
        return self.coeffs

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file)
        self.coeffs= checkpoint["parameters"]
        for coeff_t in self.coeffs.values(): coeff_t.requires_grad_(False)
        self.sites= self.build_onsite_tensors()

    def build_onsite_tensors(self):
        ts= torch.stack([t for m,t in self.sym_tensors])
        sites=dict()
        for coord,c in self.coeffs.items():
            sites[coord]= torch.einsum('i,ipuldr->puldr',c,ts)
        return sites

    def add_noise(self,noise):
        for coord in self.coeffs.keys():
            rand_t = torch.rand( self.coeffs[coord].size(), dtype=self.dtype, device=self.device)
            tmp_t = self.coeffs[coord] + noise * rand_t
            self.coeffs[coord]= tmp_t/torch.max(torch.abs(tmp_t))
        self.sites= self.build_onsite_tensors()

    def get_aux_bond_dims(self):
        return [max(t[1].size()[1:]) for t in self.sym_tensors]

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_u1(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

def extend_bond_dim(state, new_d):
    return ipeps.extend_bond_dim(state, new_d)

def read_ipeps_u1(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
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
    asq = [x+1 for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # read the list of considered SU(2)-symmetric tensors
        sym_tensors=[]
        for symt in raw_state["sym_tensors"]:
            meta=dict({"meta": symt["meta"]})
            dims=[symt["physDim"]]+[symt["auxDim"]]*4
            
            t= torch.zeros(tuple(dims), dtype=global_args.dtype, device=global_args.device)
            for elem in symt["entries"]:
                tokens= elem.split(' ')
                inds=tuple([int(i) for i in tokens[0:5]])
                t[inds]= float(tokens[5])

            sym_tensors.append((meta,t))

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        coeffs=OrderedDict()
        for ts in raw_state["map"]:
            coord = (ts["x"],ts["y"])

            # find the corresponding tensor of coeffs (and its elements) 
            # identified by "siteId" in the "sites" list
            t = None
            for s in raw_state["coeffs"]:
                if s["siteId"] == ts["siteId"]:
                    t = s
            if t == None:
                raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"sites\"") 

            X = torch.zeros(t["numEntries"], dtype=global_args.dtype, device=global_args.device)

            # 1) fill the tensor with elements from the list "entries"
            # which list the coefficients in the following
            # notation: Dimensions are indexed starting from 0
            # 
            # index (integer) of coeff, (float) Re, Im  
            for entry in t["entries"]:
                tokens = entry.split()
                X[int(tokens[0])]=float(tokens[1])

            coeffs[coord]=X

        # Unless given, construct a function mapping from
        # any site of square-lattice back to unit-cell
        if vertexToSite == None:
            # check for legacy keys
            lX = 0
            lY = 0
            if "sizeM" in raw_state:
                lX = raw_state["sizeM"]
            else:
                lX = raw_state["lX"]
            if "sizeN" in raw_state:
                lY = raw_state["sizeN"]
            else:
                lY = raw_state["lY"]

            def vertexToSite(coord):
                x = coord[0]
                y = coord[1]
                return ( (x + abs(x)*lX)%lX, (y + abs(y)*lY)%lY )

            state = IPEPS_U1SYM(sym_tensors=sym_tensors, coeffs=coeffs, \
                vertexToSite=vertexToSite, \
                lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
        else:
            state = IPEPS_U1SYM(sym_tensors=sym_tensors, coeffs=coeffs, \
                vertexToSite=vertexToSite, \
                peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_u1(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
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
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "sym_tensors": [], "coeffs": []})
    
    # write list of considered SU(2)-symmetric tensors
    for meta,t in state.sym_tensors:
        json_tensor=dict()
        json_tensor["meta"]=meta["meta"]

        tdims = t.size()
        tlength = tdims[0]*tdims[1]*tdims[2]*tdims[3]*tdims[4]
        json_tensor["physDim"]= tdims[0]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[1]
        # get non-zero elements
        t_nonzero= t.nonzero()
        json_tensor["numEntries"]= len(t_nonzero)
        entries = []
        for elem in t_nonzero:
            ei=tuple(elem.tolist())
            entries.append(f"{ei[0]} {ei[asq[0]]} {ei[asq[1]]} {ei[asq[2]]} {ei[asq[3]]}"\
                +f" {t[ei]}")
        json_tensor["entries"]=entries
        json_state["sym_tensors"].append(json_tensor)

    site_ids=[]
    site_map=[]
    for nid,coord,c in [(t[0], *t[1]) for t in enumerate(state.coeffs.items())]:
        if normalize:
            c= c/torch.max(torch.abs(c))

        json_tensor=dict()
        
        tdims = c.size()
        tlength = tdims[0]
        
        site_ids.append(f"A{nid}")
        site_map.append(dict({"siteId": site_ids[-1], "x": coord[0], "y": coord[1]} ))
        json_tensor["siteId"]=site_ids[-1]
        # assuming all auxBondDim are identical
        json_tensor["numEntries"]= tlength
        entries = []
        for i in range(len(c)):
            entries.append(f"{i} {c[i]}")
            
        json_tensor["entries"]=entries
        json_state["coeffs"].append(json_tensor)

    json_state["siteIds"]=site_ids
    json_state["map"]=site_map

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))