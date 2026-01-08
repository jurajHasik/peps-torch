from abc import ABC, abstractmethod
import torch
from collections import OrderedDict
import json
import warnings
import math
import config as cfg
import ipeps.ipeps as ipeps

class IPEPS_LC(ipeps.IPEPS, ABC):
    
    def __init__(self, elem_tensors, coeffs, vertexToSite=None, lX=None, lY=None, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param elem_tensors: elementary tensors
        :param coeffs: coefficients combining elementary tensors into regular iPEPS
        :param vertexToSite: function mapping arbitrary vertex of a square lattice 
                             into a vertex within elementary unit cell
        :param lX: length of the elementary unit cell in X direction
        :param lY: length of the elementary unit cell in Y direction
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type elem_tensors: iterable
        :type coeffs: iterable
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS
        
        IPEPS_LC, where LC stands for linear combination, encapsulates a subclass
        of iPEPS states where on-site tensors are built as a linear combination of
        selected elementary tensors.

        The function `build_onsite_tensors` is expected to construct regular IPEPS,
        which can be then be treated by CTMRG, by combining `elem_tensors` and `coeffs`.
        """
        self.elem_tensors= elem_tensors
        self.coeffs= OrderedDict(coeffs)
        sites= self.build_onsite_tensors()

        super().__init__(sites, vertexToSite=vertexToSite, peps_args=peps_args,\
            global_args=global_args)

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def get_checkpoint(self):
        pass

    @abstractmethod
    def load_checkpoint(self,checkpoint_file):
        pass

    @abstractmethod
    def build_onsite_tensors(self):
        pass

    @abstractmethod
    def add_noise(self,noise):
        pass

    @abstractmethod
    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        pass


class IPEPS_LC_1SITE_PG(IPEPS_LC):
    def __init__(self, elem_tensors, coeffs, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param elem_tensors: elementary tensors
        :param coeffs: coefficients combining elementary tensors into regular iPEPS
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type elem_tensors: list(tuple(dict,torch.Tensor))
        :type coeffs: dict
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Single-site iPEPS with on-site tensor built from a linear combination
        of elementary tensors. These elementary tensors are representatives 
        of some point group irrep.

        Currently supported combinations are real :math:`A_1`, and complex :math:`A_1 + iA_2`. 
        Where elementary tensors `e` are assumed to be real representatives of either  
        :math:`A_1` or :math:`A_2` irreps of :math:`C_{4v}` point group. The on-site
        tensor is then built as

        .. math:: 

            a = \sum_i \lambda_i e_{A_1;i} + i\sum_j \lambda_j e_{A_2;j}


        where :math:`\vec{\lambda}` is a real vector. 

        Each elementary tensor is described by dict::
            
            elem_tensors= [
                ...,
                ({"meta": {"pg": "A_1"}, ...}, torch.Tensor),
                ...,
            ]

        where the value of "pg" (specified inside dict "meta") is either "A_1" or "A_2". 
        """
        self.pg_irreps= set([m["meta"]["pg"] for m,t in elem_tensors])
        super().__init__(elem_tensors, coeffs, peps_args=peps_args,\
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
        for i,su2t in enumerate(self.elem_tensors):
            print(f"{i} {su2t[0]}")

        # print coefficients
        for nid,coord,c in [(t[0], *t[1]) for t in enumerate(self.coeffs.items())]:
            tdims = c.size()
            tlength = tdims[0]
            
            print(f"x: {coord[0]}, y: {coord[1]}")
            els=[f"{c[i]}" for i in range(tlength)]
            print(els)

        return ""

    def site(self,coord=(0,0)):
        r"""
        :param coord: vertex (x,y). Can be ignored, since the ansatz is single-site.
        :type coord: tuple(int,int)
        :return: on-site tensor
        :rtype: torch.tensor 
        """
        return super().site(coord)

    def get_parameters(self):
        return self.coeffs.values()

    def get_checkpoint(self):
        checkpoint= {"coeffs": self.coeffs, "elem_tensors": self.elem_tensors,\
            "pg_irreps": self.pg_irreps}
        return checkpoint

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file, weights_only=False)
        params= checkpoint["parameters"]
        if "coeffs" in params.keys():
            self.coeffs= params["coeffs"]
        else:
            # legacy checkpoints
            self.coeffs= params
        for coeff_t in self.coeffs.values(): coeff_t.requires_grad_(False)
        if "elem_tensors" in params.keys():
            assert any([ coeff_t.numel()==len(params["elem_tensors"]) for coeff_t \
                in params["coeffs"].values()]),"Length of coefficient vectors does "\
                +"not match the set of elementary tensors"
            self.elem_tensors= params["elem_tensors"]
        else:
            warnings.warn("Elementary tensors not included in checkpoint. Using class file instead", Warning)
        self.pg_irreps= set([m["meta"]["pg"] for m,t in self.elem_tensors])
        self.sites= self.build_onsite_tensors()

    def build_onsite_tensors(self):
        r"""
        :return: sites
        :rtype: dict[tuple(int,int): torch.tensor]

        Builds ``sites`` by combining elementary tensors.
        """
        if len(self.pg_irreps)==1 and self.pg_irreps==set(["A_1"]):
            ts= torch.stack([t for m,t in self.elem_tensors])
        elif len(self.pg_irreps)==2 and self.pg_irreps==set(["A_1","A_2"]):
            sym_t_A1= list(filter(lambda x: x[0]["meta"]["pg"]=="A_1", self.elem_tensors))
            sym_t_A2= list(filter(lambda x: x[0]["meta"]["pg"]=="A_2", self.elem_tensors))
            ts= torch.stack( [t for m,t in sym_t_A1] + [ 1.0j*t for m,t in sym_t_A2] )
        else:
            raise NotImplementedError("unexpected point group irrep "+str(self.pg_irreps))

        sites=dict()
        for coord,c in self.coeffs.items():
            if ts.is_complex(): c= c*(1.0+0.j)
            sites[coord]= torch.einsum('i,ipuldr->puldr',c,ts)

        return sites

    def add_noise(self,noise):
        r"""
        :param noise: magnitude of noise
        :type noise: float

        Take IPEPS_LC_1SITE_PG and add random uniform noise with magnitude noise to 
        vector of coefficients ``coeffs``.
        """
        for coord in self.coeffs.keys():
            rand_t = torch.rand( self.coeffs[coord].size(), dtype=self.coeffs[coord].dtype,\
                device=self.device)
            tmp_t = self.coeffs[coord] + noise * (rand_t-0.5)
            self.coeffs[coord]= tmp_t/torch.max(torch.abs(tmp_t))
        self.sites= self.build_onsite_tensors()

    def get_aux_bond_dims(self):
        return [max(t[1].size()[1:]) for t in self.elem_tensors]

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        r"""
        Write state to file. See :meth:`write_ipeps_lc_1site_pg`
        """
        write_ipeps_lc_1site_pg(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

    def clone(self, peps_args=cfg.peps_args, global_args=cfg.global_args, requires_grad=False):
        tmp_elem_t=[]
        for m,t in self.elem_tensors:
            tmp_elem_t.append((m, t.detach().clone()))
        tmp_coeffs= dict()
        for k,c in self.coeffs.items():
            tmp_coeffs[k]= c.detach().clone()
        
        state_clone= IPEPS_LC_1SITE_PG(tmp_elem_t, tmp_coeffs,\
            peps_args=peps_args, global_args=global_args)

        return state_clone

    def move_to(self, device):
        if device=='cpu' or device==torch.device('cpu'):
            for i,mt in enumerate(self.elem_tensors):
                self.elem_tensors[0][1]= mt[1].to(device)
            for k,c in self.coeffs.items():
                self.coeffs[k]= c.to(device)
        elif device.type=='cuda':
            for i,mt in enumerate(self.elem_tensors):
                self.elem_tensors[0][1]= mt[1].to(device)
            for k,c in self.coeffs.items():
                self.coeffs[k]= c.to(device)
        else:
            raise RuntimeError(f"Unsupported device {device}")

def read_ipeps_lc_1site_pg(jsonfile, aux_seq=[0,1,2,3],\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_LC_1SITE_PG in json format
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_LC_1SITE_PG
    

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
    with open(jsonfile) as j:
        raw_state= json.load(j)
        state= from_json_str(json.dumps(raw_state), aux_seq=aux_seq,\
            peps_args=peps_args, global_args=global_args)
    return state

def from_json_str(json_str, aux_seq=[0,1,2,3],\
    peps_args=cfg.peps_args, global_args=cfg.global_args):

    r"""
    :param json_str: str describing IPEPS_LC_1SITE_PG in json format
    :param aux_seq: array specifying order of auxiliary indices of on-site tensors stored
                    in `jsonfile`
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type aux_seq: list[int]
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_LC_1SITE_PG
    

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
    dtype= global_args.torch_dtype
    asq = [x+1 for x in aux_seq]
    sites = OrderedDict()
    
    raw_state = json.loads(json_str)

    # check for presence of "aux_seq" field in jsonfile
    if "aux_ind_seq" in raw_state.keys():
        asq = [x+1 for x in raw_state["aux_ind_seq"]]

    # read the list of considered SU(2)-symmetric tensors
    ten_list_key="sym_tensors"
    if "elem_tensors" in raw_state.keys(): 
        ten_list_key= "elem_tensors"
    elif "su2_tensors" in raw_state.keys(): 
        ten_list_key= "su2_tensors"
    elem_tensors=[]
    for symt in raw_state[ten_list_key]:
        loc_dtype= torch.float64 # assume float64 by default 
        if "dtype" in symt.keys():
            if "complex128"==symt["dtype"]:
                loc_dtype= torch.complex128  
            elif "float64"==symt["dtype"]:
                loc_dtype= torch.float64
            else:
                raise RuntimeError("Invalid dtype: "+symt["dtype"])
        # NOTE elementary tensors are real, yet the final on-site tensor might be complex
        # assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
        #     +str(dtype)+" vs elementary tensor "+str(loc_dtype)

        meta=dict({"meta": symt["meta"]})
        dims=[symt["physDim"]]+[symt["auxDim"]]*4
        
        t= torch.zeros(tuple(dims), dtype=loc_dtype, device=global_args.device)
        if t.is_complex():
            for elem in symt["entries"]:
                tokens= elem.split(' ')
                inds=tuple([int(i) for i in tokens[0:5]])
                t[inds]= float(tokens[5]) + (0.+1.j)*float(tokens[6])
        else:
            for elem in symt["entries"]:
                tokens= elem.split(' ')
                inds=tuple([int(i) for i in tokens[0:5]])
                t[inds]= float(tokens[5])

        elem_tensors.append((meta,t))

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

        loc_dtype= torch.float64
        if "dtype" in t.keys():
            if "complex128"==t["dtype"]:
                loc_dtype= torch.complex128  
            elif "float64"==t["dtype"]:
                loc_dtype= torch.float64
            else:
                raise RuntimeError("Invalid dtype: "+t["dtype"])
        # NOTE coeff tensors are real, yet the final on-site tensor might be complex
        #      i.e. A_1 + i * A_2 ansatz
        # assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
        #     +str(dtype)+" vs elementary tensor "+str(loc_dtype)

        X = torch.zeros(t["numEntries"], dtype=loc_dtype, device=global_args.device)

        # 1) fill the tensor with elements from the list "entries"
        # which list the coefficients in the following
        # notation: Dimensions are indexed starting from 0
        #
        # index (integer) of coeff, (float) Re, Im
        if X.is_complex():
            for entry in t["entries"]:
                tokens = entry.split()
                X[int(tokens[0])]=float(tokens[1]) + (0.+1.j)*float(tokens[2]) 
        else:
            for entry in t["entries"]:
                tokens = entry.split()
                X[int(tokens[0])]=float(tokens[1])

        coeffs[coord]=X

    state = IPEPS_LC_1SITE_PG(elem_tensors=elem_tensors, coeffs=coeffs, \
        peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_lc_1site_pg(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS_LC_1SITE_PG
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
    # TODO drop constrain for aux bond dimension to be identical on 
    # all bond indices
    # TODO implement cutoff on elements with magnitude below tol
    asq = [x+1 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "elem_tensors": [], "coeffs": []})
    
    # write list of considered elementary tensors
    for meta,t in state.elem_tensors:
        json_tensor=dict()
        json_tensor["dtype"]="complex128" if t.is_complex() else "float64"
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
        json_state["elem_tensors"].append(json_tensor)

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
        json_tensor["dtype"]="complex128" if c.is_complex() else "float64"
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

# TODO make consistent with class method
def load_checkpoint_lc_1site_pg(checkpoint_file, vertexToSite=None, lX=None, lY=None,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    checkpoint= torch.load(checkpoint_file, weights_only=False)
    params= checkpoint["parameters"]
    assert "coeffs" in params,"missing 'coeffs' in checkpoint" 
    assert "elem_tensors" in params,"missing 'elem_tensors' in checkpoint"
    assert "pg_irreps" in params,"missing 'pg_irreps' in checkpoint"
    assert any([ coeff_t.numel()==len(params["elem_tensors"]) for coeff_t in params["coeffs"].values()]),\
        "Length of coefficient vectors does not match the set of elementary tensors"
    assert params["pg_irreps"]==set([m["meta"]["pg"] for m,t in params["elem_tensors"]]),\
        "Expected point groups do not match"
    state= IPEPS_LC_1SITE_PG(params["elem_tensors"], params["coeffs"], vertexToSite=vertexToSite,\
        lX=lX, lY=lY, peps_args=peps_args, global_args=global_args)
    for coeff_t in state.coeffs.values(): coeff_t.requires_grad_(False)
    return state