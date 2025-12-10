import torch
from collections import OrderedDict
import json
import warnings
import math
import config as cfg
from ipeps.ipeps_lc import IPEPS_LC


class IPEPS_LC_BP(IPEPS_LC):
    def __init__(self, elem_tensors, coeffs, \
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
        :type elem_tensors: list(tuple(dict,torch.Tensor))
        :type coeffs: dict
        :type vertexToSite: function(tuple(int,int))->tuple(int,int)
        :type lX: int
        :type lY: int
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Two-site iPEPS with on-site tensors built from a linear combination
        of elementary tensors. The two tensors decorate square lattice in bipartite pattern. 
        These elementary tensors are representatives of some point group irrep.
        """
        def lattice_to_site(coord):
            vx = (coord[0] + abs(coord[0]) * 2) % 2
            vy = abs(coord[1])
            return ((vx + vy) % 2, 0)

        self.pg_irreps= set([m["meta"]["pg"] for m,t in elem_tensors["site"]])
        super().__init__(elem_tensors, coeffs, vertexToSite=lattice_to_site,
            peps_args=peps_args, global_args=global_args)

    def __str__(self):
        print(f"lX x lY: {self.lX} x {self.lY}")
        for nid,coord,site in [(t[0], *t[1]) for t in enumerate(self.sites.items())]:
            print(f"A{nid} {coord}: {site.size()}")
        
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
                print(f"A{coord_list.index(self.vertexToSite((x,y)))} ", end="")
            print("")
        
        # print meta-information of considered symmetric tensors
        for t_type in self.elem_tensors.keys():
            for i,su2t in enumerate(self.elem_tensors[t_type]):
                print(f"{i} {su2t[0]}")

        # print coefficients
        for nid,c in self.coeffs.items():
            tdims = c.size()
            tlength = tdims[0]
            els=[f"{c[i]}" for i in range(tlength)]
            print(f"{nid} {els}")

        return ""

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
        for t_type in self.coeffs.keys():
            for coeff_t in self.coeffs[t_type].values(): coeff_t.requires_grad_(False)
        if "elem_tensors" in params.keys():
            self.elem_tensors= params["elem_tensors"]
        else:
            warnings.warn("Elementary tensors not included in checkpoint. Using class file instead", Warning)
        self.pg_irreps= set([m["meta"]["pg"] for m,t in self.elem_tensors])
        self.sites= self.build_onsite_tensors()

    def build_onsite_tensors(self):
        # check presence of "A_2" irrep
        if len(self.pg_irreps)==1 and self.pg_irreps==set(["A_1"]):
            ts= torch.stack([t for m,t in self.elem_tensors["site"]])
        elif len(self.pg_irreps)==2 and self.pg_irreps==set(["A_1","A_2"]):
            sym_t_A1= list(filter(lambda x: x[0]["meta"]["pg"]=="A_1", self.elem_tensors["site"]))
            sym_t_A2= list(filter(lambda x: x[0]["meta"]["pg"]=="A_2", self.elem_tensors["site"]))
            ts= torch.stack( [t for m,t in sym_t_A1] + [ 1.0j*t for m,t in sym_t_A2] )
        else:
            raise NotImplementedError("unexpected point group irrep "+str(self.pg_irreps))

        ts_b= torch.stack([t for m,t in self.elem_tensors["bond"]])

        sites=dict()
        c_A= self.coeffs["site"]
        c_b= self.coeffs["bond"]
        if ts.is_complex(): 
            c_A= c_A*(1.0+0.j)
            c_b= c_b*(1.0+0.j)
        sites[(0,0)]= torch.einsum('i,ipuldr->puldr',c_A,ts)
        
        b_T= torch.einsum('i,ilr->lr',c_b,ts_b)
        #
        #        |
        #       b_T
        #        |
        # --b_T--A--b_T--
        #        |
        #       b_T
        #        |
        #
        sites[(1,0)]= torch.einsum('um,ln,dx,ry,pmnxy->puldr',b_T,b_T,b_T,b_T,sites[(0,0)])
        sites[(1,0)]= sites[(1,0)].contiguous()

        return sites

    def add_noise(self,noise):
        for c_type in self.coeffs.keys():
            rand_t = torch.rand( self.coeffs[c_type].size(), dtype=self.dtype, device=self.device)
            self.coeffs[c_type] = self.coeffs[c_type] + noise * (rand_t - 0.5)
        self.coeffs["site"]= self.coeffs["site"]/self.coeffs["site"].abs().max()
        self.coeffs["bond"]= self.coeffs["bond"]/(self.coeffs["site"].abs().max()**(1/4))
        self.sites= self.build_onsite_tensors()

    def get_aux_bond_dims(self):
        return [max(t[1].size()[1:]) for t in self.sites.values()]

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_lc_bp(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

    def clone(self, peps_args=cfg.peps_args, global_args=cfg.global_args, requires_grad=False):
        tmp_elem_t= dict()
        for t_type in self.elem_tensors.keys():
            tmp_elem_t[t_type]= []
            for m,t in self.elem_tensors[t_type]:
                tmp_elem_t[t_type].append((m, t.detach().clone()))
        tmp_coeffs= dict()
        for k,c in self.coeffs.items():
            tmp_coeffs[k]= c.detach().clone()
        
        state_clone= IPEPS_LC_BP(tmp_elem_t, tmp_coeffs,\
            peps_args=peps_args, global_args=global_args)

        return state_clone

    def move_to(self, device):
        for t_type in self.elem_tensors.keys():
            for i,mt in enumerate(self.elem_tensors[t_type]):
                self.elem_tensors[0][1]= mt[1].to(device)
        for k,c in self.coeffs.items():
            self.coeffs[k]= c.to(device)
        
def read_ipeps_lc_bp(jsonfile, aux_seq=[0,1,2,3],\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
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
    """
    dtype= global_args.torch_dtype
    asq = [x+1 for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state= json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # read the list of considered SU(2)-symmetric tensors
        elem_tensors=dict({"site": [], "bond": []})
        for t_type in elem_tensors.keys():
            for symt in raw_state["elem_tensors"][t_type]:
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

                if t_type=="site":
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
                elif t_type=="bond":
                    meta=dict({"meta": symt["meta"]})
                    dims=[symt["auxDim"]]*2
                    
                    t= torch.zeros(tuple(dims), dtype=loc_dtype, device=global_args.device)
                    if t.is_complex():
                        for elem in symt["entries"]:
                            tokens= elem.split(' ')
                            inds=tuple([int(i) for i in tokens[0:2]])
                            t[inds]= float(tokens[2]) + (0.+1.j)*float(tokens[3])
                    else:
                        for elem in symt["entries"]:
                            tokens= elem.split(' ')
                            inds=tuple([int(i) for i in tokens[0:2]])
                            t[inds]= float(tokens[2])
                else:
                    raise RuntimeError("Invalid t_type: "+t_type)

                elem_tensors[t_type].append((meta,t))

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        coeffs=OrderedDict()
        for c_type in elem_tensors.keys():

            # find the corresponding tensor of coeffs (and its elements) 
            # identified by "c_type" 
            t= raw_state["coeffs"][c_type]
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

            coeffs[c_type]=X

    state = IPEPS_LC_BP(elem_tensors=elem_tensors, coeffs=coeffs, \
        peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_lc_bp(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
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
    json_state=dict({"lX": state.lX, "lY": state.lY, "elem_tensors": {"site": [], "bond": []}\
        , "coeffs": {}})
    
    # write list of considered SU(2)-symmetric tensors
    for meta,t in state.elem_tensors["site"]:
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
        json_state["elem_tensors"]["site"].append(json_tensor)

    for meta,t in state.elem_tensors["bond"]:
        json_tensor=dict()
        json_tensor["dtype"]="complex128" if t.is_complex() else "float64"
        json_tensor["meta"]=meta["meta"]

        tdims = t.size()
        tlength = tdims[0]*tdims[1]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[0]
        # get non-zero elements
        t_nonzero= t.nonzero()
        json_tensor["numEntries"]= len(t_nonzero)
        entries = []
        for elem in t_nonzero:
            ei=tuple(elem.tolist())
            entries.append(f"{ei[0]} {ei[1]} {t[ei]}")
        json_tensor["entries"]=entries
        json_state["elem_tensors"]["bond"].append(json_tensor)

    for nid,c in state.coeffs.items():
        json_tensor=dict()
        if normalize:
            if nid=="site": c= c/c.abs().max()
            if nid=="bond": c= c/(c.abs().max()**(1/4))
        
        tdims = c.size()
        tlength = tdims[0]
        
        json_tensor["dtype"]="complex128" if c.is_complex() else "float64"
        json_tensor["numEntries"]= tlength
        entries = []
        for i in range(len(c)):
            entries.append(f"{i} {c[i]}")
            
        json_tensor["entries"]=entries
        json_state["coeffs"][nid]=json_tensor

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