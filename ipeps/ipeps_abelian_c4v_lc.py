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
import yastn.yastn as yastn
from groups.pg_abelian import make_c4v_symm_A1
from ipeps.tensor_io import *
from ipeps.ipeps_abelian_c4v import IPEPS_ABELIAN_C4V

class IPEPS_ABELIAN_C4V_LC(IPEPS_ABELIAN_C4V):

    def __init__(self, settings, elem_tensors, coeffs, abelian_sym_data, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param elem_tensors: set of elementary tensors
        :param coeffs: coefficients of linear superposition forming on-site tensor
        :param abelian_sym_data: TODO
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type settings: TODO
        :type elem_tensors: list[tuple(dict(str,str),torch.tensor)]
        :type coeffs: dict(tuple(int,int),torch.tensor)
        :type abelian_sym_data: dict(str,list(int))
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Build iPEPS as linear combination of elementary tensors ``elem_tensors``,
        which possess abelian symmetry structure. This structure is given by a 
        list of (integer) charges assigned to physical and auxiliary indices
        ``abelian_sym_data["abelian_charges"]`` and total charge of the on-site tensor
        ``abeluan_sym_data["total_abelian_charge"].

        The index-position convetion for on-site tensor is defined as follows::

           (+1)u (+1)s 
               |/ 
        (+1)l--a--(+1)r  <=> a[s,u,l,d,r] with reference symmetry signature [1,1,1,1,1]
               |
           (+1)d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.
        """
        super().__init__(settings, None, peps_args=peps_args, \
            global_args=global_args)

        self.abelian_sym_data= abelian_sym_data
        self.elem_tensors= elem_tensors
        self.coeffs= OrderedDict(coeffs)
        if (elem_tensors and len(elem_tensors)>0):
            assert len(coeffs)==1, "single-site ipeps is assumed"
            self.sites[(0,0)]= self.build_onsite_tensors()

    def build_onsite_tensors(self, verbosity=0):
        if not self.abelian_sym_data or self.nsym==0:
            # assume a regular dense on-site tensor is requested
            ts= torch.stack([t for m,t in self.elem_tensors])
            sites=dict()
            for coord,c in self.coeffs.items():
                sites[coord]= torch.einsum('i,ipuldr->puldr',c,ts)
            
            site= yastn.Tensor(config=self.engine, s=IPEPS_ABELIAN_C4V._REF_S_DIRS)
            site.set_block(val=next(iter(sites.values())))

            return site

        # give block-structure to on-site tensor
        charges= self.abelian_sym_data["abelian_charges"]
        tot_charge= self.abelian_sym_data["total_abelian_charge"]

        # Assume charges is a list(int) of length len(charges)= physDim + auxBondDim
        # Split charges into physical and auxiliary assuming physDim=2
        oc_p= charges[:2]
        c_a= charges[2:]

        # the initial state creates association index_value->charge
        # 0) sort the charges to create charged sectors with D>1
        #    e.g. D=7 aux-charges (0, 2, -2, 0, 2, -2, 2) -> (-2,-2,0,0,2,2,2) <=> (-2, D=2), (0, D=2), (2, D=3) 
        oc_a= sorted(c_a) # sort charges
        oc_d= {k: oc_a.count(k) for k in set(oc_a)} # compute the block sizes

        # 1) we need to map the index_values from unsorted charges to index_values within sectors
        # 1a) while sorting the charges, sort the index_values occordingly
        #    e.g. D=7 (0, 2, -2, 0, 2, -2, 2)->(-2, -2, 0, 0, 2, 2, 2)
        #             [0,1,2,3,4,5,6]        ->[ 2,  5, 0, 3, 1, 4, 6]
        oc_a_i= sorted(range(len(c_a)), key=c_a.__getitem__)

        # 1b) now map the sorted index values according to charge sectors
        i0= 0 # begin with index value 0
        c0= oc_a[0] # begin with charge at index 0 of list with sorted auxiliary charges
        oc_a_si= []
        a_map= dict() # maps original index_value into (charge, sorted_index_value)
        for i in range(len(oc_a)):
            if c0 != oc_a[i]:
                c0= oc_a[i]
                i0= 0
            oc_a_si.append(i0)
            a_map[oc_a_i[i]]= (oc_a[i], i0)
            i0+=1

        # 2) build blocks
        coeff= next(iter(self.coeffs.values()))
        blocks= dict()
        for i,meta_and_T in enumerate(self.elem_tensors):
            x= coeff[i]
            m,T= meta_and_T
            Tnz= torch.nonzero(T, as_tuple=False)
            for row in Tnz:
                # assign charges to values of indices: first index corresponds to physical leg
                c= tuple([oc_p[row[0]]]+[c_a[row[i]] for i in range(1,5)]) # charges
                iv= tuple(row.tolist())                                    # index_values
                # check if the charged block (key) exists in blocks
                if c not in blocks:
                    blocks[c]= torch.zeros([1]+[oc_d[_c] for _c in c[1:]],
                        dtype=cfg.global_args.torch_dtype, device=cfg.global_args.device)
                    if verbosity>0: print(f"Creating block c={c} of D={blocks[c].shape}")
                # map dense index_values to block index_values (physical dimension has always size 1)
                iv_b= tuple([0]+[a_map[v][1] for v in iv[1:]])
                blocks[c][iv_b]+= x*T[iv]
                if verbosity>0: print(f"elem {c},{iv} -> {iv_b} val {x*T[iv]}")

        # 3) build on-site tensor cls._ref_s_dir,
        c_phys= set(oc_p) # charges on physical leg
        c_aux= set(oc_a)  # charges on (each) auxiliary leg
        d_aux= (oc_d[c] for c in c_aux)
        site = yastn.Tensor(config=self.engine, s=IPEPS_ABELIAN_C4V._REF_S_DIRS, n=tot_charge,
                            t=(c_phys, c_aux, c_aux, c_aux, c_aux),
                            D=((1, 1), d_aux, d_aux, d_aux, d_aux))
        for c,b in blocks.items():
            site.set_block(c, b.shape, val=b)

        return site

    def to(self, device):
        r"""
        :param device: device identifier
        :type device: str
        :return: returns a copy of the state on ``device``. If the state
                 already resides on `device` returns ``self``.
        :rtype: IPEPS_ABELIAN_C4V

        Move the entire state to ``device``.        
        """
        if device==self.device: return self
        elem_tensors= [t.to(device) for t in self.elem_tensors]
        coeffs= self.coeffs.to(device)
        state= IPEPS_ABELIAN_C4V_LC(self.engine, elem_tensors, coeffs, self.abelian_sym_data)
        return state

    def to_dense(self):
        r"""
        :return: returns a copy of the state with all on-site tensors in their dense 
                 representation. If the state already has just dense on-site tensors 
                 returns ``self``.
        :rtype: IPEPS_ABELIAN_C4V_LC

        Create a copy of state with all on-site tensors as dense possesing no explicit
        block structure (symmetry). This operations preserves gradients on returned
        dense state.
        """
        if self.nsym==0: return self
        # TODO don't pass through site conversion
        site_dense= self.site().to_nonsymmetric()
        settings_dense= site_dense.config
        state_dense= IPEPS_ABELIAN_C4V_LC(settings_dense, self.elem_tensors,\
            self.coeffs, None)
        return state_dense

    def get_parameters(self):
        return self.coeffs.values()

    def get_checkpoint(self):
        r"""
        :return: serializable (pickle-able) representation of IPEPS_ABELIAN state
        :rtype: dict

        Return dict containing serialized on-site (block-sparse) tensors. The individual
        blocks are serialized into Numpy ndarrays
        """
        data= {"elem_tensors": self.elem_tensors, "coeffs": self.coeffs, \
            "abelian_sym_data": self.abelian_sym_data}
        return data

    def load_checkpoint(self, checkpoint_file):
        checkpoint= torch.load(checkpoint_file, weights_only=False)
        data= checkpoint["parameters"]
        self.abelian_sym_data= data["abelian_sym_data"]
        self.elem_tensors= data["elem_tensors"]
        self.coeffs= data["coeffs"]
        for coeff_t in self.coeffs.values(): coeff_t.requires_grad_(False)
        self.sites= OrderedDict({(0,0): self.build_onsite_tensors()})

    def write_to_file(self, outputfile, tol=None, normalize=False):
        write_ipeps_c4v_lc(self, outputfile, tol=tol, normalize=normalize)

    def add_noise(self, noise=0):
        r"""
        :param noise: magnitude of the noise
        :type noise: float
        :return: a copy of state with noisy on-site tensors. For default value of 
                 ``noise`` being zero ``self`` is returned. 
        :rtype: IPEPS_ABELIAN_C4V_LC

        Create a new state by adding random uniform noise with magnitude ``noise`` to all 
        copies of on-site tensors. The noise is added to all blocks making up the individual 
        on-site tensors.
        """
        if noise==0: return self
        coeffs= {}
        for coord in self.coeffs.keys():
            rand_t = torch.rand_like(self.coeffs[coord])-0.5
            tmp_t = self.coeffs[coord] + noise * rand_t
            coeffs[coord]= tmp_t/torch.max(torch.abs(tmp_t))
        state= IPEPS_ABELIAN_C4V_LC(self.engine, self.elem_tensors, coeffs, self.abelian_sym_data)
        return state

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
        print(f"{self.abelian_sym_data}")
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

    def serialize_to_json(self, normalize=False, tol=None):
        r"""
        :param state: wavefunction to serialize to json format
        :param tol: minimum magnitude of tensor elements which are written out
        :param normalize: if True, on-site tensors are normalized before writing
        :type state: IPEPS_ABELIAN_C4V_LC
        :type tol: float
        :type normalize: bool
        :return: serialized state
        :rtype: str
        """
        json_state=dict({"lX": self.lX, "lY": self.lY, "elem_tensors": [], "coeffs": []})
        
        # write abelian charge data
        json_state["abelian_charges"]=self.abelian_sym_data["abelian_charges"]
        json_state["total_abelian_charge"]= self.abelian_sym_data["total_abelian_charge"]

        # write list of considered elementary tensors
        for meta,t in self.elem_tensors:
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
                entries.append(f"{ei[0]} {ei[1]} {ei[2]} {ei[3]} {ei[4]}"\
                    +f" {t[ei]}")
            json_tensor["entries"]=entries
            json_state["elem_tensors"].append(json_tensor)

        site_ids=[]
        site_map=[]
        for nid,coord,c in [(t[0], *t[1]) for t in enumerate(self.coeffs.items())]:
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

        return json.dumps(json_state, indent=4, separators=(',', ': '))

def deserialize_from_json(raw_json, settings, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    dtype= global_args.torch_dtype
    raw_state= raw_json
    if isinstance(raw_json,str):
        raw_state= json.loads(raw_json)

    # read abelian charge data
    if "abelian_charges" in raw_state.keys() and "total_abelian_charge" in raw_state.keys():
        abelian_sym_data={}
        abelian_sym_data["abelian_charges"]= raw_state["abelian_charges"]
        abelian_sym_data["total_abelian_charge"]= raw_state["total_abelian_charge"]
    else:
        raise Exception("missing abelian charge data")

    # read the list of elementary tensors
    elem_ten_key="elem_tensors"
    if "su2_tensors" in raw_state.keys():
        elem_ten_key="su2_tensors"
    elif "sym_tensors" in raw_state.keys():
        elem_ten_key="sym_tensors"
    elem_tensors=[]
    for symt in raw_state[elem_ten_key]:
        meta=dict({"meta": symt["meta"]})
        dims=[symt["physDim"]]+[symt["auxDim"]]*4
        
        t= torch.zeros(tuple(dims), dtype=dtype, device=global_args.device)
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
            raise Exception("Tensor with siteId: "+ts["sideId"]+" NOT FOUND in \"coeffs\"") 

        X = torch.zeros(t["numEntries"], dtype=dtype, device=global_args.device)

        # 1) fill the tensor with elements from the list "entries"
        # which list the coefficients in the following
        # notation: Dimensions are indexed starting from 0
        # 
        # index (integer) of coeff, (float) Re, Im  
        for entry in t["entries"]:
            tokens = entry.split()
            X[int(tokens[0])]=float(tokens[1])

        coeffs[coord]=X

    state= IPEPS_ABELIAN_C4V_LC(settings, elem_tensors, coeffs, abelian_sym_data, \
        peps_args=peps_args, global_args=global_args)

    return state

def read_ipeps_c4v_lc(jsonfile, settings, peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing iPEPS_C4V_LC in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_ABELIAN_C4V_LC
    """
    dtype= global_args.torch_dtype
    
    with open(jsonfile) as j:
        raw_state = json.load(j)
        state= deserialize_from_json(raw_state, settings, peps_args=peps_args,\
            global_args=global_args)

    return state

def write_ipeps_c4v_lc(state, outputfile, tol=None, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :type state: IPEPS_ABELIAN_C4V_LC
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    """
    json_str= state.serialize_to_json(tol=tol, normalize=normalize)

    with open(outputfile,'w') as f:
        f.write(json_str)
