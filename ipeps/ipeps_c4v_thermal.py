import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from groups.pg import make_c4v_symm
from ipeps.ipeps_c4v import IPEPS_C4V
from linalg.custom_eig import truncated_eig_sym
from linalg.custom_svd import truncated_svd_gesdd

class IPEPS_C4V_THERMAL(ipeps.IPEPS):
    def __init__(self, site=None, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param site: on-site rank-6 tensor 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type site: torch.Tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Thermal iPEPO defined by single rank-6 tensor with C4v symmetry.
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
        if site is not None:
            assert isinstance(site,torch.Tensor), "site is not a torch.Tensor"
            sites= {(0,0): site}
        else:
            sites= dict()
        super().__init__(sites, lX=1, lY=1, peps_args=peps_args,\
            global_args=global_args)

    def site(self,coord=None):
        return next(iter(self.sites.values()))

    def add_noise(self,noise,symmetrize=False):
        r"""
        :param noise: magnitude of the noise
        :type noise: float

        Take IPEPS and add random uniform noise with magnitude ``noise`` to on-site tensor
        """
        rand_t = torch.rand( self.site().size(), dtype=self.dtype, device=self.device)
        self.sites[(0,0)]= self.site() + noise * rand_t
        if symmetrize:
            dims= self.site().size()
            # C4v group assumes rank-5 tensor with last four indices corresponding
            # to u,l,d,r
            _tmp_rank5= self.site().view(dims[0]*dims[1], *dims[2:])
            if _tmp_rank5.is_complex():
                _tmp_rank5= make_c4v_symm(_tmp_rank5.real ) \
                + make_c4v_symm(_tmp_rank5.imag, irreps=["A2"]) * 1.0j
            else:
                _tmp_rank5= make_c4v_symm(_tmp_rank5)
            self.sites[(0,0)]= _tmp_rank5.view(dims)

    def to_fused_ipeps_c4v(self):
        r"""
        Transform ipepo into ipeps defined by single rank-5 tensor with the
        physical and ancilla dimensions fused.

        Returns:
            IPEPS_C4v: ipeps representaion of the ipepo 
        """
        site= self.site()
        site= site.view(site.size(0)*site.size(1), site.size(2), site.size(3),\
            site.size(4), site.size(5))
        return IPEPS_C4V(site=site)

    def to_nophys_ipeps_c4v(self):
        r"""
        Transform ipepo into ipeps defined by single rank-4 tensor with the
        physical and ancilla dimensions traced over. Ancilla and physical space
        must be compatible.

        Returns:
            IPEPS_C4v: ipeps representation with only aux indices 
        """
        site= torch.einsum('iiuldr->uldr',self.site()).contiguous()
        return IPEPS_C4V(site=site)

    def write_to_file(self, outputfile, symmetrize=True, **kwargs):
        # symmetrize before writing out
        tmp_state= to_ipeps_c4v_thermal(self) if symmetrize else self
        write_ipeps_c4v_thermal(tmp_state, outputfile, **kwargs)

def to_ipeps_c4v_thermal(state, normalize=False):
    #TODO other classes of C4v-symmetric ansatz ?
    # we choose A1 irrep, in principle, other choices are possible (A2, B1, ...)
    assert len(state.sites.items())==1, "state has more than a single on-site tensor"
    A= next(iter(state.sites.values()))
    assert len(A.size())==6, "rank-6 tensor expected as on-site tensor"

    _tmp_A= A.view(A.size(0)*A.size(1), *A.size()[2:])
    if A.is_complex():
        _tmp_A= make_c4v_symm(_tmp_A.real) + make_c4v_symm(_tmp_A.imag, irreps=["A2"]) * 1.0j
    else:
        _tmp_A= make_c4v_symm(_tmp_A)
    A= A.view(A.size())

    if normalize: A= A/A.norm()
    return IPEPS_C4V_THERMAL(A)

def write_ipeps_c4v_thermal(state, outputfile, tol=1.0e-14, normalize=False):
    pass

class IPEPS_C4V_THERMAL_LC(IPEPS_C4V_THERMAL):
    def __init__(self, elem_tensors, coeffs, \
        peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param elem_tensors: list of selected elementary rank-6 tensors
        :param coeffs: coefficients of linear superposition
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type elem_tensors: list[tuple(dict, torch.Tensor)]
        :type coeffs: torch.Tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        The index-position convention for elementary tensors is defined as follows::

               u s 
               |/ 
            l--A--r  <=> A[a,s,u,l,d,r]
              /|
             a d
            
            where a denotes ancilla index, s denotes physical index, and u,l,d,r label 
            four principal directions up, left, down, right in anti-clockwise order 
            starting from up

        """
        self.elem_tensors= elem_tensors
        self.coeffs= OrderedDict({(0,0): coeffs})
        sites= self.build_onsite_tensors()

        super().__init__(next(iter(sites.values())), \
            peps_args=peps_args, global_args=global_args)

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
        
        # TODO for generic linear combination ?
        # print meta-information of considered symmetric tensors
        # for i,et in enumerate(self.elem_tensors):
        #    print(f"{i} {et[0]}")

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
        ts= torch.stack([t for m,t in self.elem_tensors])
        sites=dict()
        for coord,c in self.coeffs.items():
            sites[coord]= torch.einsum('i,iapuldr->apuldr',c,ts)

        return sites

    def add_noise(self,noise):
        for coord in self.coeffs.keys():
            rand_t = torch.rand( self.coeffs[coord].size(), dtype=self.dtype, device=self.device)
            tmp_t = self.coeffs[coord] + noise * rand_t
            self.coeffs[coord]= tmp_t/torch.max(torch.abs(tmp_t))
        self.sites= self.build_onsite_tensors()

    def to_fused_ipeps_c4v(self):
        self.sites= self.build_onsite_tensors()
        return super().to_fused_ipeps_c4v()

    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_c4v_thermal_lc(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

def read_ipeps_c4v_thermal_lc(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
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
    dtype= global_args.torch_dtype
    asq = [x+2 for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # read the list of considered elementary tensors
        ten_list_key="elem_tensors"
        if "sym_tensors" in raw_state.keys(): 
            ten_list_key= "sym_tensors"
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
            assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
                +str(dtype)+" vs elementary tensor "+str(loc_dtype)

            meta=dict({"meta": symt["meta"]})
            dims=[symt["ancDim"]]+[symt["physDim"]]+[symt["auxDim"]]*4
            
            t= torch.zeros(tuple(dims), dtype=dtype, device=global_args.device)
            if t.is_complex():
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:6]])
                    t[inds]= float(tokens[6]) + (0.+1.j)*float(tokens[7])
            else:
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:6]])
                    t[inds]= float(tokens[6])

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
            assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
                +str(dtype)+" vs elementary tensor "+str(loc_dtype)

            X = torch.zeros(t["numEntries"], dtype=dtype, device=global_args.device)

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

    assert len(coeffs)==1,"single site ipepo expected"

    state = IPEPS_C4V_THERMAL_LC(elem_tensors=elem_tensors, coeffs=next(iter(coeffs.values())), \
        peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_c4v_thermal_lc(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
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
    asq = [x+2 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "elem_tensors": [], "coeffs": []})
    
    # write list of considered elementary tensors
    for meta,t in state.elem_tensors:
        json_tensor=dict()
        json_tensor["dtype"]="complex128" if t.is_complex() else "float64"
        json_tensor["meta"]=meta["meta"]

        tdims = t.size()
        tlength = t.numel()
        json_tensor["ancDim"], json_tensor["physDim"]= tdims[0], tdims[1]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[2]
        # get non-zero elements
        t_nonzero= t.nonzero()
        json_tensor["numEntries"]= len(t_nonzero)
        entries = []
        for elem in t_nonzero:
            ei=tuple(elem.tolist())
            entries.append(f"{ei[0]} {ei[1]} {ei[asq[0]]} {ei[asq[1]]} {ei[asq[2]]} {ei[asq[3]]}"\
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


class IPEPS_C4V_THERMAL_TTN(IPEPS_C4V_THERMAL):
    def __init__(self, seed_site, iso_Ds=[], isometries=[], metadata=None,\
            peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param seed_site: seed on-site rank-6 tensor
        :param isometries: isometries used for compression of iPEPO layers
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type seed_site: torch.Tensor
        :type isometries: list(torch.Tensor)
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Thermal iPEPO defined by single seed rank-6 tensor with C4v symmetry.
        The index-position convetion for on-site tensors is defined as follows::

               u s 
               |/ 
            l--A--r  <=> A[a,s,u,l,d,r]
              /|
             a d
            
        where a denotes ancilla index, s denotes physical index, and u,l,d,r label 
        four principal directions up, left, down, right in anti-clockwise order 
        starting from up.

        The on-site tensor is built from tower of seed tensors, with auxiliary indices
        reduced by TTN isometry. The height of the tower is 2**len(isometries)

                    |   
                   /A\
                 W0 | W0
                /  \A/  \
            --W1    |    W1--
                \  /A\  /
                 W0 | W0
                   \A/
                    |

        where W0 is D x D x D_0, W1 is D_0 x D_0 x D_1, etc. where W_i are obtained
        from parent matrix M_i = (D_i x D_i) x (D_i x D_i)

        """
        self.metadata= metadata
        if seed_site is not None:
            assert isinstance(seed_site,torch.Tensor), "site is not a torch.Tensor"
            self.seed_site= seed_site
        self.iso_Ds= iso_Ds
        self.isometries= isometries
        super().__init__(self.build_onsite_tensors(), peps_args=peps_args,\
            global_args=global_args)

    def __str__(self):
        super().__str__()
        print(f"norm(seed_site) {self.seed_site.norm()}")
        print(f"isometries {len(self.isometries)}")
        for iso in self.isometries:
            print(f"{iso.size()}")
        return ""

    def build_onsite_tensors(self):
        A0= self.seed_site
        A= A0.clone()
        for i in range(len(self.isometries)):
            # create hermitian matrix
            M= self.isometries[i]
            M= M/M.abs().max()
            D_i, D_ip1= M.size(0), self.iso_Ds[i]
            # MMdag= M.view([D_i*D_i]*2)@(M.view([D_i*D_i]*2).T.conj())
            # D,U= truncated_eig_sym(MMdag, D_ip1, keep_multiplets=True,\
                # verbosity=cfg.ctm_args.verbosity_projectors)
            U,S,V= truncated_svd_gesdd(M.view([D_i*D_i]*2), D_ip1,\
                keep_multiplets=True, verbosity=cfg.ctm_args.verbosity_projectors)
            # U= U @ V.conj().transpose(1,0)[:,:D_ip1]
            U= U.view(D_i, D_i, D_ip1)
            #   
            #            |/
            #     /--tmp_A--\
            # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
            #     \x       y/
            #
            #              a
            #            |/
            #     /--tmp_A--\
            # l--U      /|   U--r   s^2 x (D_i)^4 x (D_i+1)^2
            #     \x   b   y/
            #
            tmp_A_lr= torch.einsum('mxl,nyr,spambn->spalxbry',U,U,A)
            #
            #             a   u
            #              \ /  
            #               U
            #             |/
            #      x--tmp_A--y
            #            /|  
            #        b\ /
            #          U
            #         / 
            #        d
            #
            tmp_A_ud= torch.einsum('amu,bnd,spmxny->spauxbdy',U,U,A)
            A= torch.einsum('skalxbry,kpauxbdy->spuldr',tmp_A_lr,tmp_A_ud).contiguous()
        return A

    def update_(self):
        self.sites= {(0,0): self.build_onsite_tensors()}

    def get_parameters(self):
        return self.isometries

    # def add_noise(self,noise,symmetrize=False):
    #     r"""
    #     :param noise: magnitude of the noise
    #     :type noise: float

    #     Take IPEPS and add random uniform noise with magnitude ``noise`` to on-site tensor
    #     """
    #     rand_t = torch.rand( self.site().size(), dtype=self.dtype, device=self.device)
    #     self.sites[(0,0)]= self.site() + noise * rand_t
    #     if symmetrize:
    #         dims= self.site().size()
    #         # C4v group assumes rank-5 tensor with last four indices corresponding
    #         # to u,l,d,r
    #         _tmp_rank5= self.site().view(dims[0]*dims[1], *dims[2:])
    #         if _tmp_rank5.is_complex():
    #             _tmp_rank5= make_c4v_symm(_tmp_rank5.real ) \
    #             + make_c4v_symm(_tmp_rank5.imag, irreps=["A2"]) * 1.0j
    #         else:
    #             _tmp_rank5= make_c4v_symm(_tmp_rank5)
    #         self.sites[(0,0)]= _tmp_rank5.view(dims)

    def symmetrize_isometries(self):
        # self.isometries= [
        new_iso= [
            0.5*(iso+iso.permute(1,0,3,2)) for iso in self.isometries
        ]
        return new_iso

    def extend_layers(self, new_layers):
        self.isometries.extend(new_layers)


    def write_to_file(self, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
        write_ipeps_c4v_thermal_ttn(self, outputfile, aux_seq=aux_seq, tol=tol, normalize=normalize)

def read_ipeps_c4v_thermal_ttn(jsonfile, vertexToSite=None, aux_seq=[0,1,2,3], peps_args=cfg.peps_args,\
    global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_C4V_THERMAL_TTN in json format
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
    dtype= global_args.torch_dtype
    asq = [x+2 for x in aux_seq]
    sites = OrderedDict()
    
    with open(jsonfile) as j:
        raw_state = json.load(j)

        # check for presence of "aux_seq" field in jsonfile
        if "aux_ind_seq" in raw_state.keys():
            asq = [x+1 for x in raw_state["aux_ind_seq"]]

        # read the list of considered elementary tensors
        iso_Ds= raw_state["iso_Ds"]
        ten_list_key="isometries"
        isometries=dict()
        for i,symt in raw_state[ten_list_key].items():
            loc_dtype= torch.float64 # assume float64 by default 
            if "dtype" in symt.keys():
                if "complex128"==symt["dtype"]:
                    loc_dtype= torch.complex128  
                elif "float64"==symt["dtype"]:
                    loc_dtype= torch.float64
                else:
                    raise RuntimeError("Invalid dtype: "+symt["dtype"])
            assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
                +str(dtype)+" vs isometry "+str(loc_dtype)

            dims=[symt["D_in"]]*4
            
            t= torch.zeros(tuple(dims), dtype=dtype, device=global_args.device)
            if t.is_complex():
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:4]])
                    t[inds]= float(tokens[4]) + (0.+1.j)*float(tokens[5])
            else:
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:4]])
                    t[inds]= float(tokens[4])

            isometries[i]= t
        assert set([int(k) for k in isometries.keys()])==set(range(len(isometries.keys())))
        isometries= [ isometries[str(i)] for i in range(len(isometries.keys())) ]

        # Loop over non-equivalent tensor,coeffs pairs in the unit cell
        seed_site=None
        if not raw_state["seed_site"] is None:
            symt= raw_state["seed_site"]
            loc_dtype= torch.float64 # assume float64 by default 
            if "dtype" in symt.keys():
                if "complex128"==symt["dtype"]:
                    loc_dtype= torch.complex128  
                elif "float64"==symt["dtype"]:
                    loc_dtype= torch.float64
                else:
                    raise RuntimeError("Invalid dtype: "+symt["dtype"])
            assert loc_dtype==dtype, "dtypes do not match - iPEPS "\
                +str(dtype)+" vs elementary tensor "+str(loc_dtype)

            dims=[symt["ancDim"]]+[symt["physDim"]]+[symt["auxDim"]]*4
            
            t= torch.zeros(tuple(dims), dtype=dtype, device=global_args.device)
            if t.is_complex():
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:6]])
                    t[inds]= float(tokens[6]) + (0.+1.j)*float(tokens[7])
            else:
                for elem in symt["entries"]:
                    tokens= elem.split(' ')
                    inds=tuple([int(i) for i in tokens[0:6]])
                    t[inds]= float(tokens[6])
            seed_site=t

    state = IPEPS_C4V_THERMAL_TTN(seed_site, iso_Ds, isometries, \
        peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_c4v_thermal_ttn(state, outputfile, aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param aux_seq: array specifying order in which the auxiliary indices of on-site tensors 
                    will be stored in the `outputfile`
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, tensors are normalized before writing
    :type state: IPEPS_C4V_THERMAL_TTN
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
    asq = [x+2 for x in aux_seq]
    json_state=dict({"lX": state.lX, "lY": state.lY, "isometries": {}, "seed_site": None,\
        "metadata": None})
        
    if hasattr(state,"metadata") and type(state.metadata)==dict:
        json_state["metadata"]= state.metadata

    json_state["iso_Ds"]= state.iso_Ds

    # write list of considered isometries
    for i,t in enumerate(state.isometries):
        json_tensor=dict()
        json_tensor["dtype"]="complex128" if t.is_complex() else "float64"

        tdims = t.size()
        tlength = t.numel()
        assert len(tdims)==4, "Unexpected dimensionality of isometry parent tensor rank"+str(len(tdims))
        json_tensor["D_in"]= tdims[0]
        # get non-zero elements
        t_nonzero= t.nonzero()
        json_tensor["numEntries"]= len(t_nonzero)
        entries = []
        for elem in t_nonzero:
            ei=tuple(elem.tolist())
            entries.append(f"{ei[0]} {ei[1]} {ei[2]} {ei[3]} {t[ei]}")
        json_tensor["entries"]=entries
        json_state["isometries"][i]=json_tensor

    if not state.seed_site is None:
        t= state.seed_site
        json_tensor=dict()
        json_tensor["dtype"]="complex128" if t.is_complex() else "float64"

        tdims = t.size()
        tlength = t.numel()
        json_tensor["ancDim"], json_tensor["physDim"]= tdims[0], tdims[1]
        # assuming all auxBondDim are identical
        json_tensor["auxDim"]= tdims[2]
        # get non-zero elements
        t_nonzero= t.nonzero()
        json_tensor["numEntries"]= len(t_nonzero)
        entries = []
        for elem in t_nonzero:
            ei=tuple(elem.tolist())
            entries.append(f"{ei[0]} {ei[1]} {ei[asq[0]]} {ei[asq[1]]} {ei[asq[2]]} {ei[asq[3]]}"\
                +f" {t[ei]}")
        json_tensor["entries"]=entries
        json_state["seed_site"]= json_tensor

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))
