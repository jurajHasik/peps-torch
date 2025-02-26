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
from ipeps.ipeps_abelian import IPEPS_ABELIAN, write_ipeps
from groups.pg_abelian import make_c4v_symm_A1, make_d2_SW_NE_symm, make_d2_NW_SE_symm
from ipeps.tensor_io import *

class IPEPS_ABELIAN_C4V(IPEPS_ABELIAN):
    
    _REF_S_DIRS=(1,1,1,1,1)

    def __init__(self, settings, site=None, irrep="A1", peps_args=cfg.peps_args,\
        global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param site: on-site tensor
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :param irrep: information for symmetrization
        :type irrep: str
        :type settings: TODO
        :type site: yastn.Tensor
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        The index-position convetion for on-site tensor is defined as follows::

               (+1)u (+1)s 
                   |/ 
            (+1)l--a--(+1)r  <=> a[s,u,l,d,r] with reference symmetry signature [1,1,1,1,1]
                   |
               (+1)d
        
        where s denotes physical index, and u,l,d,r label four principal directions
        up, left, down, right in anti-clockwise order starting from up.

        .. note::
            This signature convention leads to the `bra` interpretation of physical space.
            The on-site tensor is to be understood as 
            :math:`a=\sum_{suldr} a_{suldr} \langle s|\langle u|\langle l|\langle d|\langle r|` 
        """
        def vertexToSite(coord): return (0,0)

        sites= dict()
        if not (site is None):
            sites= OrderedDict({(0,0): site})
        super().__init__(settings, sites, vertexToSite=vertexToSite, lX=1, lY=1,\
            peps_args=peps_args, global_args=global_args)
        self.irrep=irrep

    def site(self, coord=(0,0)):
        return super().site(coord)

    def to(self, device):
        r"""
        :param device: device identifier
        :type device: str
        :return: returns a copy of the state on ``device``. If the state
                 already resides onthe  `device` returns ``self``.
        :rtype: IPEPS_ABELIAN_C4V

        Move the entire state to ``device``.        
        """
        if device==self.device: return self
        site= self.site().to(device)
        state= IPEPS_ABELIAN_C4V(self.engine, site, self.irrep)
        return state

    def to_dense(self):
        r"""
        :return: returns a copy of the state with all on-site tensors in their dense 
                 representation. If the state already has just dense on-site tensors 
                 returns ``self``.
        :rtype: IPEPS_ABELIAN_C4V

        Create a copy of state with all on-site tensors as dense possesing no explicit
        block structure (symmetry). 

        .. note::
            This operations preserves gradients on returned dense state.
        """
        if self.nsym==0: return self
        site_dense= self.site().to_nonsymmetric()
        settings_dense= site_dense.config
        state_dense= IPEPS_ABELIAN_C4V(settings_dense, site_dense)
        return state_dense

    def write_to_file(self, outputfile, tol=None, symmetrize=True, normalize=False):
        """
        Writes state to file. See :meth:`ipeps.ipeps_abelian.write_ipeps`.
        """
        sym_state=self
        if symmetrize:
            sym_state= self.symmetrize()
        write_ipeps(sym_state, outputfile, tol=tol, normalize=normalize,\
            metadata={"irrep": self.irrep})

    def symmetrize(self,irrep=None):
        r"""
        :return: symmetrize on-site tensor by projecting to :math:`A_1` irrep of C4v point group.
        :rtype: IPEPS_ABELIAN_C4V
        """
        irrep= self.irrep if not irrep else irrep
        if not irrep:
            return self
        if irrep=="A1":
            site= make_c4v_symm_A1(self.site())
        elif irrep=="NEEL_TRIANGULAR":
            site= make_d2_NW_SE_symm(make_d2_SW_NE_symm(self.site()))
        state= IPEPS_ABELIAN_C4V(self.engine, site, irrep)
        return state

    # NOTE what about non-initialized blocks, which are however allowed by the symmetry ?
    def add_noise(self, noise=0):
        r"""
        :param noise: magnitude of the noise
        :type noise: float
        :return: a copy of state with noisy on-site tensors. For default value of 
                 ``noise`` being zero ``self`` is returned. 
        :rtype: IPEPS_ABELIAN_C4V

        Create a new state by adding random uniform noise with magnitude ``noise`` to all 
        copies of on-site tensors. The noise is added to all blocks making up the individual 
        on-site tensors.
        """
        if noise==0: return self
        _tmp= self.site()
        t_data, D_data= (l.t for l in _tmp.get_legs(native=True)), (l.D for l in _tmp.get_legs(native=True))
        t_noise= yastn.rand(config=_tmp.config, s=_tmp.s, n=_tmp.n, \
            t=t_data, D=D_data, isdiag=_tmp.isdiag)
        site= _tmp + noise * t_noise
        state= IPEPS_ABELIAN_C4V(self.engine, site, self.irrep)
        state= state.symmetrize()
        return state

    def __str__(self):
        print(f"irrep {self.irrep} lX x lY: {self.lX} x {self.lY}")
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

def read_ipeps_c4v(jsonfile, settings, default_irrep=None,\
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_ABELIAN_C4V in json format
    :param settings: YAST configuration
    :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_ABELIAN_C4V
    
    Read state in JSON format from file.
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
                if not "dims" in t.keys():
                    # assume legacy keys
                    t["dims"]= [t["physDim"]]+[t["auxDim"]]*4
                if not "dtype" in t.keys():
                    t["dtype"]= "float64"
                tmp_t= {"blocks": [t]}
                tmp_t["format"]="abelian"
                tmp_t["dtype"]= "float64"
                tmp_t["nsym"]=0
                tmp_t["symmetry"]=[]
                tmp_t["signature"]= IPEPS_ABELIAN_C4V._REF_S_DIRS
                tmp_t["n"]=0
                tmp_t["isdiag"]=False
                tmp_t["rank"]= len(t["dims"])
                X= read_json_abelian_tensor_legacy(tmp_t, settings)

            sites[coord]= X

        assert len(sites)==1, "Invalid number of on-site tensors. Expected one."
        site= next(iter(sites.values()))
        
        meta= raw_state.get("metadata",{})
        irrep= meta.get("irrep",None)
        if irrep:
            if default_irrep and not (default_irrep==irrep):
                raise Exception(f"default_irrep {default_irrep} does not match state's irrep {irrep}")
        elif default_irrep:
            irrep= default_irrep

        state = IPEPS_ABELIAN_C4V(settings, site, irrep,\
            peps_args=peps_args, global_args=global_args)

    # check dtypes of all on-site tensors for newly created state
    assert (False not in [state.dtype==s.yast_dtype for s in sites.values()]), \
        "incompatible dtype among state and on-site tensors"

    # move to desired device and return
    return state.to(global_args.device)