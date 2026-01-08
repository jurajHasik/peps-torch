import torch
from collections import OrderedDict
import json
import math
import config as cfg
import ipeps.ipeps as ipeps
from ipeps.tensor_io import *

class IPEPS_TRGL_1S_TTPHYS_PG(ipeps.IPEPS):
    PG_A1= {'t_aux': 'A_1', 't_phys': 'A_1'}

    def __init__(self, t_aux=None, t_phys=None, pgs=None, pg_symmetrize=False,\
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param t_aux: auxiliary trivalent tensor with left, down, and extra index
                      of bond dimension D
        :type t_aux: torch.Tensor
        :param t_phys: physical trivalent tensor with up, extra, and right auxiliary 
                       index and physical index
        :param pgs: dictionary assigning point-group irreps to elementary tensors
        :type pgs: dict(str,str)
        :type t_phys: torch.Tensor
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Builds single-site ansatz for triangular lattice from two trivalent 
        tensors as::

                                                                              u p
                 |/          |/          |/                                   |/
               --y-- --x-- --y-- --x-- --y-- where on-site tensor is l--x-- --y--r
                       |           |                                    | 
                       |/          |/                                   d
               --x-- --y-- --x-- --y-- 
                 |           |

        """
        self.elem_tensors= OrderedDict({"t_aux": t_aux, "t_phys": t_phys})

        if pgs==None: pgs=dict()
        assert isinstance(pgs,dict) and set(list(pgs.keys()))<=set(['t_aux','t_phys']),\
            "Invalid point-group specification "+str(pgs)
        self.pgs= pgs
        if pg_symmetrize:
            self.elem_tensors= type(self)._to_PG_symmetric(self.pgs, self.elem_tensors)

        sites= None
        if not (t_aux is None or t_phys is None):
            sites= self.build_onsite_tensors()
        super().__init__(sites, lX=1, lY=1,\
            peps_args=peps_args, global_args=global_args)

    def get_parameters(self):
        return self.elem_tensors.values()

    def get_checkpoint(self):
        checkpoint= {"elem_tensors": self.elem_tensors}
        return checkpoint       

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file,map_location=self.device, weights_only=False)
        self.elem_tensors= checkpoint['elem_tensors']
        for t in self.elem_tensors.values(): t.requires_grad_(False)
        if True in [t.is_complex() for t in self.elem_tensors.values()]:
            self.dtype= torch.complex128
        self.sites= self.build_onsite_tensors()

    def write_to_file(self,outputfile,aux_seq=[0,1,2,3], tol=1.0e-14, \
        normalize=False, pg_symmetrize=True):
        """
        Writes state to file. See :meth:`write_ipeps_trgl_1s_trivalent`.
        """
        write_ipeps_trgl_1s_ttphys_pg(self,outputfile,tol=tol, normalize=normalize,\
            pg_symmetrize=pg_symmetrize)

    def build_onsite_tensors(self):
        return {(0,0): torch.einsum('ldx,xurp->puldr',self.elem_tensors['t_aux'],\
            self.elem_tensors['t_phys']).contiguous()}

    def add_noise(self,noise):
        t_aux= self.elem_tensors['t_aux']
        t_phys= self.elem_tensors['t_phys']
        t_aux= t_aux + noise * (torch.rand_like(t_aux) - 0.5)
        t_phys= t_phys + noise * (torch.rand_like(t_phys) - 0.5)
        self.elem_tensors= OrderedDict({'t_aux': t_aux, 't_phys': t_phys})
        return to_PG_symmetric(self, self.pgs)

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPEPS_TRGL_1S_TTPHYS_PG

        Take IPEPS_TRGL_1S_TTPHYS_PG and enlarge all auxiliary bond dimensions 
        of all tensors up to size ``new_d``.
        """
        size_ta= self.elem_tensors['t_aux'].size()
        size_tp= self.elem_tensors['t_phys'].size()
        size_check_ta= [ new_d >= d for d in size_ta ] 
        size_check_tp= [ new_d >= d for d in size_tp[:-1] ]
        if False in size_check_ta or False in size_check_tp:
            raise ValueError("Desired dimension is smaller one of the aux dimensions.")

        t_aux= torch.zeros((new_d,new_d,new_d), dtype=self.elem_tensors['t_aux'].dtype, \
            device=self.elem_tensors['t_aux'].device)
        t_phys= torch.zeros((new_d,new_d,new_d,size_tp[-1]), dtype=self.elem_tensors['t_phys'].dtype, \
            device=self.elem_tensors['t_phys'].device)

        t_aux[:size_ta[0],:size_ta[1],:size_ta[2]]= self.elem_tensors['t_aux']
        t_phys[:size_tp[0],:size_tp[1],:size_tp[2],:]= self.elem_tensors['t_phys'] 
        
        return type(self)(t_aux, t_phys, pgs=self.pgs,\
            peps_args=peps_args, global_args=global_args)

    def normalize_(self):
        self.elem_tensors= { t_id: t/t.abs().max() for t_id,t in self.elem_tensors.items() }
        self.sites= self.build_onsite_tensors()

    @staticmethod
    def _to_PG_symmetric(pgs, elem_ts):
        pg_elem_ts= OrderedDict(elem_ts)
        for t_id,pg in pgs.items():
            if pg is None: continue

            # trivalent tensor 
            if t_id in ["t_aux","t_phys"] and t_id in elem_ts.keys():    
                pd=[3] if t_id=="t_phys" else []
                # A_1 + iA_2
                if pg=="A_1":
                    tmp_t= (1./3)*(elem_ts[t_id]\
                        + elem_ts[t_id].permute([1,2,0]+pd)\
                        + elem_ts[t_id].permute([2,0,1]+pd))
                    tmp_t= 0.5*(tmp_t + tmp_t.permute([0,2,1]+pd).conj())
                    pg_elem_ts[t_id]= tmp_t
                # A_2 + iA_1
                elif pg=="A_2":
                    tmp_t= (1./3)*(elem_ts[t_id]\
                        + elem_ts[t_id].permute([1,2,0]+pd)\
                        + elem_ts[t_id].permute([2,0,1]+pd))
                    tmp_t= 0.5*(tmp_t - tmp_t.permute([0,2,1]+pd).conj())
                    pg_elem_ts[t_id]= tmp_t
                else:
                    raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
        return pg_elem_ts


def write_ipeps_trgl_1s_ttphys_pg(state, outputfile, tol=1.0e-14, normalize=False,\
    pg_symmetrize=False, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :param pg_symmetrize: if True, symmetrize elementary tensors before writing out
    :type pg_symmetrize: bool
    :type state: IPEPS_TRGL_1S_TTPHYS_PG
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    """
    sym_state= to_PG_symmetric(state) if pg_symmetrize else state
    json_state=dict({"lX": sym_state.lX, "lY": sym_state.lY, "elem_tensors": {},  "pgs": state.pgs})
    
    for t_id,t in sym_state.elem_tensors.items():
        if normalize:
            t= t/t.abs().max()
        
        if global_args.tensor_io_format=="legacy":
            json_tensor= serialize_bare_tensor_legacy(t)
            # json_tensor["physDim"]= site.size(0)
            # assuming all auxBondDim are identical
            # json_tensor["auxDim"]= site.size(1)
        elif global_args.tensor_io_format=="1D":
            json_tensor= serialize_bare_tensor_np(t)

        json_state["elem_tensors"][t_id]= json_tensor

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))

def read_ipeps_trgl_1s_ttphys_pg(jsonfile, \
    peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_KAGOME in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_TRGL_1S_TTPHYS_PG
    
    See :meth:`ipeps.ipeps.read_ipeps`.
    """
    WARN_REAL_TO_COMPLEX=False
    with open(jsonfile) as j:
        raw_state = json.load(j)

        assert "elem_tensors" in raw_state.keys(), "Missing elem_tensors."
        raw_elem_t= raw_state["elem_tensors"]
        assert "t_aux" in raw_elem_t.keys() and "t_phys" in raw_elem_t.keys(),\
            "Missing expected elem_tensors."

        pgs= raw_state["pgs"] if "pgs" in raw_state.keys() else None

        # Loop over elementary tensors
        elem_tensors={}
        for t_id,t in raw_state["elem_tensors"].items():

            # depending on the "format", read the bare tensor
            if "format" in t.keys():
                if t["format"]=="1D":
                    X= torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

             # allow promotion of real to complex dtype
            _typeT= torch.zeros(1,dtype=global_args.torch_dtype)
            if _typeT.is_complex() and not X.is_complex():
                X= X+0.j
                WARN_REAL_TO_COMPLEX= True

            # move to selected device
            elem_tensors[t_id]= X .to(global_args.device)

        if WARN_REAL_TO_COMPLEX: warnings.warn("Some of the tensors were promoted from float to"\
            +" complex dtype", Warning)

    return IPEPS_TRGL_1S_TTPHYS_PG(elem_tensors['t_aux'], elem_tensors['t_phys'], \
        pgs=pgs, peps_args=peps_args, global_args=global_args)


class IPEPS_TRGL_1S_TBT_PG(ipeps.IPEPS):
    PG_A1_A= {'t_aux': 'A_1', 't_phys': 'A'}

    def __init__(self, t_aux=None, t_phys=None, pgs=None, pg_symmetrize=False,\
                 peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param t_aux: auxiliary trivalent tensor with left, down, (up, right) 
                      and extra index of bond dimension D
        :type t_aux: torch.Tensor
        :param t_phys: physical trivalent tensor with and two extra auxiliary 
                       indices and physical index
        :type t_phys: torch.Tensor
        :param pgs: dictionary assigning point-group irreps to elementary tensors
        :type pgs: dict(str,str) 
        :param peps_args: ipeps configuration
        :param global_args: global configuration
        :type peps_args: PEPSARGS
        :type global_args: GLOBALARGS

        Builds single-site ansatz for triangular lattice from two trivalent 
        tensors as::

                                                                       p  u
                 |         / |                                        /   |
               --x-- --x--y--x-- --x-- where on-site tensor is l--x--y----x--r
                       |           |                              | 
                     / |         / |                              d
               --x--y--x-- --x--y--x-- 
                 |           |

        """
        self.elem_tensors= OrderedDict({"t_aux": t_aux, "t_phys": t_phys})
        
        if pgs==None: pgs=dict()
        assert isinstance(pgs,dict) and set(list(pgs.keys()))<=set(['t_aux','t_phys']),\
            "Invalid point-group specification "+str(pgs)
        self.pgs= pgs
        if pg_symmetrize:
            self.elem_tensors= type(self)._to_PG_symmetric(self.pgs, self.elem_tensors)

        sites= None
        if not (t_aux is None or t_phys is None):
            sites= self.build_onsite_tensors()
        super().__init__(sites, lX=1, lY=1,\
            peps_args=peps_args, global_args=global_args)

    def get_parameters(self):
        return self.elem_tensors.values()

    def get_checkpoint(self):
        checkpoint= {"elem_tensors": self.elem_tensors}
        return checkpoint       

    def load_checkpoint(self,checkpoint_file):
        checkpoint= torch.load(checkpoint_file,map_location=self.device, weights_only=False)
        self.elem_tensors= checkpoint['elem_tensors']
        for t in self.elem_tensors.values(): t.requires_grad_(False)
        if True in [t.is_complex() for t in self.elem_tensors.values()]:
            self.dtype= torch.complex128
        self.sites= self.build_onsite_tensors()

    def write_to_file(self,outputfile,aux_seq=[0,1,2,3], tol=1.0e-14, normalize=False,\
        pg_symmetrize=True):
        """
        Writes state to file. See :meth:`write_ipeps_trgl_1s_pg`.
        """
        write_ipeps_trgl_1s_pg(self,outputfile,tol=tol, normalize=normalize,\
            pg_symmetrize=pg_symmetrize)

    def build_onsite_tensors(self):
        return {(0,0): torch.einsum('ldx,xyp,yur->puldr',self.elem_tensors['t_aux'],\
            self.elem_tensors['t_phys'],self.elem_tensors['t_aux']).contiguous()}

    def add_noise(self,noise):
        t_aux= self.elem_tensors['t_aux']
        t_phys= self.elem_tensors['t_phys']
        t_aux= t_aux + noise * (torch.rand_like(t_aux) - 0.5)
        t_phys= t_phys + noise * (torch.rand_like(t_phys) - 0.5)
        self.elem_tensors= OrderedDict({'t_aux': t_aux, 't_phys': t_phys})
        return to_PG_symmetric(self, self.pgs)

    def extend_bond_dim(self, new_d, peps_args=cfg.peps_args, global_args=cfg.global_args):
        r"""
        :param new_d: new enlarged auxiliary bond dimension
        :type new_d: int
        :return: wavefunction with enlarged auxiliary bond dimensions
        :rtype: IPEPS_TRGL_1S_TTPHYS_PG

        Take IPEPS_TRGL_1S_TTPHYS_PG and enlarge all auxiliary bond dimensions 
        of all tensors up to size ``new_d``.
        """
        size_ta= self.elem_tensors['t_aux'].size()
        size_tp= self.elem_tensors['t_phys'].size()
        size_check_ta= [ new_d >= d for d in size_ta ] 
        size_check_tp= [ new_d >= d for d in size_tp[:-1] ]
        if False in size_check_ta or False in size_check_tp:
            raise ValueError("Desired dimension is smaller one of the aux dimensions.")

        t_aux= torch.zeros((new_d,new_d,new_d), dtype=self.elem_tensors['t_aux'].dtype, \
            device=self.elem_tensors['t_aux'].device)
        t_phys= torch.zeros((new_d,new_d,size_tp[-1]), dtype=self.elem_tensors['t_phys'].dtype, \
            device=self.elem_tensors['t_phys'].device)

        t_aux[:size_ta[0],:size_ta[1],:size_ta[2]]= self.elem_tensors['t_aux']
        t_phys[:size_tp[0],:size_tp[1],:]= self.elem_tensors['t_phys'] 
        
        return type(self)(t_aux, t_phys, \
            peps_args=peps_args, global_args=global_args)

    def normalize_(self):
        self.elem_tensors= { t_id: t/t.abs().max() for t_id,t in self.elem_tensors.items() }
        self.sites= self.build_onsite_tensors()

    @staticmethod
    def _to_PG_symmetric(pgs, elem_ts):
        pg_elem_ts= OrderedDict(elem_ts)
        for t_id,pg in pgs.items():
            if pg is None: continue
            # bond-tensors        
            if t_id in ["t_phys"] and t_id in elem_ts.keys():
                # A+iB
                if pg=="A":
                    pg_elem_ts[t_id]= 0.5*(elem_ts[t_id]\
                        + elem_ts[t_id].permute(1,0,2).conj())
                elif pg=="B":
                # B + iA 
                    pg_elem_ts[t_id]= 0.5*(elem_ts[t_id]\
                        - elem_ts[t_id].permute(1,0,2).conj())
                else:
                    raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
            # trivalent tensor 
            if t_id in ["t_aux"] and t_id in elem_ts.keys():    
                # A_1 + iA_2
                if pg=="A_1":
                    tmp_t= (1./3)*(elem_ts[t_id]\
                        + elem_ts[t_id].permute(1,2,0)\
                        + elem_ts[t_id].permute(2,0,1))
                    tmp_t= 0.5*(tmp_t + tmp_t.permute(0,2,1).conj())
                    pg_elem_ts[t_id]= tmp_t
                # A_2 + iA_1
                elif pg=="A_2":
                    tmp_t= (1./3)*(elem_ts[t_id]\
                        + elem_ts[t_id].permute(1,2,0)\
                        + elem_ts[t_id].permute(2,0,1))
                    tmp_t= 0.5*(tmp_t - tmp_t.permute(0,2,1).conj())
                    pg_elem_ts[t_id]= tmp_t
                else:
                    raise RuntimeError("Unsupported point-group "+t_id+" "+pg)
        return pg_elem_ts

def to_PG_symmetric(state, pgs=None, peps_args=cfg.peps_args, global_args=cfg.global_args):
    assert type(state)._to_PG_symmetric,"Expected instance with _to_PG_symmetric"
    if pgs is None: pgs= state.pgs

    symm_elem_ts= type(state)._to_PG_symmetric(pgs, state.elem_tensors)

    return type(state)(t_aux= symm_elem_ts['t_aux'], t_phys=symm_elem_ts['t_phys'],\
        pgs=pgs, peps_args=peps_args, global_args=global_args)

def read_ipeps_trgl_1s_tbt_pg(jsonfile, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param jsonfile: input file describing IPEPS_TRGL_1S_TBT_PG in json format
    :param peps_args: ipeps configuration
    :param global_args: global configuration
    :type jsonfile: str or Path object
    :type peps_args: PEPSARGS
    :type global_args: GLOBALARGS
    :return: wavefunction
    :rtype: IPEPS_TRGL_1S_TBT_PG

    Read IPEPS_TRGL_1S_TBT_PG state from file.
    """
    dtype = global_args.torch_dtype
    WARN_REAL_TO_COMPLEX=False

    with open(jsonfile) as j:
        raw_state = json.load(j) 

        assert "pgs" in raw_state.keys(),"Missing point-group specification \"pgs\""
        pgs= raw_state["pgs"]

        # Loop over elementary tensors
        elem_tensors={}
        for t_id,t in raw_state["elem_tensors"].items():

            # depending on the "format", read the bare tensor
            if "format" in t.keys():
                if t["format"]=="1D":
                    X= torch.from_numpy(read_bare_json_tensor_np(t))
            else:
                # default
                X= torch.from_numpy(read_bare_json_tensor_np_legacy(t))

             # allow promotion of real to complex dtype
            _typeT= torch.zeros(1,dtype=global_args.torch_dtype)
            if _typeT.is_complex() and not X.is_complex():
                X= X+0.j
                WARN_REAL_TO_COMPLEX= True

            # move to selected device
            elem_tensors[t_id]= X .to(global_args.device)

        if WARN_REAL_TO_COMPLEX: warnings.warn("Some of the tensors were promoted from float to"\
            +" complex dtype", Warning)

        state = IPEPS_TRGL_1S_TBT_PG(t_aux= elem_tensors['t_aux'], t_phys=elem_tensors['t_phys'],\
        pgs=pgs, peps_args=peps_args, global_args=global_args)
    return state

def write_ipeps_trgl_1s_tbt_pg(state, outputfile, tol=1.0e-14, normalize=False,\
    pg_symmetrize=False, peps_args=cfg.peps_args, global_args=cfg.global_args):
    r"""
    :param state: wavefunction to write out in json format
    :param outputfile: target file
    :param tol: minimum magnitude of tensor elements which are written out
    :param normalize: if True, on-site tensors are normalized before writing
    :param pg_symmetrize: if True, symmetrize elementary tensors before writing out
    :type pg_symmetrize: bool
    :type state: IPEPS_TRGL_1S_TBT_PG
    :type ouputfile: str or Path object
    :type tol: float
    :type normalize: bool
    """
    sym_state= to_PG_symmetric(state) if pg_symmetrize else state
    json_state=dict({"lX": state.lX, "lY": state.lY, "elem_tensors": {}, "pgs": state.pgs})
    
    for t_id,t in state.elem_tensors.items():
        t= t/t.abs().max() if normalize else t
        
        if global_args.tensor_io_format=="legacy":
            json_tensor= serialize_bare_tensor_legacy(t)
        elif global_args.tensor_io_format=="1D":
            json_tensor= serialize_bare_tensor_np(t)

        json_state["elem_tensors"][t_id]= json_tensor

    with open(outputfile,'w') as f:
        json.dump(json_state, f, indent=4, separators=(',', ': '))