import warnings
from itertools import product
import json
import numpy as np
try:
    import yastn.yastn as yastn
except ImportError as e:
    warnings.warn("yast not available", Warning)
try:
    import torch
except ImportError as e:
    warnings.warn("torch not available", Warning)

class NumPy_Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPy_Encoder, self).default(obj)

# assume bare tensor is defined as JSON object with
# 
#      key type content
# i)   dtype [str] datatype 
# ii)  dims  [list[int]] integer array of dimensions
# iii) data  [list[str]] 1D array of elements as strings
#
# in order to represent both real and complex floats.
# The immediate representation is by default in Numpy.
#
def read_bare_json_tensor(json_obj,backend="numpy"):
    t=None
    if backend=="numpy":
        t= read_bare_json_tensor_np(json_obj)
    else:
        raise Exception(f"Unsupported backend: {backend}")

    return t

def read_bare_json_tensor_np(json_obj):
    dtype_str= json_obj["dtype"].lower()
    assert dtype_str in ["float64","complex128"], "Invalid dtype"+dtype_str
    dims= json_obj["dims"]
    raw_data= json_obj["data"]

    # convert raw_data list[str] into list[dtype]
    if "complex" in dtype_str:
        raw_data= np.asarray(raw_data, dtype=np.complex128)
    else:
        raw_data= np.asarray(raw_data, dtype=np.float64)

    return raw_data.reshape(dims)

def read_bare_json_tensor_np_legacy(json_obj):
    t= json_obj
    # 0) find dtype, else assume float64
    dtype_str= json_obj["dtype"].lower() if "dtype" in t.keys() else "float64"
    assert dtype_str in ["float64","complex128"], "Invalid dtype"+dtype_str

    # 1) find the dimensions of indices
    if "dims" in t.keys():
        dims= t["dims"]
    else:
        # assume all auxiliary indices have the same dimension
        dims= [t["physDim"], t["auxDim"], t["auxDim"], \
            t["auxDim"], t["auxDim"]]

    X= np.zeros(dims, dtype=dtype_str)

    # 1) fill the tensor with elements from the list "entries"
    # which list the non-zero tensor elements in the following
    # notation. Dimensions are indexed starting from 0
    # 
    # index (integer) of physDim, left, up, right, down, (float) Re, Im
    #                             (or generic auxilliary inds ...)  
    if dtype_str=="complex128":
        for entry in t["entries"]:
            l = entry.split()
            X[tuple(int(i) for i in l[:-2])]=float(l[-2])+float(l[-1])*1.0j
    else:
        for entry in t["entries"]:
            l= entry.split()
            k= 1 if len(l)==len(dims)+1 else 2
            X[tuple(int(i) for i in l[:-k])]+=float(l[-k])
    
    return X

# assume abelian block as JSON object is composed of bare tensor
# with additional charge data
# 
#      key type content
# i)   charges [list[int]] list of charges
#
def read_json_abelian_block_np_legacy(json_obj):
    charges= json_obj["charges"]
    bare_t= read_bare_json_tensor_np_legacy(json_obj)
    return charges, bare_t

# assume abelian tensor is defined as JSON object with
#
#      key type content 
# i)   rank      [int] rank of tensor (number of dimensions)
# ii)  signature [list[int]] direction of indices (+1 or -1)
# iii) symmetry  [str] abelian group
# iv)  n         [int] total charge
# v)   isdiag    [bool] diagonal tensor
# vi)  dtype     [str] datatype
# vii) blocks    [list[abelian_block]] list of blocks
#
# NOTE: the default settings (abelian group, backend) of Tensor are injected 
#       by tensor_abelian module 
#
def read_json_abelian_tensor_legacy(json_obj, config, dtype=None, device=None):
    r"""
    The dtype and device is either implicitly given by defaults in config
    or explicity passed through `dtype` and `device` parameters

    :param json_obj: dictionary from parsed json file
    :type json_obj: dict
    :param config: yastn.Tensor configuration
    :type config: namedtuple
    :param dtype: dtype 
    :type dtype: str
    :param device: device
    :type device: str
    """

    # TODO validation
    tensor_io_format= json_obj["format"]
    assert tensor_io_format=="abelian", "Invalid JSON format of tensor: "+tensor_io_format
    nsym= json_obj["nsym"]
    assert nsym==config.sym.NSYM, "Number of abelian symmetries does not match: "\
        +" settings.nsym "+str(config.sym.NSYM)+" tensor "+str(nsym)
    symmetry= json_obj["symmetry"]
    # TODO equivalence between different names such as U1 and U(1)
    # assert symmetry==config.sym.SYM_ID, "Symmetries of settings.sym and tensor do not match"
    s= json_obj["signature"]
    n= json_obj["n"]
    isdiag= json_obj["isdiag"]
    
    if not dtype:
        assert hasattr(config,'default_dtype'), "Either dtype or valid config has to be provided"
        dtype= config.default_dtype
    if not device:
        assert hasattr(config,'default_device'), "Either device or valid config has to be provided"
        device= config.default_device

    dtype_str= json_obj["dtype"].lower()
    assert dtype_str in ["float64","complex128"], "Invalid dtype"+dtype_str
    # allow upcasting from float to complex
    _UPCAST= False
    if dtype_str== dtype: pass
    elif dtype_str=="float64" and dtype=="complex128":
        _UPCAST= True
        warnings.warn(f"Upcasting from "+dtype_str+" to "+dtype)
    else:
        raise RuntimeError("Incompatible dtypes: input tensor "+dtype_str+" config: "+dtype)

    # create empty abelian tensor
    T= yastn.Tensor(config=config, s=s, n=n, isdiag=isdiag, device=device)
    # TODO assign symmetry in constructor or settings ?
    # TODO equivalence between different names such as U1 and U(1)
    if symmetry!=T.config.sym.SYM_ID:
        warnings.warn(f"Incompatible tensor symmetry: Expected {T.config.sym.SYM_ID},"\
            +f" read {symmetry}")
    # T.sym= symmetry

    # parse blocks
    for b in json_obj["blocks"]:
        charges, bare_b= read_json_abelian_block_np_legacy(b)
        if symmetry:
            assert len(charges)==len(nsym*bare_b.shape), f"Number of charges {len(charges)}"\
                +f" incompatible with bare tensor rank {len(bare_b.shape)}"
        # T.set_block(ts=tuple(charges), val=(1.+0.j)*bare_b if _UPCAST else bare_b,\
        #     dtype=dtype)
        T.set_block(ts=tuple(charges), Ds=bare_b.shape, val=(1.+0.j)*bare_b if _UPCAST else bare_b)

    return T

def serialize_bare_tensor_np(t):
    r"""
    Parameters
    ----------
    t: numpy.ndarray

    Returns
    -------
    json_tensor: dict

        JSON-compliant representation of numpy.ndarray
    """
    json_tensor=dict()

    dtype_str= f"{t.dtype}"
    if dtype_str.find(".")>0:
        dtype_str= dtype_str[dtype_str.find(".")+1:]

    json_tensor["format"]= "1D"
    json_tensor["dtype"]= dtype_str
    json_tensor["dims"]= list(t.size())

    json_tensor["data"]= []
    t_1d= t.view(-1)
    for i in range(t.numel()):
        json_tensor["data"].append(f"{t_1d[i]}")
    return json_tensor

def serialize_bare_tensor_legacy(t):
    r"""
    Parameters
    ----------
    t: torch.tensor

    Returns
    -------
    json_tensor: dict

        JSON-compliant representation of torch.tensor
    """
    json_tensor=dict()

    dtype_str= f"{t.dtype}"
    if dtype_str.find(".")>0:
        dtype_str= dtype_str[dtype_str.find(".")+1:]

    json_tensor["dtype"]= dtype_str
    json_tensor["dims"]= list(t.shape)
    
    entries = []
    elem_inds = list(product( *(range(i) for i in t.shape) ))
    if "complex" in dtype_str:
        for ei in elem_inds:
            entries.append(" ".join([f"{i}" for i in ei])\
                +f" {t[ei].real} {t[ei].imag}")
    else:
        for ei in elem_inds:
            entries.append(" ".join([f"{i}" for i in ei])\
                +f" {t[ei]}")

    json_tensor["numEntries"]= len(entries)
    json_tensor["entries"]=entries
    return json_tensor

def serialize_abelian_tensor_legacy(t, native=False):
    r"""
    Parameters
    ----------
    t: yastn.Tensor

    native: bool
        if True serialize tensor with all legs unfused

    Returns
    -------
    json_tensor: dict

        JSON-compliant representation of yastn.Tensor
    """
    json_tensor=dict()

    json_tensor["format"]= "abelian"
    json_tensor["nsym"]= t.config.sym.NSYM
    json_tensor["symmetry"]= t.config.sym.SYM_ID
    json_tensor["rank"]= t.get_rank(native=native)
    json_tensor["signature"]= t.get_signature(native=native)
    json_tensor["n"]= t.get_tensor_charge()
    json_tensor["isdiag"]= t.isdiag
    unique_dtype = t.yast_dtype
    if unique_dtype:
        json_tensor["dtype"]= unique_dtype

    # json_tensor["struct"]= t.struct
    # json_tensor["data"]= serialize_bare_tensor_legacy(t._data)

    json_tensor["blocks"]= []
    for k,D in zip(t.struct.t,t.struct.D):
        json_block= serialize_bare_tensor_legacy(t[k])
        json_block["charges"]= k
        json_tensor["blocks"].append(json_block)

    return json_tensor


def serialize_basis_t(meta,t):
    # assume sparse tensor
    assert isinstance(t,torch.Tensor),"torch.tensor is expected"

    json_tensor=dict()
    json_tensor["dtype"]="complex128" if t.is_complex() else "float64"
    json_tensor["meta"]=meta

    tdims = t.size()
    tlength = t.numel()
    json_tensor["dims"]= list(tdims)
    t_nonzero= t.nonzero()
    json_tensor["numEntries"]= len(t_nonzero)
    entries = []
    for elem in t_nonzero:
        ei=tuple(elem.tolist())
        if t.is_complex():
            entries.append(" ".join(f"{ei[i]}" for i in range(len(ei)))\
                +f" {t[ei].real} {t[ei].imag}")
        else:
            entries.append(" ".join(f"{ei[i]}" for i in range(len(ei)))+f" {t[ei]}")
    json_tensor["entries"]=entries
    return json_tensor

def read_basis_t(json_obj, dtype=None, device=None):
    t_dtype=None
    if json_obj["dtype"]=="float64":
        t_dtype= torch.float64
    if json_obj["dtype"]=="complex128":
        t_dtype= torch.complex128
    if not dtype is None:
        assert dtype==t_dtype,"Selected dtype does not match dtype of the tensor"

    t= torch.zeros(tuple(json_obj["dims"]), dtype=t_dtype, device=device)
    r= len(json_obj["dims"])
    if t.is_complex():
        for elem in json_obj["entries"]:
            tokens= elem.split(' ')
            inds=tuple([int(i) for i in tokens[0:r]])
            t[inds]= float(tokens[r]) + (0.+1.j)*float(tokens[r+1])
    else:
        for elem in json_obj["entries"]:
            tokens= elem.split(' ')
            inds=tuple([int(i) for i in tokens[0:r]])
            t[inds]= float(tokens[r])

    meta= None
    if "meta" in json_obj.keys():
        meta= json_obj["meta"]
    return meta, t