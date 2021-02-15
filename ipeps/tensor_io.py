import warnings
from itertools import product
import json
import numpy as np
try:
    # from yamps.tensor import Tensor
    from yamps.yast import Tensor
except ImportError as e:
    warnings.warn("yamps.tensor not available", Warning)

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
def read_json_abelian_tensor_legacy(json_obj, backend):
    r"""
    :param json_obj: dictionary from parsed json file
    :type json_obj: dict
    """

    # TODO validation
    tensor_io_format= json_obj["format"]
    assert tensor_io_format=="abelian", "Invalid JSON format of tensor: "+tensor_io_format
    nsym= json_obj["nsym"]
    assert nsym==backend.sym.nsym, "Number of abelian symmetries does not match: "\
        +" settings.nsym "+str(backend.sym.nsym)+" tensor "+str(nsym)
    symmetry= json_obj["symmetry"]
    # TODO equivalence between different names such as U1 and U(1)
    # assert symmetry==backend.sym.name, "Symmetries of settings.sym and tensor do not match"
    s= json_obj["signature"]
    n= json_obj["n"]
    isdiag= json_obj["isdiag"]
    dtype_str= json_obj["dtype"].lower()
    assert dtype_str in ["float64","complex128"], "Invalid dtype"+dtype_str
    assert dtype_str==backend.dtype, "dtype of tensor and settings.dtype do not match"

    # create empty abelian tensor
    T= Tensor(settings=backend, s=s, n=n, isdiag=isdiag, dtype= dtype_str)
    # TODO assign symmetry in constructor or settings ?
    # TODO equivalence between different names such as U1 and U(1)
    if symmetry!=T.config.sym.name:
        warnings.warn(f"Incompatible tensor symmetry: Expected {T.config.sym.name},"\
            +f" read {symmetry}")
    # T.sym= symmetry

    # parse blocks
    for b in json_obj["blocks"]:
        charges, bare_b= read_json_abelian_block_np_legacy(b)
        if symmetry:
            assert len(charges)==len(nsym*bare_b.shape), f"Number of charges {len(charges)}"\
                +f" incompatible with bare tensor rank {len(bare_b.shape)}"
        T.set_block(ts=tuple(charges), val=bare_b)

    return T

def serialize_bare_tensor_np(t):
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

def serialize_abelian_tensor_legacy(t):
    json_tensor=dict()

    json_tensor["format"]= "abelian"
    json_tensor["nsym"]= t.config.sym.nsym
    json_tensor["symmetry"]= t.config.sym.name
    json_tensor["rank"]= t._ndim
    json_tensor["signature"]= t.s
    json_tensor["n"]= t.n
    json_tensor["isdiag"]= t.isdiag
    json_tensor["dtype"]= t.config.dtype

    json_tensor["blocks"]= []
    for k in t.A.keys():
        json_block= serialize_bare_tensor_legacy(t.A[k])
        json_block["charges"]= k
        json_tensor["blocks"].append(json_block)

    return json_tensor