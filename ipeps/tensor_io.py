from itertools import product
import json
import numpy as np

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