import torch

def tensordot_complex(t1, t2, *args):
    return torch.tensordot(t1.real, t2.real, *args) \
        - torch.tensordot(t1.imag, t2.imag, *args) \
        + (torch.tensordot(t1.real, t2.imag, *args) \
        + torch.tensordot(t1.imag, t2.real, *args)) * 1.0j

def contract(t1, t2, *args):
    if t1.is_complex() and t2.is_complex():
        return tensordot_complex(t1, t2, *args)
    elif not t1.is_complex() and not t2.is_complex():
        return torch.tensordot(t1, t2, *args)
    else:
        raise NotImplementedError(f"Tensors {t1} and {t2} are not either both "\
            +"complex or both real")

def mm_complex(m1, m2):
    return torch.mm(m1.real, m2.real) - torch.mm(m1.imag, m2.imag) \
        + (torch.mm(m1.real, m2.imag) + torch.mm(m1.imag, m2.real)) * 1.0j

def mm(m1, m2):
    if m1.is_complex() and m2.is_complex():
        return mm_complex(m1, m2)
    elif not m1.is_complex() and not m2.is_complex():
        return torch.mm(m1, m2)
    else:
        raise NotImplementedError(f"Tensors {m1} and {m2} are not either both "\
            +"complex or both real")

def einsum_complex(op, *ts):
    if len(ts)!=2: raise NotImplementedError("einsum implementation limited to two tensors")
    return torch.einsum(op, ts[0].real, ts[1].real) \
        - torch.einsum(op, ts[0].imag, ts[1].imag) \
        + (torch.einsum(op, ts[0].real, ts[1].imag) \
        + torch.einsum(op, ts[0].imag, ts[1].real)) * 1.0j

def einsum(op, *ts):
    if False not in [t.is_complex() for t in ts]:
        return einsum_complex(op, *ts)
    elif True not in [t.is_complex() for t in ts]:
        return torch.einsum(op, *ts)
    else:
        raise NotImplementedError(f"Tensors are not either all "\
            +"complex or all real")

def view(t, *args):
    return t.view(*args)

def permute(t, *args):
    return t.permute(*args)

def contiguous(t):
    return t.contiguous()

def transpose(t):
    return torch.transpose(t, 0, 1)

def conj(t):
    return t.conj()