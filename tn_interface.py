import torch

def contract(t1, t2, *args):
    return torch.tensordot(t1, t2, *args)

def mm(m1, m2):
    return torch.mm(m1, m2)

def einsum(op, *ts):
    return torch.einsum(op, *ts)

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