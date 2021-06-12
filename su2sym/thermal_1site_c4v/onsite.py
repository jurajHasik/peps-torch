import torch
import su2sym.thermal_1site_c4v.base_tensors.base_tensor as bt
import copy

class OnSiteTensor():
    r"""Onsite tensor object. Contains the tensor and every information about it.
    :param param: dict containing the base tensors, their symmetry, the
    coefficients and also the basic parameters of bond dimension, type of the
    torch.tensor and device.
    """
    def __init__(self, param):
        self.symmetry = param['symmetry']
        self.dict = param['base_tensor_dict']
        self.bond_dim = param['bond_dim']
        self.dtype = param['dtype']
        self.device = param['device'] 
        self.base_tensor = bt.base_tensor_sym(param['base_tensor_dict'],
            self.symmetry, self.bond_dim, device=self.device)
        self.coeff = torch.tensor(param['coeff'], dtype=self.dtype, device=self.device)
        
    def site(self):
        """Return the on-site tensor with physical and ancilla index fused."""
        tensor = torch.zeros(tuple([4]+[self.bond_dim]*4), dtype=self.dtype)
        for i in range(len(self.coeff)):
            tensor += self.coeff[i]*self.base_tensor[i]
        return tensor

    def site_unfused(self):
        _tmp= self.site()
        return _tmp.view( tuple([2,2]+[self.bond_dim]*4) )

    def normalize(self):
        self.coeff = self.coeff/self.coeff.abs().max()

    def convert(self, new_symmetry):
        """Convert coeff list from a symmetry to another."""
        new_coeff = bt.convert_list(self, new_symmetry, self.bond_dim)
        self.coeff = new_coeff
        self.symmetry = new_symmetry
        self.base_tensor = bt.base_tensor_sym(self.dict, new_symmetry, self.bond_dim)

    def add_noise(self, noise):
        """Add noise to onsite tensor for the optimization."""
        for i in range(len(self.coeff)):
            self.coeff[i] += noise 
        
    def copy(self):
        return copy.copy(self)
        
    def permute(self, permutation):
        for i in range(len(self.base_tensor)):
            self.base_tensor[i] = self.base_tensor[i].permute(permutation).contiguous()
            
    def unpermute(self, permutation):
        def inv(perm):
            inverse = [0] * len(perm)
            for i, p in enumerate(perm):
                inverse[p] = i
            return inverse
        return self.permute(inv(permutation))