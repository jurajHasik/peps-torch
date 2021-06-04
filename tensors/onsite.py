import torch
import numpy as np
import pickle
import tensors.base_tensors.base_tensor as bt
import ipeps.ipeps as ipeps
import ipeps.ipeps_c4v as ipepsc4v
import copy

class OnSiteTensor():
    r"""Onsite tensor object. Contains the tensor and every information about it.
    :param coeff: list of the tensor coefficients
    :param symmetry: symmetry of the onsite tensor
    :param bond_dim: bond dimension
    """
    def __init__(self, param):
        self.symmetry = param['symmetry']
        self.coeff = param['coeff']
        self.dict = param['base_tensor_dict']
        self.bond_dim = param['bond_dim']
        self.dtype = param['dtype']
        self.device = param['device'] 
        self.base_tensor = bt.base_tensor_sym(param['base_tensor_dict'],
                                              self.symmetry, self.bond_dim, device=self.device)
        self.write_to_json(param['file'])
        self.coeff_list = list(); self.coeff_list.append(self.coeff)
        
    def site(self):
        """Return the onsite tensor."""
        tensor = torch.zeros(tuple([4]+[self.bond_dim]*4), dtype=self.dtype)
        for i in range(len(self.coeff)):
            tensor += self.coeff[i]*self.base_tensor[i]
        return tensor
    
    def normalize(self):
        self.coeff = list(np.array(self.coeff)/np.max(np.abs(np.array(self.coeff))))

    def convert(self, new_symmetry):
        """Convert coeff list from a symmetry to an other."""
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
    
    def history(self):
        self.normalize()
        self.coeff_list.append(self.coeff)
            
    def write_to_json(self, file):
        self.normalize()
        ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(self.site()), file)

    def save_coeff_to_txt(self, output_file):
        with open(output_file, "a+") as fo:
            fo.write(' '.join([str(val) for val in self.coeff])+'\n')
            
    def save_coeff_to_bin(self, output_file):
        file =  open(output_file, "ab")
        pickler = pickle.Pickler(file)
        pickler.dump(self.coeff_list)
        file.close()