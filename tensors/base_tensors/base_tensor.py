import tensors.base_tensors.tensors_R3 as r3
import tensors.base_tensors.tensors_D4 as d4
import torch


def contract_R3R5(bond_dim, tensor_r3, tensor_r5, phys_dim=4):
        return torch.einsum('atp,auldr->tpuldr', tensor_r3, tensor_r5)\
                .view(tuple([phys_dim]+[tensor_r5.size(1)]*4)).contiguous()


def base_tensor_dict(bond_dim=4):
    """ Create a dictionary with all the tensors of the according bond
    dimension."""
    # Define import to use
    bond = {'4': d4}
    r5 = bond[str(bond_dim)]
    # Initialize dictionary
    base_tensor_dict = r5.__dict__.copy()
    delete = [key for key in base_tensor_dict if 'T' not in key]
    for key in delete : del base_tensor_dict[key]
    # Create rank 6 tensor dictionary from imports
    # Tensors divided into 2 groups : singlets (S0) and triplets (S1)
    S0 = [key for key in base_tensor_dict if 'S0' in key]
    S1 = [key for key in base_tensor_dict if 'S1' in key]
    for key in S0 : 
        base_tensor_dict[key] = contract_R3R5(bond_dim, r3.T_S0, base_tensor_dict[key])
    for key in S1 : 
        base_tensor_dict[key] = contract_R3R5(bond_dim, r3.T_S1, base_tensor_dict[key])
    return base_tensor_dict
    

def base_tensor_sym(base_tensor_dict, sym, bond_dim=4):
    """Create the list with the tensors of the symmetry considered."""
    # Define import to use
    bond = {'4': d4}
    r5 = bond[str(bond_dim)]
    # Select appropriate symmetry
    name_list = r5.symmetry['list_'+sym]
    # Initialize list
    sym_list = list()
    for names in name_list:
        tensor = torch.zeros(tuple([4]+[bond_dim]*4), dtype=torch.float64)
        for name in names.split('+'):
            tensor += base_tensor_dict[name]
        sym_list.append(tensor)
    return sym_list


def convert_list(OnSiteTensor, new_symmetry, bond_dim=4):
    """Convert the coefficients list."""
    # Define import to use
    bond = {'4': d4}
    r5 = bond[str(bond_dim)]
    # Select appropriate symmetry
    name_list_ini = r5.symmetry['list_'+OnSiteTensor.symmetry]
    name_list_base = r5.symmetry['list_']
    name_list_fin = r5.symmetry['list_'+new_symmetry]
    # Initialize lists
    base_list = list()
    sym_list = list()
    # step 1 : Convert coeff to base list without symmetry
    for names in name_list_ini:
        base_list += [OnSiteTensor.coeff[name_list_ini.index(names)]]*len(names.split('+'))
    # step 2 : Convert base list to the list with changed symmetry
    for names in name_list_fin:
        coef = 0.
        for name in names.split('+'):
            coef += base_list[name_list_base.index(name)]
        sym_list.append(coef/len(names.split('+')))
    return sym_list
