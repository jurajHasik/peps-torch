""""Build the Alpha tensor SU(2) and C4v symmetric from Mathieu's Mathematica
    library."""
import torch
import tensors.tensors_D4_R3
import tensors.tensors_D4_R5
import tensors.tensors_D4_R5_Cx
import tensors.tensors_D4_R5_all
import ipeps.ipeps as ipeps
import ipeps.ipeps_c4v as ipepsc4v

### WRITING BASIC TENSORS INTO IPEPS CLASS IN THE .JSON FORMAT ###

def contract_Ta(tensor1, tensor2):
    """ Contract spin-SU(2) symmetric tensors of rank-5 and rank-3 into a
    basic tensor T_{\alpha} 
    Indices: (ancilla, physical, u, l, d, r) """
    return torch.einsum('atp,auldr->tpuldr', tensor1, tensor2).view(4,4,4,4,4).contiguous()


def contract_A_c4v():
    """Return the list of the 8 basic c4v Ta tensors.
    """
    tensor_Ta_list = []
    for tensor in tensors.tensors_D4_R5.list_S0:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_B_1, tensor))
    for tensor in tensors.tensors_D4_R5.list_S1:
        tensor_Ta_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_A_1, tensor))
    return tensor_Ta_list


def build_Ta_tensors():
    """Build the .json file of the 8 Ta tensors c4v-sym for the peps-torch CTM.
    """
    tensor_Ta_list = contract_A_c4v()
    for i in range(len(tensor_Ta_list)):
        tensor = tensor_Ta_list[i].view(4,4,4,4,4).contiguous()
        ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(tensor), f"tensors/input-states/tensor_Ta{i}.json")


def build_A_tensor(coef_list):
    """Build the .json file associated with the A tensor from the Ta .json files.
    """
    A_tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    for i in range(len(coef_list)):
        tensor_Ta = ipepsc4v.read_ipeps_c4v(f"tensors/input-states/tensor_Ta{i}.json")
        A_tensor += coef_list[i]*next(iter(tensor_Ta.sites.values()))
    ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(A_tensor),"tensors/input-states/A_tensor.json")
    return A_tensor

def contract_B_cx():
    """Return the list of the 16 basic Tb Cx tensors.
    """
    tensor_Tb_list = []
    for tensor in tensors.tensors_D4_R5_Cx.list_S0:
        tensor_Tb_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_B_1, tensor))
    for tensor in tensors.tensors_D4_R5_Cx.list_S1:
        tensor_Tb_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_A_1, tensor))
    return tensor_Tb_list


def build_Tb_tensors():
    """Build the .json file of the 13 Tb tensors cs-sym for the peps-torch CTM.
    """
    tensor_Tb_list = contract_B_cx()
    for i in range(len(tensor_Tb_list)):
        tensor = tensor_Tb_list[i].view(4,4,4,4,4).contiguous()
        ipeps.write_ipeps(ipeps.IPEPS({(0,0): tensor}),f"tensors/input-states/tensor_Tb{i}.json")


def build_B_tensor(coef_list):
    """Build the .json file associated with the B tensor from the Ta2 .json files.
    """
    B_tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    for i in range(len(coef_list)):
        tensor_Tb = ipeps.read_ipeps(f"tensors/input-states/tensor_Tb{i}.json")
        B_tensor += coef_list[i]*tensor_Tb.site(coord=(0,0))
    ipeps.write_ipeps(ipeps.IPEPS({(0,0): B_tensor}),"tensors/input-states/B_tensor.json")
    return B_tensor


def contract_D_all():
    """Return the list of the 21 basic Tc no-sym tensors.
    """
    tensor_Tc_list = []
    for tensor in tensors.tensors_D4_R5_all.list_S0:
        tensor_Tc_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_B_1, tensor))
    for tensor in tensors.tensors_D4_R5_all.list_S1:
        tensor_Tc_list.append(contract_Ta(tensors.tensors_D4_R3.T_2_A_1, tensor))
    return tensor_Tc_list


def build_Tc_tensors():
    """Build the .json file of the 21 Tc tensors no-sym for the peps-torch CTM.
    """
    tensor_Tc_list = contract_D_all()
    for i in range(len(tensor_Tc_list)):
        tensor = tensor_Tc_list[i].view(4,4,4,4,4).contiguous()
        ipeps.write_ipeps(ipeps.IPEPS({(0,0): tensor}),f"tensors/input-states/tensor_Tc{i}.json")


def write_tensor(coef_list, bond_type):
    """Build the .json file associated with the corresponding tensor from the
    corresponding .json files.
    """
    tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    if bond_type == 'a':
        for i in range(len(coef_list)):
            tensor_Tb = ipeps.read_ipeps(f"tensors/input-states/tensor_Tb{i}.json")
            tensor += coef_list[i]*tensor_Tb.site(coord=(0,0))
        ipeps.write_ipeps(ipeps.IPEPS({(0,0): tensor}),"tensors/input-states/B_tensor.json")
    if bond_type == 'b':
        for i in range(len(coef_list)):
            tensor_Tb = ipeps.read_ipeps(f"tensors/input-states/tensor_Tb{i}.json")
            tensor += coef_list[i]*tensor_Tb.site(coord=(0,0))
        ipeps.write_ipeps(ipeps.IPEPS({(0,0): tensor}),"tensors/input-states/C_tensor.json")
    if bond_type == 'c':
        for i in range(len(coef_list)):
            tensor_Tc = ipeps.read_ipeps(f"tensors/input-states/tensor_Tc{i}.json")
            tensor += coef_list[i]*tensor_Tc.site(coord=(0,0))
        ipeps.write_ipeps(ipeps.IPEPS({(0,0): tensor}),"tensors/input-states/D_tensor.json")
    if bond_type == 'd':
        for i in range(len(coef_list)):
            tensor_Ta = ipepsc4v.read_ipeps_c4v(f"tensors/input-states/tensor_Ta{i}.json")
            tensor += coef_list[i]*next(iter(tensor_Ta.sites.values()))
        ipeps.write_ipeps(ipepsc4v.IPEPS_C4V(tensor),"tensors/input-states/A_tensor.json")
    return tensor

def build_tensor(coef_list, basic_tensors_list):
    tensor = torch.zeros([4,4,4,4,4], dtype=torch.float64)
    for i in range(len(basic_tensors_list)):
        tensor += coef_list[i]*basic_tensors_list[i]
    return tensor

### BUILDING IPEPO TENSORS

def build_A(coef_list):
    """Return the A tensor which is the sum of the basic c4v tensors.
    """
    # revoir les coef
    A_tensor = torch.zeros([2, 2, 4, 4, 4, 4], dtype=torch.float64)
    for tensor in contract_A_c4v():
        A_tensor += tensor*coef_list.pop()
    return A_tensor


def build_E(X_tensor):
    """ Contract two tensors of rank-6 along the ancilla degree of freedom and then
    merge the auxiliary dimensions to get a tensor of rank 6.
    
                 phys u
                    |/
                 l--X--r
                   /|
                  d | u
                    |/
                 l--X*--r
                   /|
                  d phys
                  
    """
    if len(X_tensor.size()) <= 6:
        X_tensor = X_tensor.view(2, 2, 4, 4, 4, 4).contiguous()
    E = torch.tensordot(X_tensor, X_tensor, dims=([0], [0]))
    length0 = len(E.size())
    length = length0
    while length > length0//2:
        E = E.permute(length0//2-1, length-1, *range(0,length0//2-1),
                  *range(length0//2,length-1))
        E = E.reshape(E.shape[0]**2, *E.shape[2:])
        length=len(E.size())
    return E

## MAP COEFFICIENTS ACCORDING TO SYMMETRY
def Cx(init_coef, epsilon=0):
    if len(init_coef) == 8:
        coef = [epsilon]*16
        coef[0] += init_coef[0]
        coef[1] += init_coef[1]
        coef[2] += init_coef[2]
        coef[3] += init_coef[2]
        coef[4] += init_coef[3]
        coef[5] += init_coef[3]
        coef[6] += init_coef[4]
        coef[7] += init_coef[5]
        coef[8] += init_coef[5]
        coef[9] += init_coef[5]
        coef[10] += init_coef[6]
        coef[11] += init_coef[6]
        coef[12] += init_coef[6]
        coef[13] += init_coef[7]
        coef[14] += init_coef[7]
        coef[15] += init_coef[7]
        return coef
    else:
        return init_coef
    
    
def C(init_coef, epsilon=0):
    if len(init_coef) == 16:
        coef = [epsilon]*21
        coef[0] += init_coef[0]
        coef[1] += init_coef[1]
        coef[2] += init_coef[2]
        coef[3] += init_coef[3]
        coef[4] += init_coef[4]
        coef[5] += init_coef[4]
        coef[6] += init_coef[5]
        coef[7] += init_coef[5]
        coef[8] += init_coef[6]
        coef[9] += init_coef[7]
        coef[10] += init_coef[8]
        coef[11] += init_coef[9]
        coef[12] += init_coef[9]
        coef[13] += init_coef[10]
        coef[14] += init_coef[11]
        coef[15] += init_coef[12]
        coef[16] += init_coef[12]
        coef[17] += init_coef[13]
        coef[18] += init_coef[14]
        coef[19] += init_coef[15]
        coef[20] += init_coef[15]
        return coef
    else:
        return init_coef
    

def C4v(init_coef, epsilon=0):
    if len(init_coef) == 21:
        coef = [epsilon]*8
        coef[0] += init_coef[0]
        coef[1] += init_coef[1]
        coef[2] += (init_coef[2]+init_coef[3])/2
        coef[3] += (init_coef[4]+init_coef[5]+init_coef[6]+init_coef[7])/4
        coef[4] += init_coef[8]
        coef[5] += (init_coef[9]+init_coef[10]+init_coef[11]+init_coef[12])/4
        coef[6] += (init_coef[13]+init_coef[14]+init_coef[15]+init_coef[16])/4
        coef[7] += (init_coef[17]+init_coef[18]+init_coef[19]+init_coef[20])/4
        return coef
