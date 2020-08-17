import torch
import config as cfg

def einsum_complex(operation, tensor1, tensor2):
    tensor_new1 = torch.einsum(operation, tensor1[0], tensor2[0])\
                    - torch.einsum(operation, tensor1[1], tensor2[1])
    tensor_new2 = torch.einsum(operation, tensor1[0], tensor2[1])\
                    + torch.einsum(operation, tensor1[1], tensor2[0])
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def einsumtrace_complex(operation, tensor):
    trace1 = torch.einsum(operation, tensor[0])
    trace2 = torch.einsum(operation, tensor[1])
    return torch.sqrt(trace1 ** 2 + trace2 ** 2)

def tensordot_complex(tensor1, tensor2, operation):
    tensor_new1 = torch.tensordot(tensor1[0], tensor2[0], operation)\
                    - torch.tensordot(tensor1[1], tensor2[1], operation)
    tensor_new2 = torch.tensordot(tensor1[0], tensor2[1], operation)\
                    + torch.tensordot(tensor1[1], tensor2[0], operation)
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def mm_complex(matrix1, matrix2):
    matrix_new1 = torch.mm(matrix1[0], matrix2[0])\
                    - torch.mm(matrix1[1], matrix2[1])
    matrix_new2 = torch.mm(matrix1[0], matrix2[1])\
                    + torch.mm(matrix1[1], matrix2[0])
    #dim = matrix_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    matrix_new = torch.stack((matrix_new1, matrix_new2), dim=0)
    #matrix_new[0] = matrix_new1; matrix_new[1] = matrix_new2
    return matrix_new

def view_complex(operation, tensor):
    tensor_new1 = tensor[0].view(operation)
    tensor_new2 = tensor[1].view(operation)
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def permute_complex(operation, tensor):
    tensor_new1 = tensor[0].permute(operation)
    tensor_new2 = tensor[1].permute(operation)
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def contiguous_complex(tensor):
    tensor_new1 = tensor[0].contiguous()
    tensor_new2 = tensor[1].contiguous()
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def complex_conjugate(tensor):
    tensor_new1 = tensor[0]
    tensor_new2 = -tensor[1]
    #dim = tensor_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    tensor_new = torch.stack((tensor_new1, tensor_new2), dim=0)
    #tensor_new[0] = tensor_new1; tensor_new[1] = tensor_new2
    return tensor_new

def size_complex(tensor):
    return tensor[0].size()

def trace_complex(tensor):
    trace1 = torch.trace(tensor[0])
    trace2 = torch.trace(tensor[1])
    return torch.sqrt(trace1 ** 2 + trace2 ** 2)

def abs_complex(tensor):
    return torch.sqrt(tensor[0] ** 2 + tensor[1] ** 2)

def transpose_complex(matrix):
    matrix_new1 = matrix[0].t()
    matrix_new2 = matrix[1].t()
    #dim = matrix_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    matrix_new = torch.stack((matrix_new1, matrix_new2), dim=0)
    #matrix_new[0] = matrix_new1; matrix_new[1] = matrix_new2
    return matrix_new

def diag_complex(vector):
    vector_new1 = torch.diag(vector[0])
    vector_new2 = torch.diag(vector[1])
    #dim = vector_new1.size()
    #temp = torch.zeros(dim, dtype=cfg.global_args.dtype, device=cfg.global_args.device)
    vector_new = torch.stack((vector_new1, vector_new2), dim=0)
    #vector_new[0] = vector_new1; vector_new[1] = vector_new2
    return vector_new

def eye_complex(M, dtype, device):
    matrix_new1 = torch.eye(M, dtype, device)
    #dim = matrix_new1.size()
    temp = torch.zeros(dim, dtype, device)
    matrix_new = torch.stack((matrix_new1, temp), dim=0)
    #matrix_new[0] = matrix_new1
    return matrix_new

def max_complex(tensor):
    num = torch.numel(tensor[0])
    tensor1 = torch.flatten(tensor[0])
    tensor2 = torch.flatten(tensor[1])
    max_num = torch.tensor(0.0)
    for i in range(0, num):
        if i == 0:
            max_num = torch.sqrt(tensor1[i]**2 + tensor2[i]**2)
        else:
            if torch.sqrt(tensor1[i]**2 + tensor2[i]**2) > max_num:
                max_num = torch.sqrt(tensor1[i]**2 + tensor2[i]**2)
    return max_num
        
