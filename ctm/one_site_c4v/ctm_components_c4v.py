import torch
import logging
log = logging.getLogger(__name__)

def _log_cuda_mem(device, who="unknown", uuid=""):
    log.info(f"{who} {uuid} GPU-MEM MAX_ALLOC {torch.cuda.max_memory_allocated(device)}"\
            + f" CURRENT_ALLOC {torch.cuda.memory_allocated(device)}")

def c2x2_dl(A, C, T, verbosity=0):
    who= "c2x2_dl"
    loc_device=A.device
    is_cpu= loc_device==torch.device('cpu')
    log_gpu_mem= False
    if not is_cpu and verbosity>0: log_gpu_mem=True

    #      ------>
    # C--1 1--T--0->1
    # 0       2
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CT_init")
    C2x2 = torch.tensordot(C, T, ([1],[1]))
    # C2x2= torch.diag(C).view(-1,1,1)*T
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CT_end")

    #        ---->
    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CTT_init")
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CTT_end")

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3 
    # 2->1    2
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CTTA_init")
    C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))
    if log_gpu_mem: _log_cuda_mem(loc_device, who=who, uuid="CTTA_end")

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T.size(1)*A.size(3),T.size(1)*A.size(2))

    # C2x2--1
    # |
    # 0
    return C2x2

def c2x2_sl(a, C, T, verbosity=0):
    #      ------>      
    # C--1 1--T--0->1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[1]))
    # C2x2= torch.diag(C).view(-1,1,1)*T
    #        
    #   C------T--1->0
    #   0      2->1
    # A 0
    # | T--2->3
    # | 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3
    # 2->1    2
    # C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))

    # 4) double-layer tensor contraction - layer by layer
    # 4i) untangle the fused D^2 indices
    # 
    # C-----T--0
    # |     1->1,2
    # |  
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view([C2x2.size(0),a.size(1),a.size(1),C2x2.size(2),\
        a.size(2),a.size(2)])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # C---------T----0
    # |         |\---2->1
    # |         1    
    # |         1 /0->4
    # T----4 2--a--4->6 
    # | |       3->5
    # |  --5->3
    # 3->2
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))

    # 4iii) second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |/4 0\  |
    # T----a----------6->3 
    # | |  |      \1
    # |  -----3 2--a--4->5
    # |    |       3->4
    # |    |
    # 2->1 5->2
    C2x2= torch.tensordot(C2x2, a.conj(),([1,3,4],[1,2,0]))

    # 4iv) fuse pairs of aux indices
    #
    # C----T----0
    # |    |\
    # T----a----3\ 
    # | |  |\|    ->3
    # |  ----a--5/
    # |    | |
    # 1   (2 4)->2
    # 
    # and simultaneously
    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2= C2x2.permute([1,2,4,0,3,5]).contiguous()\
        .view([C2x2.size(1)*a.size(3)*a.size(3),C2x2.size(0)*a.size(4)*a.size(4)])

    # C2x2--1
    # |
    # 0
    if verbosity>1: print(C2x2)

    return C2x2