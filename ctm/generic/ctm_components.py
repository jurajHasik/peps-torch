import torch
from torch.utils.checkpoint import checkpoint
from config import ctm_args
from tn_interface import contract
from tn_interface import view, permute, contiguous, conj

#####################################################################
# functions building pair of 4x2 (or 2x4) halves of 4x4 TN
#####################################################################
def halves_of_4x4_CTM_MOVE_UP(coord, state, env, mode='sl', verbosity=0):
    r"""
    :param coord: site for which to build two halfs of 2x2 subsystem embedded 
                  in environment 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :return: right and left half of the system as matrices
    :rtype: torch.Tensor, torch.Tensor

    Builds right and left half of 2x2 subsystem embedded into environment. 
    The `coord` specifies the upper-right site of the 2x2 subsystem. 
    Performs following contraction and then reshaping the resulting tensors into matrices::

        C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
        T A B(coord) T   C2x2_LD(coord+(-1,1))  C2x2(coord+(0,1))
        T C D        T
        C T T        C

        C2x2--1->0 0--C2x2(coord) =     _0 0_
        |0           1|                |     |
        |0           0|             half2    half1
        C2x2--1    1--C2x2             |_1 1_|
    """
    # RU, RD, LU, LD
    tensors= c2x2_RU_t(coord,state,env) + c2x2_RD_t((coord[0], coord[1]+1),state,env) \
        + c2x2_LU_t((coord[0]-1, coord[1]),state,env) + c2x2_LD_t((coord[0]-1, coord[1]+1),state,env)
    if mode in ['sl']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_UP_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_UP_c(*tensors)

def halves_of_4x4_CTM_MOVE_UP_t(coord, state, env):
    # RU, RD, LU, LD
    tensors= c2x2_RU_t(coord,state,env) + c2x2_RD_t((coord[0], coord[1]+1),state,env) \
        + c2x2_LU_t((coord[0]-1, coord[1]),state,env) + c2x2_LD_t((coord[0]-1, coord[1]+1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_UP_c(*tensors):
    # C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
    # T A B(coord) T   C2x2_LD(coord+(-1,1))  C2x2(coord+(0,1))
    # T C D        T
    # C T T        C

    # C2x2--1->0 0--C2x2(coord) =     _0 0_
    # |0           1|                |     |
    # |0           0|             half2    half1
    # C2x2--1    1--C2x2             |_1 1_|
    
    # C_1, T1_1, T2_1, A_1= tensors[0:4]
    # C_2, T1_2, T2_2, A_2= tensors[4:8]
    # C_3, T1_3, T2_3, A_3= tensors[8:12]
    # C_4, T1_4, T2_4, A_4= tensors[12:16]
    if tensors[-1]: # mode
        return contract(c2x2_RU_sl_c(*tensors[0:4]),c2x2_RD_sl_c(*tensors[4:8]),([1],[0])), \
            contract(c2x2_LU_sl_c(*tensors[8:12]),c2x2_LD_sl_c(*tensors[12:16]),([0],[0]))
    else:
        return contract(c2x2_RU_c(*tensors[0:4]),c2x2_RD_c(*tensors[4:8]),([1],[0])), \
            contract(c2x2_LU_c(*tensors[8:12]),c2x2_LD_c(*tensors[12:16]),([0],[0]))

def halves_of_4x4_CTM_MOVE_LEFT(coord, state, env, mode='sl', verbosity=0):
    r"""
    :param coord: site for which to build two halfs of 2x2 subsystem embedded 
                  in environment 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :return: upper and lower half of the system as matrices
    :rtype: torch.Tensor, torch.Tensor

    Builds upper and lower half of 2x2 subsystem embedded into environment. 
    The `coord` specifies the upper-left site of the 2x2 subsystem. 
    Performs following contraction and then reshaping the resulting tensors into matrices::

        C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
        T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
        T C        D T
        C T        T C

        C2x2(coord)--1 0--C2x2 = half1
        |0               1|      |0  |1
        
        |0            1<-0|      |0  |1
        C2x2--1 1---------C2x2   half2
    """
    # LU, RU, LS, RD
    tensors= c2x2_LU_t(coord,state,env) + c2x2_RU_t((coord[0]+1, coord[1]),state,env) \
        + c2x2_LD_t((coord[0], coord[1]+1),state,env) + c2x2_RD_t((coord[0]+1, coord[1]+1),state,env)
    if mode in ['sl']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_LEFT_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_LEFT_c(*tensors)

def halves_of_4x4_CTM_MOVE_LEFT_t(coord, state, env):
    # LU, RU, LS, RD
    tensors= c2x2_LU_t(coord,state,env) + c2x2_RU_t((coord[0]+1, coord[1]),state,env) \
        + c2x2_LD_t((coord[0], coord[1]+1),state,env) + c2x2_RD_t((coord[0]+1, coord[1]+1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_LEFT_c(*tensors):
    # C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
    # T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
    # T C        D T
    # C T        T C

    # C2x2(coord)--1 0--C2x2 = half1
    # |0               1|      |0  |1
    # 
    # |0            1<-0|      |0  |1
    # C2x2--1 1---------C2x2   half2
    if tensors[-1]: # mode
        return contract(c2x2_LU_sl_c(*tensors[0:4]),c2x2_RU_sl_c(*tensors[4:8]),([1],[0])), \
            contract(c2x2_LD_sl_c(*tensors[8:12]),c2x2_RD_sl_c(*tensors[12:16]),([1],[1]))
    else:   
        return contract(c2x2_LU_c(*tensors[0:4]),c2x2_RU_c(*tensors[4:8]),([1],[0])), \
            contract(c2x2_LD_c(*tensors[8:12]),c2x2_RD_c(*tensors[12:16]),([1],[1]))

def halves_of_4x4_CTM_MOVE_DOWN(coord, state, env, mode='sl', verbosity=0):
    r"""
    :param coord: site for which to build two halfs of 2x2 subsystem embedded 
                  in environment 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :return: left and right half of the system as matrices
    :rtype: torch.Tensor, torch.Tensor

    Builds left and right half of 2x2 subsystem embedded into environment. 
    The `coord` specifies the lower-left site of the 2x2 subsystem. 
    Performs following contraction and then reshaping the resulting tensors into matrices::

        C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
        T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
        T C(coord) D T
        C T        T C

        C2x2---------1    1<-0--C2x2 =     _1 1_
        |0                      |1        |     |
        |0                      |0      half1    half2
        C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|
    """
    # LD, LU, RD, RU
    tensors= c2x2_LD_t(coord,state,env) + c2x2_LU_t((coord[0], coord[1]-1),state,env) \
        + c2x2_RD_t((coord[0]+1, coord[1]),state,env) + c2x2_RU_t((coord[0]+1, coord[1]-1),state,env)
    if mode in ['sl']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_DOWN_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_DOWN_c(*tensors)

def halves_of_4x4_CTM_MOVE_DOWN_t(coord, state, env):
    # LD, LU, RD, RU
    tensors= c2x2_LD_t(coord,state,env) + c2x2_LU_t((coord[0], coord[1]-1),state,env) \
        + c2x2_RD_t((coord[0]+1, coord[1]),state,env) + c2x2_RU_t((coord[0]+1, coord[1]-1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_DOWN_c(*tensors):
    # C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
    # T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
    # T C(coord) D T
    # C T        T C

    # C2x2---------1    1<-0--C2x2 =     _1 1_
    # |0                      |1        |     |
    # |0                      |0      half1    half2
    # C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|
    if tensors[-1]: # mode
        return contract(c2x2_LD_sl_c(*tensors[0:4]),c2x2_LU_sl_c(*tensors[4:8]),([0],[0])), \
            contract(c2x2_RD_sl_c(*tensors[8:12]),c2x2_RU_sl_c(*tensors[12:16]),([0],[1]))
    else:
        return contract(c2x2_LD_c(*tensors[0:4]),c2x2_LU_c(*tensors[4:8]),([0],[0])), \
            contract(c2x2_RD_c(*tensors[8:12]),c2x2_RU_c(*tensors[12:16]),([0],[1]))

def halves_of_4x4_CTM_MOVE_RIGHT(coord, state, env, mode='sl', verbosity=0):
    r"""
    :param coord: site for which to build two halfs of 2x2 subsystem embedded 
                  in environment 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :return: upper and lower half of the system as matrices
    :rtype: torch.Tensor, torch.Tensor

    Builds uoper and lower half of 2x2 subsystem embedded into environment. 
    The `coord` specifies the lower-right site of the 2x2 subsystem. 
    Performs following contraction and then reshaping the resulting tensors into matrices::

        C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
        T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
        T C D(coord) T
        C T T        C

        C2x2--1 0--C2x2        = half2
        |0->1      |1->0         |1  |0
        
        |0->1      |0            |1  |0
        C2x2--1 1--C2x2(coord)   half1
    """
    # RD, LD, RU, LU
    tensors= c2x2_RD_t(coord,state,env) + c2x2_LD_t((coord[0]-1, coord[1]),state,env) \
        + c2x2_RU_t((coord[0], coord[1]-1),state,env) + c2x2_LU_t((coord[0]-1, coord[1]-1),state,env)
    if mode in ['sl']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_halves:
        return checkpoint(halves_of_4x4_CTM_MOVE_RIGHT_c,*tensors)
    else:
        return halves_of_4x4_CTM_MOVE_RIGHT_c(*tensors)

def halves_of_4x4_CTM_MOVE_RIGHT_t(coord, state, env):
    # RD, LD, RU, LU
    tensors= c2x2_RD_t(coord,state,env) + c2x2_LD_t((coord[0]-1, coord[1]),state,env) \
        + c2x2_RU_t((coord[0], coord[1]-1),state,env) + c2x2_LU_t((coord[0]-1, coord[1]-1),state,env)
    return tensors

def halves_of_4x4_CTM_MOVE_RIGHT_c(*tensors):
    # C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
    # T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
    # T C D(coord) T
    # C T T        C

    # C2x2--1 0--C2x2        = half2
    # |0->1      |1->0         |1  |0
    # 
    # |0->1      |0            |1  |0
    # C2x2--1 1--C2x2(coord)   half1
    if tensors[-1]: # mode
        return contract(c2x2_RD_sl_c(*tensors[0:4]),c2x2_LD_sl_c(*tensors[4:8]),([1],[1])), \
            contract(c2x2_RU_sl_c(*tensors[8:12]),c2x2_LU_sl_c(*tensors[12:16]),([0],[1]))
    else:
        return contract(c2x2_RD_c(*tensors[0:4]),c2x2_LD_c(*tensors[4:8]),([1],[1])), \
            contract(c2x2_RU_c(*tensors[8:12]),c2x2_LU_c(*tensors[12:16]),([0],[1]))

#####################################################################
# functions building 2x2 Corner
#####################################################################
def c2x2_LU(coord, state, env, mode='dl', verbosity=0):
    r"""
    :param coord: site for which to build enlarged upper-left corner 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :param mode: single ``'sl'`` or double-layer ``'dl'`` contraction 
    :type mode: str
    :return: enlarged upper-left corner
    :rtype: torch.Tensor

    Builds upper-left corner at site `coord` by performing following
    contraction and then reshaping the resulting tensor into matrix::

        C----T1--2         => C2x2--(2,3)->1
        |     |               |
        |     |               (0,1)->0
        T2----A(coord)--3
        |     |
        0     1
    """
    # tensors= C, T1, T2, A
    tensors= c2x2_LU_t(coord,state,env)

    _f_c2x2= c2x2_LU_c if mode in ['dl','dl-open'] else c2x2_LU_sl_c
    if mode in ['dl-open', 'sl-open']:
        tensors += (torch.ones(1, dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1, dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(_f_c2x2,*tensors)
    else:
        C2x2= _f_c2x2(*tensors)

    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_LU_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(-1,-1))], \
        env.T[(state.vertexToSite(coord),(0,-1))], \
        env.T[(state.vertexToSite(coord),(-1,0))], \
        state.site(coord)
    return tensors

def c2x2_LU_c(*tensors):
    C, T1, T2, A, mode= tensors if len(tensors)==5 else tensors+(None,)
    # flops \chi x (D^2 \chi^2)
    #
    # C--10--T1--2
    # 0      1
    C2x2 = contract(C, T1, ([1],[0]))

    # flops \chi x (D^4 \chi^2)
    #
    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2 = contract(C2x2, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 A--3 
    # 2->1    2
    if not mode:
        # flops D^4 x (D^4 \chi^2)
        #
        C2x2 = contract(C2x2, A, ([0,3],[0,1]))
        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(1)*A.size(2),T1.size(2)*A.size(3)))
    else:
        # flops D^4 x (D^4 \chi^2 p^2)
        #
        # C-------T1--1->0
        # |       0
        # |       2/0->2
        # T2--3 3 A--5->5 
        # 2->1    4\1->3
        #         ->4
        C2x2 = contract(C2x2, A, ([0,3],[2,3]))
        # permute 012345->140523
        # reshape (14)(05)23->0123
        C2x2 = contiguous(permute(C2x2,(1,4,0,5,2,3)))
        C2x2 = view(C2x2,(T2.size(1)*A.size(4),T1.size(2)*A.size(5),\
            A.size(0),A.size(1)))

    # C2x2--1
    # |\
    # 0 optionally(2,3)
    return C2x2

def c2x2_LU_sl_c(*tensors):
    C, T1, T2, a, mode= tensors if len(tensors)==5 else tensors+(None,)
    # flops \chi x (D^2 \chi^2)
    #
    # C--1 0--T1--2
    # 0       1
    C2x2 = contract(C, T1, ([1],[0]))

    # flops \chi x (D^4 \chi^2)
    #
    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2 = contract(C2x2, T2, ([0],[0]))
    C2x2 = view(C2x2, (a.size(1),a.size(1),T1.size(2),\
        T2.size(1),a.size(2),a.size(2)) )

    # flops D^2 x (D^4 \chi^2 p)
    #
    # C---------T1--2->1
    # |         0,1->0
    # |         1
    # T2--4 2---a--4->6 
    # |   5->3  3\0->4
    # 3->2      ->5
    #
    # C---------T1--0
    # |         |
    # |         |
    # T2-------a*a--3,5 
    # |         |    
    # 1         2,4
    C2x2 = contract(C2x2, a, ([0,4],[1,2]))
    if not mode:
        # flops (D^2 p) x (D^4 \chi^2)
        #
        C2x2 = contract(C2x2, conj(a), ([0,3,4],[1,2,0]))
        # permute 012345->124035
        # reshape (124)(035)->01
        C2x2 = contiguous(permute(C2x2,(1,2,4,0,3,5)))
        C2x2 = view(C2x2,(T2.size(1)*(a.size(3)**2),T1.size(2)*(a.size(4)**2)))
    else:
        # flops D^2 x (D^4 \chi^2 p^2)
        #
        # C---------T1--0
        # |         |
        # |         |/2,5
        # T2-------a*a--4,7
        # |         |    
        # 1         3,6
        C2x2 = contract(C2x2, conj(a), ([0,3],[1,2]))
        # permute 01234567->13604725
        # reshape (136)(047)25->0123
        C2x2 = contiguous(permute(C2x2,(1,3,6,0,4,7,2,5)))
        C2x2 = view(C2x2,(T2.size(1)*(a.size(3)**2),T1.size(2)*(a.size(4)**2),
            a.size(0),a.size(0)))

    # C2x2--1
    # |\
    # 0 optionally(2,3)
    return C2x2


def c2x2_RU(coord, state, env, mode='dl', verbosity=0):
    r"""
    :param coord: site for which to build enlarged upper-right corner 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :param mode: single ``'sl'`` or double-layer ``'dl'`` contraction 
    :type mode: str
    :return: enlarged upper-left corner
    :rtype: torch.Tensor

    Builds upper-right corner at site `coord` by performing following
    contraction and then reshaping the resulting tensor into matrix::

            0--T2--------C  =>  0<-(0,1)--C2x2
               |         |                |
               |         |          1<-(2,3)
            1--A(coord)--T1
               3         2
    """
    # tensors= C, T1, T2, A
    tensors= c2x2_RU_t(coord,state,env)

    _f_c2x2= c2x2_RU_c if mode in ['dl','dl-open'] else c2x2_RU_sl_c
    if mode in ['dl-open', 'sl-open']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(_f_c2x2,*tensors)
    else:
        C2x2= _f_c2x2(*tensors)

    if verbosity>0:
        print("C2X2 RU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,-1)")
    if verbosity>1: 
        print(C2x2)

    return C2x2

def c2x2_RU_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(1,-1))], \
        env.T[(state.vertexToSite(coord),(1,0))], \
        env.T[(state.vertexToSite(coord),(0,-1))], \
        state.site(coord)
    return tensors

def c2x2_RU_c(*tensors):
    C, T1, T2, A, mode= tensors if len(tensors)==5 else tensors+(None,)
    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2 = contract(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2 = contract(C2x2, T2, ([0],[2]))

    if not mode:
        # 1<-2--T2------C
        #       3       |
        #       0       |
        # 2<-1--A--3 0--T1
        #    3<-2    0<-1
        C2x2 = contract(C2x2, A, ([0,3],[3,0]))

        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(1),T1.size(2)*A.size(2)))
    else:
        #    1<-2--T2------C
        #          3       |
        # 2,3<-0,1\2       |
        #    4<-3--A--5 0--T1
        #       5<-4    0<-1
        C2x2 = contract(C2x2, A, ([0,3],[5,2]))
        # permute 012345->140523
        # reshape (14)(05)23->0123
        C2x2 = contiguous(permute(C2x2,(1,4,0,5,2,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(3),T1.size(2)*A.size(4),\
            A.size(0),A.size(1)))
 
    # 0--C2x2
    #    |\
    #    1 optionally(2,3)
    return C2x2

def c2x2_RU_sl_c(*tensors):
    C, T1, T2, a, mode= tensors if len(tensors)==5 else tensors+(None,)
    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2 = contract(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2 = contract(C2x2, T2, ([0],[2]))
    C2x2 = view(C2x2, (a.size(4),a.size(4),T1.size(2),T2.size(0),\
        a.size(1),a.size(1)) )

    # 2<-3--T2------C
    #       4,5     |
    #       1 ->3   |
    # 5<-2--a--4 0--T1
    #  4<-0/3 0<-1  | 
    #    6<-     1<-2
    #
    #    1--T2------C
    #       |       |
    #       |       |
    #  2,4-a*a------T1
    #       |       | 
    #      3,5      0
    C2x2 = contract(C2x2, a, ([0,4],[4,1]))
    if not mode:
        C2x2 = contract(C2x2, conj(a), ([0,3,4],[4,1,0]))
        # permute 012345->124035
        # reshape (124)(035)->01
        C2x2 = contiguous(permute(C2x2,(1,2,4,0,3,5)))
        C2x2 = view(C2x2,(T2.size(0)*(a.size(2)**2),T1.size(2)*(a.size(3)**2)))
    else:
        #    1--T2------C
        #       |       |
        #   2,5\|       |
        #  3,6-a*a------T1
        #       |       | 
        #      4,7      0
        C2x2 = contract(C2x2, conj(a), ([0,3],[4,1]))
        # permute 01234567->12603745
        # reshape (136)(047)25->0123
        C2x2 = contiguous(permute(C2x2,(1,3,6,0,4,7,2,5)))
        C2x2 = view(C2x2,(T2.size(0)*(a.size(2)**2),T1.size(2)*(a.size(3)**2),
            a.size(0),a.size(0)))
     
    # 0--C2x2
    #    |\
    #    1 optionally(2,3)
    return C2x2


def c2x2_RD(coord, state, env, mode='dl', verbosity=0):
    r"""
    :param coord: site for which to build enlarged lower-right corner 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :param mode: single ``'sl'`` or double-layer ``'dl'`` contraction 
    :type mode: str
    :return: enlarged upper-left corner
    :rtype: torch.Tensor

    Builds lower-right corner at site `coord` by performing following
    contraction and then reshaping the resulting tensor into matrix::

               1         0 
            3--A(coord)--T2 =>      0<-(0,1)
               |         |                |
               |         |      1<-(2,3)--C2x2
            2--T1--------C
    """
    # tensors= C, T1, T2, A
    tensors= c2x2_RD_t(coord,state,env)

    _f_c2x2= c2x2_RD_c if mode in ['dl','dl-open'] else c2x2_RD_sl_c
    if mode in ['dl-open', 'sl-open']:
        tensors += (torch.ones(1, dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1, dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(_f_c2x2,*tensors)
    else:
        C2x2= _f_c2x2(*tensors)

    if verbosity>0:
        print("C2X2 RD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_RD_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(1,1))], \
        env.T[(state.vertexToSite(coord),(0,1))], \
        env.T[(state.vertexToSite(coord),(1,0))], \
        state.site(coord)
    return tensors

def c2x2_RD_c(*tensors):
    C, T1, T2, A, mode= tensors if len(tensors)==5 else tensors+(None,)
    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2 = contract(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2 = contract(C2x2, T2, ([0],[2]))

    #    2<-0    1<-2
    # 3<-1--A--3 3--T2
    #       2       |
    #       0       |
    # 0<-1--T1------C
    if not mode:
        C2x2 = contract(C2x2, A, ([0,3],[2,3]))

        # permute 0123->1203
        # reshape (12)(03)->01
        C2x2 = contiguous(permute(C2x2,(1,2,0,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(0),T1.size(1)*A.size(1)))
    else:
        #       4<-2    1<-2
        #    5<-3--A--5 3--T2
        # 2,3<-0,1/4       |
        #          0       |
        #    0<-1--T1------C
        C2x2 = contract(C2x2, A, ([0,3],[4,5]))

        # permute 012345->140523
        # reshape (14)(05)23->0123
        C2x2 = contiguous(permute(C2x2,(1,4,0,5,2,3)))
        C2x2 = view(C2x2,(T2.size(0)*A.size(2),T1.size(1)*A.size(3),\
            A.size(0),A.size(1)))

    #    0 optionally(2,3)
    #    |/
    # 1--C2x2
    return C2x2

def c2x2_RD_sl_c(*tensors):
    C, T1, T2, a, mode= tensors if len(tensors)==5 else tensors+(None,)
    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2 = contract(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2 = contract(C2x2, T2, ([0],[2]))
    C2x2 = view(C2x2, (a.size(3),a.size(3),T1.size(1),\
        T2.size(0), a.size(4), a.size(4)) )

    #    5<-1    2<-3
    # 6<-2--a--4 4--T2
    #  4<-0/3 3<-5  |
    #       0,1->0  |
    # 1<-2--T1------C
    #
    #       2,4     1
    #  3,5-a*a------T2
    #       |       |
    #       |       |
    #    0--T1------C
    C2x2 = contract(C2x2, a, ([0,4],[3,4]))
    if not mode:
        C2x2 = contract(C2x2, conj(a), ([0,3,4],[3,4,0]))
        # permute 012345->124035
        # reshape (124)(035)->01
        C2x2 = contiguous(permute(C2x2,(1,2,4,0,3,5)))
        C2x2 = view(C2x2,(T2.size(0)*(a.size(1)**2),T1.size(1)*(a.size(2)**2)))
    else:
        #       3,6     1
        #  4,7-a*a------T2
        #   2,5/|       |
        #       |       |
        #    0--T1------C
        C2x2 = contract(C2x2, conj(a), ([0,3],[3,4]))
        # permute 01234567->13604725
        # reshape (136)(047)25->0123
        C2x2 = contiguous(permute(C2x2,(1,3,6,0,4,7,2,5)))
        C2x2 = view(C2x2,(T2.size(0)*(a.size(1)**2),T1.size(1)*(a.size(2)**2),
            a.size(0),a.size(0)))


    #    0 optionally(2,3)
    #    |/
    # 1--C2x2
    return C2x2


def c2x2_LD(coord, state, env, mode='dl', verbosity=0):
    r"""
    :param coord: site for which to build enlarged lower-right corner 
    :type coord: tuple(int,int)
    :param state: wavefunction
    :type state: IPEPS
    :param env: environment
    :type env: ENV
    :param mode: single ``'sl'`` or double-layer ``'dl'`` contraction 
    :type mode: str
    :return: enlarged upper-left corner
    :rtype: torch.Tensor

    Builds lower-right corner at site `coord` by performing following
    contraction and then reshaping the resulting tensor into matrix::

        0   1
        T1--A(coord)--3
        |   |                (0,1)->0(+)
        |   |                 |
        C---T2--------2  =>   C2x2--(2,3)->1(-)
    """
    #tensors= C, T1, T2, A
    tensors= c2x2_LD_t(coord,state,env)

    _f_c2x2= c2x2_LD_c if mode in ['dl','dl-open'] else c2x2_LD_sl_c
    if mode in ['dl-open', 'sl-open']:
        tensors += (torch.ones(1,dtype=torch.bool),)
    else:
        tensors += (torch.zeros(1,dtype=torch.bool),)

    if ctm_args.fwd_checkpoint_c2x2:
        C2x2= checkpoint(_f_c2x2,*tensors)
    else:
        C2x2= _f_c2x2(*tensors)

    if verbosity>0: 
        print("C2X2 LD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_LD_t(coord, state, env):
    tensors= env.C[(state.vertexToSite(coord),(-1,1))], \
        env.T[(state.vertexToSite(coord),(-1,0))], \
        env.T[(state.vertexToSite(coord),(0,1))], \
        state.site(coord)
    return tensors

def c2x2_LD_c(*tensors):
    C, T1, T2, A, mode= tensors if len(tensors)==5 else tensors+(None,)
    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2 = contract(C, T1, ([0],[1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T2--2->3
    C2x2 = contract(C2x2, T2, ([0],[1]))

    # 0        0->2
    # T1--1 1--A--3
    # |        2
    # |        2
    # C--------T2--3->1
    if not mode:
        C2x2 = contract(C2x2, A, ([1,2],[1,2]))

        # permute 0123->0213
        # reshape (02)(13)->01
        C2x2 = contiguous(permute(C2x2,(0,2,1,3)))
        C2x2 = view(C2x2,(T1.size(0)*A.size(0),T2.size(2)*A.size(3)))
    else:
        # 0        2->4
        # T1--1 3--A--5
        # |        4\0,1->2,3
        # |        2
        # C--------T2--3->1
        C2x2 = contract(C2x2, A, ([1,2],[3,4]))
        # permute 012345->041523
        # reshape (04)(15)23->0123
        C2x2 = contiguous(permute(C2x2,(0,4,1,5,2,3)))
        C2x2 = view(C2x2,(T1.size(0)*A.size(2),T2.size(2)*A.size(5),
            A.size(0),A.size(1)))

    # 0 optionally(2,3)
    # |/
    # C2x2--1
    return C2x2

def c2x2_LD_sl_c(*tensors):
    C, T1, T2, a, mode= tensors if len(tensors)==5 else tensors+(None,)
    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2 = contract(C, T1, ([0],[1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T2--2->3
    C2x2 = contract(C2x2, T2, ([0],[1]))
    C2x2 = view(C2x2, (T1.size(0),a.size(2),a.size(2),
        a.size(3),a.size(3),T2.size(2)) )

    # 0        1->5
    # T1--1 2--a--4->6
    # |   2->1 3\0->4
    # |        3,4->2
    # C--------T2--5->3
    #
    # 0        2,4
    # T1------a*a--3,5
    # |        |
    # |        |
    # C--------T2--1
    C2x2 = contract(C2x2, a, ([1,3],[2,3]))
    if not mode:
        C2x2 = contract(C2x2, conj(a), ([1,2,4],[2,3,0]))
        # permute 012345->024135
        # reshape (024)(135)->01
        C2x2 = contiguous(permute(C2x2,(0,2,4,1,3,5)))
        C2x2 = view(C2x2,(T1.size(0)*(a.size(1)**2),T2.size(2)*(a.size(4)**2)))
    else:
        # 0        3,6
        # T1------a*a--4,7
        # |        |\2,5
        # |        |
        # C--------T2--1
        C2x2 = contract(C2x2, conj(a), ([1,2],[2,3]))
        # permute 01234567->03614725
        # reshape (036)(147)25->0123
        C2x2 = contiguous(permute(C2x2,(0,3,6,1,4,7,2,5)))
        C2x2 = view(C2x2,(T1.size(0)*(a.size(1)**2),T2.size(2)*(a.size(4)**2),
            a.size(0),a.size(0)))


    # 0 optionally(2,3)
    # |/
    # C2x2--1
    return C2x2