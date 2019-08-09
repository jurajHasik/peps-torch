import torch
from ipeps import IPEPS
from ctm.generic.env import ENV
from config import ctm_args

#####################################################################
# functions building pair of 4x2 (or 2x4) halves of 4x4 TN
#####################################################################
def halves_of_4x4_CTM_MOVE_UP(coord, ipeps, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(-1,0))  C2x2(coord)
    # T A B(coord) T   C2x2_LD(coord+(-1,-1)) C2x2(coord+(0,1))
    # T C D        T
    # C T T        C

    # C2x2--1->0 0--C2x2(coord) =     _0 0_
    # |0           1|                |     |
    # |0           0|             half2    half1
    # C2x2--1    1--C2x2             |_1 1_|
    C2x2_1 = c2x2_RU(coord, ipeps, env, verbosity)
    C2x2_2 = c2x2_RD((coord[0], coord[1]+1), ipeps, env, verbosity)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[0]))

    C2x2_1 = c2x2_LU((coord[0]-1, coord[1]), ipeps, env, verbosity)
    C2x2_2 = c2x2_LD((coord[0]-1, coord[1]-1), ipeps, env, verbosity)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[0]))

    if verbosity>0:
        print("HALVES UP "+str(coord)+" h1: "+str(half1.size())+" h2: "+str(half2.size()))
    if verbosity>1: 
        print(half1)
        print(half2)

    return half1, half2

def halves_of_4x4_CTM_MOVE_LEFT(coord, ipeps, env, verbosity=0):
    # C T        T C = C2x2_LU(coord)       C2x2(coord+(1,0))
    # T A(coord) B T   C2x2_LD(coord+(0,1)) C2x2(coord+(1,1))
    # T C        D T
    # C T        T C

    # C2x2(coord)--1 0--C2x2 = half1
    # |0               1|      |0  |1
    # 
    # |0            1<-0|      |0  |1
    # C2x2--1 1---------C2x2   half2
    C2x2_1 = c2x2_LU(coord, ipeps, env, verbosity)
    C2x2_2 = c2x2_RU((coord[0]+1, coord[1]), ipeps, env, verbosity)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[0]))

    C2x2_1 = c2x2_LD((coord[0], coord[1]+1), ipeps, env, verbosity)
    C2x2_2 = c2x2_RD((coord[0]+1, coord[1]+1), ipeps, env, verbosity)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([1],[1]))

    if verbosity>0:
        print("HALVES LEFT "+str(coord)+" h1: "+str(half1.size())+" h2: "+str(half2.size()))
    if verbosity>1:
        print(half1)
        print(half2)

    return half1, half2

def halves_of_4x4_CTM_MOVE_DOWN(coord, ipeps, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(0,-1)) C2x2(coord+(1,-1))
    # T A        B T   C2x2_LD(coord)        C2x2(coord+(1,0))
    # T C(coord) D T
    # C T        T C

    # C2x2---------1    1<-0--C2x2 =     _1 1_
    # |0                      |1        |     |
    # |0                      |0      half1    half2
    # C2x2(coord)--1->0 0<-1--C2x2      |_0 0_|
    C2x2_1 = c2x2_LD(coord, ipeps, env, verbosity)
    C2x2_2 = c2x2_LU((coord[0], coord[1]-1), ipeps, env, verbosity)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([0],[0]))

    C2x2_1 = c2x2_RD((coord[0]+1, coord[1]), ipeps, env, verbosity)
    C2x2_2 = c2x2_RU((coord[0]+1, coord[1]-1), ipeps, env, verbosity)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[1]))

    if verbosity==1:
        print("HALVES DOWN "+str(coord)+" h1: "+str(half1.size())+" h2: "+str(half2.size()))
    if verbosity==2:    
        print(half1)
        print(half2)

    return half1, half2

def halves_of_4x4_CTM_MOVE_RIGHT(coord, ipeps, env, verbosity=0):
    # C T T        C = C2x2_LU(coord+(-1,-1)) C2x2(coord+(0,-1))
    # T A B        T   C2x2_LD(coord+(-1,0))  C2x2(coord)
    # T C D(coord) T
    # C T T        C

    # C2x2--1 0--C2x2        = half2
    # |0->1      |1->0         |1  |0
    # 
    # |0->1      |0            |1  |0
    # C2x2--1 1--C2x2(coord)   half1
    C2x2_1 = c2x2_RD(coord, ipeps, env, verbosity)
    C2x2_2 = c2x2_LD((coord[0]-1, coord[1]), ipeps, env, verbosity)
    half1 = torch.tensordot(C2x2_1,C2x2_2,([1],[1]))

    C2x2_1 = c2x2_RU((coord[0], coord[1]-1), ipeps, env, verbosity)
    C2x2_2 = c2x2_LU((coord[0]-1, coord[1]-1), ipeps, env, verbosity)
    half2 = torch.tensordot(C2x2_1,C2x2_2,([0],[1]))

    if verbosity==1:
        print("HALVES RIGHT "+str(coord)+" h1: "+str(half1.size())+" h2: "+str(half2.size()))
    if verbosity==2:
        print(half1)
        print(half2)

    return half1, half2

#####################################################################
# functions building 2x2 Corner
#####################################################################
def c2x2_LU(coord, ipeps, env, verbosity=0):
    C = env.C[(ipeps.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(0,-1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(-1,0))]
    A = ipeps.site(coord)

    # C--10--T1--2
    # 0      1
    C2x2 = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 A--3 
    # 2->1    2
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[0,1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T1.size()[2]*A.size()[3],T2.size()[1]*A.size()[2])

    # C2x2--1
    # |
    # 0
    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,-1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_RU(coord, ipeps, env, verbosity=0):
    C = env.C[(ipeps.vertexToSite(coord),(1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(1,0))]
    T2 = env.T[(ipeps.vertexToSite(coord),(0,-1))]
    A = ipeps.site(coord)

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2 = torch.tensordot(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2 = torch.tensordot(C2x2, T2, ([0],[2]))

    # 1<-2--T2------C
    #       3       |
    #       0       |
    # 2<-1--A--3 0--T1
    #    3<-2    0<-1
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[3,0]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T2.size()[0]*A.size()[1],T1.size()[2]*A.size()[2])
 
    # 0--C2x2
    #    |
    #    1
    if verbosity>0:
        print("C2X2 RU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (1,-1)")
    if verbosity>1: 
        print(C2x2)

    return C2x2

def c2x2_RD(coord, ipeps, env, verbosity=0):
    C = env.C[(ipeps.vertexToSite(coord),(1,1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(0,1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(1,0))]
    A = ipeps.site(coord)

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2 = torch.tensordot(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2 = torch.tensordot(C2x2, T2, ([0],[2]))

    #    2<-0    1<-2
    # 3<-1--A--3 3--T2
    #       2       |
    #       0       |
    # 0<-1--T1------C
    C2x2 = torch.tensordot(C2x2, A, ([0,3],[2,3]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2 = C2x2.permute(1,2,0,3).contiguous().view(T2.size()[0]*A.size()[0],T1.size()[1]*A.size()[1])

    #    0
    #    |
    # 1--C2x2
    if verbosity>0:
        print("C2X2 RD "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2

def c2x2_LD(coord, ipeps, env, verbosity=0):
    C = env.C[(ipeps.vertexToSite(coord),(-1,1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(-1,0))]
    T2 = env.T[(ipeps.vertexToSite(coord),(0,1))]
    A = ipeps.site(coord)

    tensors= C, T1, T2, A
    def c2x2_LD_c(*tensors):
        C, T1, T2, A= tensors
        # 0->1
        # T1--2
        # 1
        # 0
        # C--1->0
        C2x2 = torch.tensordot(C, T1, ([0],[1]))

        # 1->0
        # T1--2->1
        # |
        # |       0->2
        # C--0 1--T1--2->3
        C2x2 = torch.tensordot(C2x2, T2, ([0],[1]))

        # 0       0->2
        # T1--1 1--A--3
        # |        2
        # |        2
        # C--------T2--3->1
        C2x2 = torch.tensordot(C2x2, A, ([1,2],[1,2]))

        # permute 0123->0213
        # reshape (02)(13)->01
        C2x2 = C2x2.permute(0,2,1,3).contiguous().view(T1.size()[0]*A.size()[0],T2.size()[2]*A.size()[3])

        # 0
        # |
        # C2x2--1
        return C2x2

    if verbosity>0: 
        print("C2X2 LD "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,1)")
    if verbosity>1:
        print(C2x2)

    return C2x2