import torch
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV
from ctm.generic.ctm_components import c2x2_LU_c, c2x2_RU_c, c2x2_RD_c, c2x2_LD_c, \
    c2x2_LU_t, c2x2_RU_t, c2x2_RD_t, c2x2_LD_t

#####################################################################
# functions building 2x2 Corner
#####################################################################
def _get_partial_C2x2_LU(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(0,-1))]
    T2 = env.T[(state.vertexToSite(coord),(-1,0))]
    a = state.site(coord)

    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,-1)")
    if verbosity>1:
        print(C2x2)

    # C--10--T1--2
    # 0      1
    C2x2 = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T2, ([0],[0]))

    # 4i) untangle the fused D^2 indices
    #
    # C------T1--1->2
    # 0      0->0,1
    # 0
    # T2--3->4,5
    # 2->3
    C2x2= C2x2.view(a.size()[1],a.size()[1],C2x2.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # C---------T1---2->1
    # |         |\---1->0
    # |         0    
    # |         1 /0->4
    # T2---4 2--a--4->6
    # | |       3->5
    # |  --5->3
    # 3->2
    C2x2= torch.tensordot(C2x2, a,([0,4],[1,2]))


    # 4iv) fuse pairs of aux indices
    #
    #  C------T1---1    C----T1--1
    #  |      |\---0    |   /|
    #  |      |         |/-|-|---3
    #  T2-----a----6 => T2---a
    #  | \    |\        |  |  \
    #  2 3    5 4       0  2   4
    # 
    # permute and reshape 0123456->(25)(16)034 ->01234
    C2x2= C2x2.permute(2,5,1,6,0,3,4).contiguous().view(C2x2.size()[2]*a.size()[3],\
        C2x2.size()[1]*a.size()[4],a.size()[1],a.size()[2],a.size()[0])
    return C2x2

def _get_partial_C2x2_RU(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(1,0))]
    T2 = env.T[(state.vertexToSite(coord),(0,-1))]
    a = state.site(coord)

    if verbosity>0:
        print("C2X2 RU "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,-1)")
    if verbosity>1: 
        print(C2x2)

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

    # 4i) untangle the fused D^2 indices
    #
    # 3<-2--T2------C
    #  4,5<-3       |
    #       0,1<-0--T1
    #            2<-1
    C2x2= C2x2.view(a.size()[4],a.size()[4],C2x2.size()[1],C2x2.size()[2],a.size()[1],\
        a.size()[1])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    #  2<-3--T2--------C
    # 3<-5--/|         |
    #        4         |
    #  4<-0\ 1         |
    #  5<-2--a--4 0----T1
    #     6<-3     /   |
    #          0<-1 1<-2
    C2x2= torch.tensordot(C2x2, a,([0,4],[4,1]))

    # 4iv) fuse pairs of aux indices
    #
    #  0--T2---C      2--T2--------C
    #     |\   |  <= 3--/|         |
    #     | |  |         |         |
    #  2--|-|-\|      4\ |         |
    #     a----T1     5--a---------T1
    #    /  |  |         6     0--/|
    #   4   3  1                   1
    # 
    # permute and reshape 0123456->(25)(16)034 ->01234
    C2x2= C2x2.permute(2,5,1,6,0,3,4).contiguous().view(C2x2.size()[2]*a.size()[2],\
        C2x2.size()[1]*a.size()[3],a.size()[4],a.size()[1],a.size()[0])
    return C2x2

def _get_partial_C2x2_RD(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(1,1))]
    T1 = env.T[(state.vertexToSite(coord),(0,1))]
    T2 = env.T[(state.vertexToSite(coord),(1,0))]
    a = state.site(coord)

    if verbosity>0:
        print("C2X2 RD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (1,1)")
    if verbosity>1:
        print(C2x2)

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2 = torch.tensordot(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2 = torch.tensordot(C2x2, T2, ([0],[2]))

    # 4i) untangle the fused D^2 indices
    #
    #           3<-2
    #      4,5<-3--T2
    #              |
    #    0,1<-0    |
    #   2<-1--T1---C
    C2x2= C2x2.view(a.size()[3],a.size()[3],C2x2.size()[1],C2x2.size()[2],a.size()[4],\
        a.size()[4])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    #    5<-1      2<-3
    # 4<-0\ |   3<-5\ |
    # 6<-2--a--4 4----T2
    #       3         |
    #  0<-1 0         |
    #      \|         |
    # 1<-2--T1--------C
    C2x2= torch.tensordot(C2x2, a,([0,4],[3,4]))

    # 4iv) fuse pairs of aux indices
    #
    #       2   0        5         2
    #   4\  |   |     4\ |      3\ |
    #     a-----T2    6--a---------T2
    #  3--|-|--/|        |         |
    #     | |   |  <=  0 |         |
    #     |/    |       \|         |
    #  1--T1----C     1--T1--------C
    # 
    # permute and reshape 0123456->(25)(16)034 ->01234
    C2x2= C2x2.permute(2,5,1,6,0,3,4).contiguous().view(C2x2.size()[2]*a.size()[1],\
        C2x2.size()[1]*a.size()[2],a.size()[4],a.size()[3],a.size()[0])
    return C2x2

def _get_partial_C2x2_LD(coord, state, env, verbosity=0):
    C = env.C[(state.vertexToSite(coord),(-1,1))]
    T1 = env.T[(state.vertexToSite(coord),(-1,0))]
    T2 = env.T[(state.vertexToSite(coord),(0,1))]
    a = state.site(coord)

    if verbosity>0: 
        print("C2X2 LD "+str(coord)+"->"+str(state.vertexToSite(coord))+" (-1,1)")
    if verbosity>1:
        print(C2x2)

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
    # C--0 1--T2--2->3
    C2x2 = torch.tensordot(C2x2, T2, ([0],[1]))

    # 4i) untangle the fused D^2 indices
    #
    # 0
    # T1--1->1,2
    # |
    # |    2->3,4
    # C----T2--3->5
    C2x2= C2x2.view(C2x2.size()[0],a.size()[2],a.size()[2],a.size()[3],a.size()[3],\
        C2x2.size()[3])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # 0        1->5
    # |/2->1   |/0->4
    # T1--1 2--a--4->6
    # |        3
    # |        3/--4->2
    # C--------T2--5->3
    C2x2= torch.tensordot(C2x2, a,([1,3],[2,3]))

    # 4iv) fuse pairs of aux indices
    #
    # 0     5       0   2   4
    # |/1   |/4     |   |  /
    # T1----a--6 => T1----a
    # |     |       |\--|-|--3
    # |     |/--2   |    \|
    # C-----T2--3   C-----T2--1
    # 
    # permute and reshape 0123456->(05)(36)214 ->01234
    C2x2= C2x2.permute(0,5,3,6,2,1,4).contiguous().view(C2x2.size()[0]*a.size()[1],\
        C2x2.size()[3]*a.size()[4],a.size()[2],a.size()[3],a.size()[0])
    return C2x2

#
# various partial and auxiliary RDMs
#
def partial_rdm2x2(coord, state, env, force_cpu=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type env: ENV
    :type force_cpu: bool
    :type verbosity: int
    :return: 4-site partial reduced density matrix
    :rtype: torch.tensor

    Computes 4-site partial reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem 
    without the "ket" tensors (leaving corresponding aux indices open)
    using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           partial density matrix
           
    TODO try single torch.einsum ?

    :: 
             0                  3
             |                  |
          C--T------------------T------------------C    = C2x2_LU(coord)--------C2x2(coord+(1,0))
          |  |/2                |/5                |      |                     |
       1--T--a^+(coord)---------a^+(coord+(1,0))---T--4   C2x2_LD(coord+(0,1))--C2x2(coord+(1,1))
          |  |/8                |/11               |
       6--T--a^+(coord+(0,1))---a^+(coord+(1,1))---T--10
          |  |                  |                  |
          C--T------------------T------------------C
             |                  |
             7                  9

    The physical indices `s` of on-sites tensors :math:`a^\dagger` 
    are left uncontracted and given in the following order::
        
        s0 s1
        s2 s3

    """
    #----- building pC2x2_LU -----------------------------------------------------
    loc_device=env.device
    is_cpu= loc_device==torch.device('cpu')
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    upper_half= _get_partial_C2x2_LU(coord, state, env, verbosity=verbosity)
    vec= (1,0)
    pC2x2_RU= _get_partial_C2x2_RU((coord[0]+vec[0],coord[1]+vec[1]), state, env, verbosity=verbosity)

    #----- build upper part pC2x2_LU--pC2x2_RU -----------------------------------
    # pC2x2_LU--1 0---pC2x2_RU
    # |___|\         /|___|
    # | | \ 3->2 5<-2 / | |
    # 0 2  4->3   7<-4  3 1
    #   V               V V
    #   1               6 4
    upper_half = torch.tensordot(upper_half, pC2x2_RU, ([1],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    vec= (0,1)
    lower_half= _get_partial_C2x2_LD((coord[0]+vec[0],coord[1]+vec[1]), state, env, verbosity=verbosity)
    vec= (1,1)
    pC2x2_RD= _get_partial_C2x2_RD((coord[0]+vec[0],coord[1]+vec[1]), state, env, verbosity=verbosity)

    #----- build upper part pC2x2_LD--pC2x2_RD -----------------------------------
    #
    #   1                5 4 
    #   ^                ^ ^       
    # 0 2  4->3    7<-4  2 0
    # |_|_/ 3->2  6<-3 \_|_|
    # |   |/          \|   |
    # pC2x2_LD----1 1--pC2x2_RD
    lower_half= torch.tensordot(lower_half, pC2x2_RD, ([1],[1]))

    # construct reduced density matrix by contracting lower and upper halfs
    #  __________________         __________________
    # |____upper_half____|       |__________________|
    # | | | \      / | | |       | | | \      / | | |
    # 0 1 2  3    7  5 6 4       | 0 1  2    5  3 4 |
    # 0                  4  =>   |                  |
    # | 1 2  3    7  6 5 |       | 6 7  8   11 10 9 |
    # |_|_|_/______\_|_|_|       |_|_|_/______\_|_|_|
    # |____lower_half____|       |____upper_half____|
    upper_half= torch.tensordot(upper_half,lower_half,([0,4],[0,4]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute such, that the index of aux indices for every on-site tensor
    # increases from "up" in the anti-clockwise
    # 01234567891011->01243576891011
    # and normalize
    upper_half = upper_half.permute(0,1,2,4,3,5,7,6,8,9,10,11).contiguous()
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    return upper_half

def fidelity_rdm2x2(coord, state, prdm0, force_cpu=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: 1-site C4v symmetric wavefunction
    :param prdm0: partial reduced density matrix (without ket part) of 2x2 subsystem
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type prdm0: torch.tensor
    :type force_cpu: bool
    :type verbosity: int
    :return: fidelity
    :rtype: torch.tensor

    Contracts 4-site partial reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem 
    with the 2x2 "ket" part given by state on-site tensors

    :: 
        prmd0

             0                3
             |                |
          C--T----------------T------------------C
          |  |/2              |/5                |
       1--T--a^+(coord)-------a^+(coord+(1,0))---T--4
          |  |/8              |/11               | 
       6--T--a^+(coord+(0,1)--a^+(coord+(1,1))---T--10
          |  |                |                  |
          C--T----------------T------------------C
             |                |
             7                9

    """
    loc_device=prdm0.device
    is_cpu= loc_device==torch.device('cpu')

    # contract prdm0 with the upper half of 2x2 section of state ipeps
    # 
    #      0              0                       0    4
    #    1/             1/                      1/   5/
    # 2--a(coord)--4 2--a(coord+(1,0))--4 => 2--a----a--7
    #    3              3                       3    6
    vec0=(1,0)
    aa_upper= torch.tensordot(state.site(coord),\
        state.site((coord[0]+vec0[0],coord[1]+vec0[1])),([4],[2]))

    # contract aa with prmd0
    #
    #                     C--T-------T-------C
    #  _3_____6__         T--a^+a----a^+a----T
    # |____aa____|        |  |   \   |   \   |
    # 1 2 0 5 7 4         |  |/2  6  |/5  7  |
    # 0_1_2_3_4_5_     0--T--a^+-----a^+-----T--4
    # |__prdm0____| =>    C--T-------T-------C
    # 6 7 8 9 10 11          1       3
    fid= torch.tensordot(prdm0,aa_upper,([0,1,2,3,4,5],[1,2,0,5,7,4]))

    # contract prdm0 with the lower half of 2x2 section of state ipeps
    # 
    #      0              0                       0    4
    #    1/             1/                      1/   5/
    # 2--a(coord)--4 2--a(coord+(1,0))--4 => 2--a----a--7
    #    3              3                       3    6
    vec0=(0,1)
    vec1=(1,1)
    aa_upper= torch.tensordot(state.site((coord[0]+vec0[0],coord[1]+vec0[1])),\
        state.site((coord[0]+vec1[0],coord[1]+vec1[1])),([4],[2]))

    # contract rest of the 2x2 section
    #  _____________
    # |_____aa______|
    # 2 3 0 6 7 4 1 5
    # 0_1_2_3_4_5_6_7
    # |____fid______|
    fid= torch.tensordot(fid,aa_upper,([0,1,2,3,4,5,6,7],[2,3,0,6,7,4,1,5]))
    return fid

def aux_rdm1x1(coord, state, env, verbosity=0):
    C1 = env.C[(state.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(0,-1))]
    C2 = env.C[(state.vertexToSite(coord),(1,-1))]
    T2 = env.T[(state.vertexToSite(coord),(1,0))]
    C3 = env.C[(state.vertexToSite(coord),(1,1))]
    T3 = env.T[(state.vertexToSite(coord),(0,1))]
    C4 = env.C[(state.vertexToSite(coord),(-1,1))]
    T4 = env.T[(state.vertexToSite(coord),(-1,0))]
    a= state.site(coord)
    dimsa = a.size()

    # C1--1->0
    # 0
    # 0
    # T4--2
    # 1
    CTC = torch.tensordot(C1,T4,([0],[0]))
    # C1--0
    # |
    # T4--2->1
    # 1
    # 0
    # C4--1->2
    CTC = torch.tensordot(CTC,C4,([1],[0]))
    # C1--0
    # |
    # T4--1
    # |        0->2
    # C4--2 1--T3--2->3
    CTC = torch.tensordot(CTC,T3,([2],[1]))
    
    # 1<-0--T1--2 0--C2
    #    2<-1     0<-1
    CTC2 = torch.tensordot(C2,T1,([0],[2]))
    # 0<-1--T1-------C2
    #    1<-2        0
    #                0
    #          2<-1--T2
    #             3<-2  
    CTC2 = torch.tensordot(CTC2,T2,([0],[0]))
    # 0--T1-------C2
    #    1        |
    #          2--T2
    #             3 
    #             0
    #       3<-1--C3
    CTC2 = torch.tensordot(CTC2,C3,([3],[0]))

    # C1--0 0--T2----C2
    # |     2<-1     |     C----T----C
    # |              |     |    0    |
    # T4--1->0 3<-2--T2 => T--1   3--T
    # |              |     |    2    |
    # |   2->1       |     C----T----C
    # C4--T3--3 3----C3
    rdm= torch.tensordot(CTC,CTC2,([0,3],[0,3]))
    if verbosity>2: env.log(f"rdm=CTCTTCTC {rdm.size()}\n")

    rdm= rdm.permute(2,0,1,3).contiguous()
    rdm= rdm.view([dimsa[1]]*8).permute(0,2,4,6, 1,3,5,7).contiguous()

    return rdm

#
# tests
#
def test_pC2x2_LU(coord, state, env, verbosity=0):
    tensors= c2x2_LU_t(coord, state, env)
    a= tensors[3]
    dimsa = a.size()
    A = torch.einsum('mefgh,mabcd->eafbgchd',(a,a)).contiguous().view(dimsa[1]**2,\
        dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    tensors_DL= (tensors[0], tensors[1], tensors[2], A)
    C2x2= c2x2_LU_c(*tensors_DL)

    # index structure
    # 
    #  C----T1--1
    #  |   /|
    #  |/-|-|---3
    #  T2---a
    #  |  |  \
    #  0  2   4
    p2x2= _get_partial_C2x2_LU(coord, state, env, verbosity=verbosity)

    # contract with second layer
    #
    #  C----T1--------1
    #  |    |------\
    #  |    |      2
    #  T2---a--4 0 1
    #  | \       \ |
    #  |  \---3 2--a--4->3
    #  0           3->2
    p2x2= torch.tensordot(p2x2, a, ([2,3,4],[1,2,0]))
    p2x2= p2x2.permute(0,2,1,3).contiguous().view(p2x2.size()[0]*dimsa[3],\
        p2x2.size()[1]*dimsa[4])

    dist= torch.dist(p2x2,C2x2)
    if verbosity>0: print(f"|p2x2_LU-C2x2_LU| {dist}")
    return dist/p2x2.numel()<1.0e-14

def test_pC2x2_RU(coord, state, env, verbosity=0):
    tensors= c2x2_RU_t(coord, state, env)
    a= tensors[3]
    dimsa = a.size()
    A = torch.einsum('mefgh,mabcd->eafbgchd',(a,a)).contiguous().view(dimsa[1]**2,\
        dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    tensors_DL= (tensors[0], tensors[1], tensors[2], A)
    C2x2= c2x2_RU_c(*tensors_DL)

    # index structure
    #
    #  0--T2---C
    #     |\   |
    #     | |  |
    #  2--|-|-\|
    #     a----T1
    #    /  |  |
    #   4   3  1
    p2x2= _get_partial_C2x2_RU(coord, state, env, verbosity=verbosity)

    # contract with second layer
    #
    # 0------T2-----C
    #      / |      | 
    #     /  |      |
    #    |   |      |
    #    3   a------T1
    #    1 0 4    / |
    #    |/      /  |
    # 2--a--4 2--   |
    #    3          1
    p2x2= torch.tensordot(p2x2, a, ([2,3,4],[4,1,0]))
    p2x2= p2x2.permute(0,2,1,3).contiguous().view(p2x2.size()[0]*dimsa[2],\
        p2x2.size()[1]*dimsa[3])

    dist= torch.dist(p2x2,C2x2)
    if verbosity>0: print(f"|p2x2_RU-C2x2_RU| {dist}")
    return dist/p2x2.numel()<1.0e-14

def test_pC2x2_RD(coord, state, env, verbosity=0):
    tensors= c2x2_RD_t(coord, state, env)
    a= tensors[3]
    dimsa = a.size()
    A = torch.einsum('mefgh,mabcd->eafbgchd',(a,a)).contiguous().view(dimsa[1]**2,\
        dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    tensors_DL= (tensors[0], tensors[1], tensors[2], A)
    C2x2= c2x2_RD_c(*tensors_DL)

    # index structure
    #
    #       2   0
    #   4\  |   |
    #     a-----T2
    #  3--|-|--/|
    #     | |   |
    #     |/    |
    #  1--T1----C
    p2x2= _get_partial_C2x2_RD(coord, state, env, verbosity=verbosity)

    # contract with second layer
    # 
    #       1->2        0
    # 3<-2--a--4 3---\  |
    #       |\        \ |
    #       3 0 4\     \|
    #       2     a-----T2
    #        \    |     | 
    #         \   |     |
    #          \--|     |
    #    1--------T1----C
    p2x2= torch.tensordot(p2x2, a, ([2,3,4],[3,4,0]))
    p2x2= p2x2.permute(0,2,1,3).contiguous().view(p2x2.size()[0]*dimsa[1],\
        p2x2.size()[1]*dimsa[2])

    dist= torch.dist(p2x2,C2x2)
    if verbosity>0: print(f"|p2x2_RD-C2x2_RD| {dist}")
    return dist/p2x2.numel()<1.0e-14

def test_pC2x2_LD(coord, state, env, verbosity=0):
    tensors= c2x2_LD_t(coord, state, env)
    a= tensors[3]
    dimsa = a.size()
    A = torch.einsum('mefgh,mabcd->eafbgchd',(a,a)).contiguous().view(dimsa[1]**2,\
        dimsa[2]**2, dimsa[3]**2, dimsa[4]**2)

    tensors_DL= (tensors[0], tensors[1], tensors[2], A)
    C2x2= c2x2_LD_c(*tensors_DL)

    # index structure
    #
    # 0   2   4
    # |   |  /
    # T1----a
    # |\--|-|--3
    # |    \|
    # C-----T2--1
    p2x2= _get_partial_C2x2_LD(coord, state, env, verbosity=verbosity)

    # contract with second layer
    # 
    # 0         1-->2  
    # |  /-3 2--a--4-->3
    # | /      0|
    # |/     4  3
    # T1----a   2
    # |     |  /
    # |     |-/
    # C-----T2-----1
    p2x2= torch.tensordot(p2x2, a, ([2,3,4],[3,2,0]))
    p2x2= p2x2.permute(0,2,1,3).contiguous().view(p2x2.size()[0]*dimsa[1],\
        p2x2.size()[1]*dimsa[4])

    dist= torch.dist(p2x2,C2x2)
    if verbosity>0: print(f"|p2x2_LD-C2x2_LD| {dist}")
    return dist/p2x2.numel()<1.0e-14