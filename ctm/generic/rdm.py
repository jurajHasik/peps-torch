import torch
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV

def rdm1x1(coord, state, env, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor

    Computes 1-site reduced density matrix :math:`\rho_{1x1}` centered on vertex ``coord`` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--A^+A--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`A` at vertex ``coord`` 
    and it's hermitian conjugate :math:`A^\dagger` are left uncontracted
    """
    # C(-1,-1)--1->0
    # 0
    # 0
    # T(-1,0)--2
    # 1
    rdm = torch.tensordot(env.C[(coord,(-1,-1))],env.T[(coord,(-1,0))],([0],[0]))
    if verbosity>0:
        print("rdm=CT "+str(rdm.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--2->1
    # 1
    # 0
    # C(-1,1)--1->2
    rdm = torch.tensordot(rdm,env.C[(coord,(-1,1))],([1],[0]))
    if verbosity>0:
        print("rdm=CTC "+str(rdm.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T(0,1)--2->3
    rdm = torch.tensordot(rdm,env.T[(coord,(0,1))],([2],[1]))
    if verbosity>0:
        print("rdm=CTCT "+str(rdm.size()))
    # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
    #        Possibly reshape indices 1,2 of rdm, which are to be contracted with 
    #        on-site tensor and contract bra,ket in two steps instead of creating 
    #        double layer tensor
    #    /
    # --A--
    #  /|s
    #  
    # s'|/
    # --A--
    #  /
    #
    dimsA = state.site(coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',state.site(coord),state.site(coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])
    # C(-1,-1)--0
    # |
    # |             0->2
    # T(-1,0)--1 1--a--3
    # |             2\45(s,s')
    # |             2
    # C(-1,1)-------T(0,1)--3->1
    rdm = torch.tensordot(rdm,a,([1,2],[1,2]))
    if verbosity>0:
        print("rdm=CTCTa "+str(rdm.size()))
    # C(-1,-1)--0 0--T(0,-1)--2->0
    # |              1
    # |              2
    # T(-1,0)--------a--3->2
    # |              |\45->34(s,s')
    # |              |
    # C(-1,1)--------T(0,1)--1
    rdm = torch.tensordot(env.T[(coord,(0,-1))],rdm,([0,1],[0,2]))
    if verbosity>0:
        print("rdm=CTCTaT "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)--0 0--C(1,-1)
    # |         |             1->0
    # |         |
    # T(-1,0)---a--2
    # |         |\34(s,s')
    # |         |
    # C(-1,1)---T(0,1)--0->1
    rdm = torch.tensordot(env.C[(coord,(1,-1))],rdm,([0],[0]))
    if verbosity>0:
        print("rdm=CTCTaTC "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)-----C(1,-1)
    # |         |           0
    # |         |           0
    # T(-1,0)---a--2 1------T(1,0) 
    # |         |\34->23(s,s')  2->0
    # |         |
    # C(-1,1)---T(0,1)--1
    rdm = torch.tensordot(env.T[(coord,(1,0))],rdm,([0,1],[0,2]))
    if verbosity>0:
        print("rdm=CTCTaTCT "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)--------C(1,-1)
    # |         |              |
    # |         |              |
    # T(-1,0)---a--------------T(1,0) 
    # |         |\23->12(s,s') 0
    # |         |              0
    # C(-1,1)---T(0,1)--1 1----C(1,1)
    rdm = torch.tensordot(rdm,env.C[(coord,(1,1))],([0,1],[0,1]))
    if verbosity>0:
        print("rdm=CTCTaTCTC "+str(rdm.size()))

    # normalize
    rdm = rdm / torch.trace(rdm)

    return rdm

def rdm2x1(coord, ipeps, env, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies position of 2x1 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of a horizontal 
    2x1 subsystem using following strategy:
    
        1. compute four individual corners 
        2. construct right and left half of the network
        3. contract right and left halt to obtain final reduced density matrix

    ::

        C--T------------T------------------C = C2x2_LU(coord)--C2x2(coord+(1,0))
        |  |            |                  |   |               |  
        T--A^+A(coord)--A^+A(coord+(1,0))--T   C2x1_LD(coord)--C2x1(coord+(1,0))
        |  |            |                  |
        C--T------------T------------------C 

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    at vertices ``coord``, ``coord+(1,0)`` are left uncontracted
    """
    #----- building C2x2_LU ----------------------------------------------------
    C = env.C[(ipeps.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(0,-1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(-1,0))]
    dimsA = ipeps.site(coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(coord),ipeps.site(coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # C--10--T1--2
    # 0      1
    C2x2_LU = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2_LU = torch.tensordot(C2x2_LU, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 a--3 
    # 2->1    2\45
    C2x2_LU = torch.tensordot(C2x2_LU, a, ([0,3],[0,1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = C2x2_LU.permute(1,2,0,3,4,5).contiguous().view(\
        T1.size()[2]*a.size()[3],T2.size()[1]*a.size()[2],dimsA[0],dimsA[0])
    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

    #----- building C2x1_LD ----------------------------------------------------
    C = env.C[(ipeps.vertexToSite(coord),(-1,1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(0,1))]

    # 0       0->1
    # C--1 1--T2--2
    C2x1_LD = torch.tensordot(C, T2, ([1],[1]))

    # reshape (01)2->(0)1
    # 0
    # |
    # C2x1--1
    C2x1_LD = C2x1_LD.view(C.size()[0]*T2.size()[0],T2.size()[2]).contiguous()
    if verbosity>0:
        print("C2X1 LD "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,1): "+str(C2x1_LD.size()))

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2_LU--1 
    # |\23
    # 0
    # 0
    # C2x1_LD--1->0
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    left_half = torch.tensordot(C2x1_LD, C2x2_LU, ([0],[0]))

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shitf_coord = ipeps.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C = env.C[(shitf_coord,(1,-1))]
    T1 = env.T[(shitf_coord,(1,0))]
    T2 = env.T[(shitf_coord,(0,-1))]
    dimsA = ipeps.site(shitf_coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(shitf_coord),ipeps.site(shitf_coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2_RU = torch.tensordot(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2_RU = torch.tensordot(C2x2_RU, T2, ([0],[2]))

    # 1<-2--T2------C
    #       3       |
    #    45\0       |
    # 2<-1--a--3 0--T1
    #    3<-2    0<-1
    C2x2_RU = torch.tensordot(C2x2_RU, a, ([0,3],[3,0]))

    # permute 012334->120345
    # reshape (12)(03)45->0123
    # 0--C2x2
    # 23/|
    #    1
    C2x2_RU = C2x2_RU.permute(1,2,0,3,4,5).contiguous().view(\
        T2.size()[0]*a.size()[1],T1.size()[2]*a.size()[2], dimsA[0], dimsA[0])
    if verbosity>0:
        print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

    #----- building C2x1_RD ----------------------------------------------------
    C = env.C[(shitf_coord,(1,1))]
    T1 = env.T[(shitf_coord,(0,1))]

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x1_RD = torch.tensordot(C, T1, ([1],[2]))

    # reshape (01)2->(0)1
    C2x1_RD = C2x1_RD.view(C.size()[0]*T1.size()[0],T1.size()[1]).contiguous()

    #    0
    #    |
    # 1--C2x1
    if verbosity>0:
        print("C2X1 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x1_RD.size()))

    

    #----- build right part C2x2_RU--C2x1_RD -----------------------------------
    # 1<-0--C2x2_RU
    #       |\23
    #       1
    #       0
    # 0<-1--C2x1_RD
    right_half = torch.tensordot(C2x1_RD, C2x2_RU, ([0],[1]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2_LU--1 1----C2x2_RU
    # |\23->01        |\23
    # |               |    
    # C2x1_LD--0 0----C2x1_RD
    rdm = torch.tensordot(left_half,right_half,([0,1],[0,1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3)
    rdm = rdm / torch.einsum('ijij',rdm)

    return rdm

def rdm1x2(coord, ipeps, env, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies position of 1x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{1x2}` of a vertical 
    1x2 subsystem using following strategy:
    
        1. compute four individual corners 
        2. construct upper and lower half of the network
        3. contract upper and lower halt to obtain final reduced density matrix

    ::

        C--T------------------C = C2x2_LU(coord)--------C1x2(coord)
        |  |                  |   |                     |
        T--A^+A(coord)--------T   C2x2_LD(coord+(0,1))--C1x2(coord+0,1))
        |  |                  |
        T--A^+A(coord+(0,1))--T
        |  |                  |
        C--T------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    at vertices ``coord``, ``coord+(0,1)`` are left uncontracted
    """
    #----- building C2x2_LU ----------------------------------------------------
    C = env.C[(ipeps.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(0,-1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(-1,0))]
    dimsA = ipeps.site(coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(coord),ipeps.site(coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # C--10--T1--2
    # 0      1
    C2x2_LU = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2_LU = torch.tensordot(C2x2_LU, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 a--3 
    # 2->1    2\45
    C2x2_LU = torch.tensordot(C2x2_LU, a, ([0,3],[0,1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = C2x2_LU.permute(1,2,0,3,4,5).contiguous().view(\
        T1.size()[2]*a.size()[3],T2.size()[1]*a.size()[2],dimsA[0],dimsA[0])
    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

    #----- building C1x2_RU ----------------------------------------------------
    C = env.C[(ipeps.vertexToSite(coord),(1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(1,0))]

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C1x2_RU = torch.tensordot(C, T1, ([1],[0]))

    # reshape (01)2->(0)1
    # 0--C1x2
    # 23/|
    #    1
    C1x2_RU = C1x2_RU.view(C.size()[0]*T1.size()[1],T1.size()[2]).contiguous()
    if verbosity>0:
        print("C1X2 RU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (1,-1): "+str(C1x2_RU.size()))

    #----- build upper part C2x2_LU--C1x2_RU -----------------------------------
    # C2x2_LU--1 0--C1x2_RU
    # |\23          |
    # 0->1          1->0
    upper_half = torch.tensordot(C1x2_RU, C2x2_LU, ([0],[1]))

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shitf_coord = ipeps.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C = env.C[(shitf_coord,(-1,1))]
    T1 = env.T[(shitf_coord,(-1,0))]
    T2 = env.T[(shitf_coord,(0,1))]
    dimsA = ipeps.site(shitf_coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(shitf_coord),ipeps.site(shitf_coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2_LD = torch.tensordot(C, T1, ([0],[1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T2--2->3
    C2x2_LD = torch.tensordot(C2x2_LD, T2, ([0],[1]))

    # 0       0->2
    # T1--1 1--a--3
    # |        2\45
    # |        2
    # C--------T2--3->1
    C2x2_LD = torch.tensordot(C2x2_LD, a, ([1,2],[1,2]))

    # permute 012345->021345
    # reshape (02)(13)45->0123
    # 0
    # |/23
    # C2x2--1
    C2x2_LD = C2x2_LD.permute(0,2,1,3,4,5).contiguous().view(\
        T1.size()[0]*a.size()[0],T2.size()[2]*a.size()[3], dimsA[0], dimsA[0])
    if verbosity>0:
        print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

    #----- building C2x2_RD ----------------------------------------------------
    C = env.C[(shitf_coord,(1,1))]
    T2 = env.T[(shitf_coord,(1,0))]

    #       0
    #    1--T2
    #       2
    #       0
    # 2<-1--C
    C1x2_RD = torch.tensordot(T2, C, ([2],[0]))

    # permute 012->021
    # reshape 0(12)->0(1)
    C1x2_RD = C1x2_RD.permute(0,2,1).contiguous().view(T2.size()[0],C.size()[1]*T2.size()[1])

    #    0
    #    |
    # 1--C1x2
    if verbosity>0:
        print("C1X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C1x2_RD.size()))

    

    #----- build lower part C2x2_LD--C1x2_RD -----------------------------------
    # 0->1          0
    # |/23          |
    # C2x2_LD--1 1--C1x2_RD 
    lower_half = torch.tensordot(C1x2_RD, C2x2_LD, ([1],[1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C1x2_RU
    # |\23->01     |
    # 1            0    
    # 1            0    
    # |/23         |
    # C2x2_LD------C1x2_RD
    rdm = torch.tensordot(upper_half,lower_half,([0,1],[0,1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3)
    rdm = rdm / torch.einsum('ijij',rdm)

    return rdm

def rdm2x2(coord, ipeps, env, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem 
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem specified
    by the vertex ``coord`` of its upper left corner using strategy:

        1. compute four individual corners
        2. construct upper and lower half of the network
        3. contract upper and lower half to obtain final reduced density matrix

    ::

        C--T------------------T------------------C = C2x2_LU(coord)--------C2x2(coord+(1,0))
        |  |                  |                  |   |                     |
        T--A^+A(coord)--------A^+A(coord+(1,0))--T   C2x2_LD(coord+(0,1))--C2x2(coord+(1,1))
        |  |                  |                  |
        T--A^+A(coord+(0,1))--A^+A(coord+(1,1))--T
        |  |                  |                  |
        C--T------------------T------------------C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    at vertices ``coord``, ``coord+(1,0)``, ``coord+(0,1)``, and ``coord+(1,1)`` are 
    left uncontracted and given in the same order::
        
        s0 s1
        s2 s3

    """
    #----- building C2x2_LU ----------------------------------------------------
    C = env.C[(ipeps.vertexToSite(coord),(-1,-1))]
    T1 = env.T[(ipeps.vertexToSite(coord),(0,-1))]
    T2 = env.T[(ipeps.vertexToSite(coord),(-1,0))]
    dimsA = ipeps.site(coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(coord),ipeps.site(coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # C--10--T1--2
    # 0      1
    C2x2_LU = torch.tensordot(C, T1, ([1],[0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2_LU = torch.tensordot(C2x2_LU, T2, ([0],[0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 a--3 
    # 2->1    2\45
    C2x2_LU = torch.tensordot(C2x2_LU, a, ([0,3],[0,1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = C2x2_LU.permute(1,2,0,3,4,5).contiguous().view(\
        T1.size()[2]*a.size()[3],T2.size()[1]*a.size()[2],dimsA[0],dimsA[0])
    if verbosity>0:
        print("C2X2 LU "+str(coord)+"->"+str(ipeps.vertexToSite(coord))+" (-1,-1): "+str(C2x2_LU.size()))

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shitf_coord = ipeps.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C = env.C[(shitf_coord,(1,-1))]
    T1 = env.T[(shitf_coord,(1,0))]
    T2 = env.T[(shitf_coord,(0,-1))]
    dimsA = ipeps.site(shitf_coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(shitf_coord),ipeps.site(shitf_coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2_RU = torch.tensordot(C, T1, ([1],[0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2_RU = torch.tensordot(C2x2_RU, T2, ([0],[2]))

    # 1<-2--T2------C
    #       3       |
    #    45\0       |
    # 2<-1--a--3 0--T1
    #    3<-2    0<-1
    C2x2_RU = torch.tensordot(C2x2_RU, a, ([0,3],[3,0]))

    # permute 012334->120345
    # reshape (12)(03)45->0123
    # 0--C2x2
    # 23/|
    #    1
    C2x2_RU = C2x2_RU.permute(1,2,0,3,4,5).contiguous().view(\
        T2.size()[0]*a.size()[1],T1.size()[2]*a.size()[2], dimsA[0], dimsA[0])
    if verbosity>0:
        print("C2X2 RU "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,-1): "+str(C2x2_RU.size()))

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU              C2x2_LU------C2x2_RU
    # |\23->12      |\23->45   & permute |\12->23      |\45
    # 0             1->3                 0             3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    upper_half = torch.tensordot(C2x2_LU, C2x2_RU, ([1],[0]))
    upper_half = upper_half.permute(0,3,1,2,4,5)

    #----- building C2x2_RD ----------------------------------------------------
    vec = (1,1)
    shitf_coord = ipeps.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C = env.C[(shitf_coord,(1,1))]
    T1 = env.T[(shitf_coord,(0,1))]
    T2 = env.T[(shitf_coord,(1,0))]
    dimsA = ipeps.site(shitf_coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(shitf_coord),ipeps.site(shitf_coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x2_RD = torch.tensordot(C, T1, ([1],[2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2_RD = torch.tensordot(C2x2_RD, T2, ([0],[2]))

    #    2<-0    1<-2
    # 3<-1--a--3 3--T2
    #       2\45    |
    #       0       |
    # 0<-1--T1------C
    C2x2_RD = torch.tensordot(C2x2_RD, a, ([0,3],[2,3]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    C2x2_RD = C2x2_RD.permute(1,2,0,3,4,5).contiguous().view(\
        T2.size()[0]*a.size()[0],T1.size()[1]*a.size()[1], dimsA[0], dimsA[0])

    #    0
    #    |/23
    # 1--C2x2
    if verbosity>0:
        print("C2X2 RD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (1,1): "+str(C2x2_RD.size()))

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shitf_coord = ipeps.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C = env.C[(shitf_coord,(-1,1))]
    T1 = env.T[(shitf_coord,(-1,0))]
    T2 = env.T[(shitf_coord,(0,1))]
    dimsA = ipeps.site(shitf_coord).size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',ipeps.site(shitf_coord),ipeps.site(shitf_coord)).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2_LD = torch.tensordot(C, T1, ([0],[1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T2--2->3
    C2x2_LD = torch.tensordot(C2x2_LD, T2, ([0],[1]))

    # 0        0->2
    # T1--1 1--a--3
    # |        2\45
    # |        2
    # C--------T2--3->1
    C2x2_LD = torch.tensordot(C2x2_LD, a, ([1,2],[1,2]))

    # permute 012345->021345
    # reshape (02)(13)45->0123
    # 0
    # |/23
    # C2x2--1
    C2x2_LD = C2x2_LD.permute(0,2,1,3,4,5).contiguous().view(\
        T1.size()[0]*a.size()[0],T2.size()[2]*a.size()[3], dimsA[0], dimsA[0])
    if verbosity>0:
        print("C2X2 LD "+str((coord[0]+vec[0],coord[1]+vec[1]))+"->"+str(shitf_coord)+" (-1,1): "+str(C2x2_LD.size()))

    #----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->3                 0             3->1
    # |/23->12      |/23->45   & permute |/12->23      |/45
    # C2x2_LD--1 1--C2x2_RD              C2x2_LD------C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
    lower_half = torch.tensordot(C2x2_LD, C2x2_RD, ([1],[1]))
    lower_half = lower_half.permute(0,3,1,2,4,5)

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |\23->01     |\45->23
    # 0            1    
    # 0            1    
    # |/23->45     |/45->67
    # C2x2_LD------C2x2_RD
    rdm = torch.tensordot(upper_half,lower_half,([0,1],[0,1]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # and normalize
    rdm = rdm.permute(0,2,4,6,1,3,5,7)
    rdm = rdm / torch.einsum('ijklijkl',rdm)

    return rdm