import torch
from ipeps import IPEPS
from ctm.one_site_c4v.env_c4v import ENV_C4V

def rdm1x1(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor
    
    Computes 1-site reduced density matrix :math:`\rho_{1x1}` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--A^+A--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`A` 
    and it's hermitian conjugate :math:`A^\dagger` are left uncontracted
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    rdm = torch.tensordot(C,T,([0],[0]))
    if verbosity>2: env.log(f"rdm=CT {rdm.size()}\n")
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    rdm = torch.tensordot(rdm,C,([1],[0]))
    if verbosity>2: env.log(f"rdm=CTC {rdm.size()}\n")
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(rdm,T,([2],[0]))
    if verbosity>2: env.log(f"rdm=CTCT {rdm.size()}\n")
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
    A= next(iter(state.sites.values()))
    dimsA = A.size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',A,A).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])
    # C--0
    # |
    # |       0->2
    # T--1 1--a--3
    # |       2\45(s,s')
    # |       3
    # C-------T--2->1
    rdm = torch.tensordot(rdm,a,([1,3],[1,2]))
    if verbosity>2: env.log(f"rdm=CTCTa {rdm.size()}\n")
    # C--0 0--T--1->0
    # |       2
    # |       2
    # T-------a--3->2
    # |       |\45->34(s,s')
    # |       |
    # C-------T--1
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))
    if verbosity>2: env.log(f"rdm=CTCTaT {rdm.size()}\n")
    # C--T--0 0--C
    # |  |       1->0
    # |  |
    # T--a--2
    # |  |\34(s,s')
    # |  |
    # C--T--1
    rdm = torch.tensordot(C,rdm,([0],[0]))
    if verbosity>2: env.log(f"rdm=CTCTaTC {rdm.size()}\n")
    # C--T---------------C
    # |  |               0
    # |  |               0
    # T--a--2 2----------T
    # |  |\34->23(s,s')  1->0
    # |  |
    # C--T--1
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))
    if verbosity>2: env.log(f"rdm=CTCTaTCT {rdm.size()}\n")
    # C--T--------------C
    # |  |              |
    # |  |              |
    # T--a--------------T
    # |  |\23->12(s,s') 0
    # |  |              0
    # C--T--1 1---------C
    rdm = torch.tensordot(rdm,C,([0,1],[0,1]))
    if verbosity>2: env.log(f"rdm=CTCTaTCTC {rdm.size()}\n")

    # symmetrize
    rdm= 0.5*(rdm+rdm.t())
    # TODO make pos-def ?
    eps=0.0
    with torch.no_grad():
        D, U= torch.eig(rdm)
        # check only real part
        if D[:,0].min() < 0:
            eps= D[:,0].min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize
    if verbosity>0: env.log(f"Tr(rdm1x1): {torch.trace(rdm)}\n")
    rdm = rdm / torch.trace(rdm)

    return rdm

def rdm2x1(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of a horizontal 
    2x1 subsystem using following strategy:
    
        1. compute upper left 2x2 corner and lower left 2x1 corner
        2. construct left half of the network
        3. contract left and right half (identical to the left) to obtain final 
           reduced density matrix

    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |  
        T--A^+A--A^+A--T   C2x1--C2x1
        |  |     |     |
        C--T-----T-----C 

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1

    The resulting reduced density matrix is identical to the one of vertical 1x2 subsystem
    due to the C4v symmetry. 
    """
    #----- building C2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    A = next(iter(state.sites.values()))
    dimsA = A.size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',A,A).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # C--1 0--T--1
    # 0       2
    C2x1 = torch.tensordot(C, T, ([1],[0]))

    # C------T--1->0
    # 0      2->1
    # 0
    # T2--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))

    # C-------T--0
    # |       1
    # |       0
    # T2--3 1 a--3
    # 2->1    2\45
    C2x2 = torch.tensordot(C2x2, a, ([1,3],[0,1]))

    # permute 012345->120345
    # reshape 12(03)45->01234
    # C2x2--2
    # | |\34
    # 0 1
    C2x2 = C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1],a.size()[3],T.size()[1]*a.size()[2],dimsA[0],dimsA[0])
    if verbosity>2: env.log(f"C2X2 {C2x2.size()}\n")

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2--2->1
    # | |\34->23
    # 0 1
    # 0 2
    # C2x1--1->0
    left_half = torch.tensordot(C2x1, C2x2, ([0,2],[0,1]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2--1 1----C2x2
    # |\23->01     |\23
    # |            |
    # C2x1--0 0----C2x1
    rdm = torch.tensordot(left_half,left_half,([0,1],[0,1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    rdm= 0.5*(rdm+rdm.t())
    # TODO make pos-def ?
    eps=0.0
    with torch.no_grad():
        D, U= torch.eig(rdm)
        # check only real part
        if D[:,0].min() < 0:
            eps= D[:,0].min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize and reshape
    if verbosity>0: env.log(f"Tr(rdm2x1): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM)

    return rdm

def rdm2x2(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix
           
    TODO try single torch.einsum ?

    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |
        T--A^+A--A^+A--T   C2x2--C2x2
        |  |     |     |
        T--A^+A--A^+A--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1
        s2 s3

    """
    #----- building C2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    A = next(iter(state.sites.values()))
    dimsA = A.size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn',A,A).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--a--3 
    # 2->1    2\45
    C2x2 = torch.tensordot(C2x2, a, ([1,3],[0,1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2 = C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1]*a.size()[2],T.size()[1]*a.size()[3],dimsA[0],dimsA[0])
    if verbosity>2: env.log(f"C2X2 {C2x2.size()}\n")

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2--1 0--C2x2                 C2x2------C2x2
    # |\23->12   |\23->45   & permute |\12->23  |\45
    # 0          1->3                 0         3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    upper_half = torch.tensordot(C2x2, C2x2, ([1],[0]))
    upper_half = upper_half.permute(0,3,1,2,4,5)

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2------C2x2
    # |\23->01  |\45->23
    # 0         1
    # 0         1
    # |/23->45  |/45->67
    # C2x2------C2x2_RD
    rdm = torch.tensordot(upper_half,upper_half,([0,1],[0,1]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # and normalize
    rdm = rdm.permute(0,2,4,6,1,3,5,7).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**4,dimsRDM[0]**4)
    rdm= 0.5*(rdm+rdm.t())
    # TODO make pos-def ?
    eps=0.0
    with torch.no_grad():
        D, U= torch.eig(rdm)
        # check only real part
        if D[:,0].min() < 0:
            eps= D[:,0].min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize and reshape
    if verbosity>0: env.log(f"Tr(rdm2x2): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM)

    return rdm