import torch
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import ENV_C4V

def rdm1x1(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
    CTC = torch.tensordot(C,T,([0],[0]))
    if verbosity>2: env.log(f"rdm=CT {rdm.size()}\n")
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))
    if verbosity>2: env.log(f"rdm=CTC {rdm.size()}\n")
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(CTC,T,([2],[0]))
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
    
    # C--T---0 0--------C
    # |  |              |
    # |  |              |
    # T--a---2 1--------T
    # |  |\34->01(s,s') |
    # |  |              |
    # C--T---1 2--------C
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[0,2,1]))
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

def rdm1x1_sl(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor
    
    Evaluates 1-site operator :math:`Tr(\rho_{1x1}O)` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--a^+a--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are left uncontracted
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    CTC = torch.tensordot(C,T,([0],[0]))
    if verbosity>2: env.log(f"rdm=CT {rdm.size()}\n")
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    CTC = torch.tensordot(CTC,C,([1],[0]))
    if verbosity>2: env.log(f"rdm=CTC {rdm.size()}\n")
    # C--0
    # |
    # T--1
    # |       2->3
    # C--2 0--T--1->2
    rdm = torch.tensordot(CTC,T,([2],[0]))
    
    # 4i) untangle the fused D^2 indices
    #
    # C--0
    # |
    # T--1->1,2
    # |    3->4,5
    # C----T--2->3
    a= next(iter(state.sites.values()))
    rdm= rdm.view(rdm.size()[0],a.size()[2],a.size()[2],rdm.size()[2],a.size()[3],\
        a.size()[3])

    if verbosity>2: env.log(f"rdm=CTCT {rdm.size()}\n")
    #    /
    # --a--
    #  /|s'
    #
    #  s|/
    # --a--
    #  /
    #

    # 4ii) first layer "bra" (in principle conjugate)
    # C--0    ->5
    # |       1/0->4
    # T--1 2--a--4->6
    # |\2->1  3
    # |       4 5->3
    # |       |/
    # C-------T--3->2
    rdm= torch.tensordot(rdm,a,([1,4],[2,3]))
    if verbosity>2: env.log(f"rdm=CTCTa {rdm.size()}\n")
    
    # 4iii) second layer "ket"
    # C--0
    # |   5->3     1->6
    # |  -|---1 2--a--4->7
    # | / |        3\0->5
    # |/  |/4->2   3
    # T---a-----------6->4
    # |   | ------/
    # |   |/
    # C---T-----------2->1
    rdm= torch.tensordot(rdm,a,([1,3],[2,3]))

    # 4iv) fuse pairs of aux indices    
    # C--0 (3 6)->2
    # |     | |/5
    # | ----|-a--7\
    # |/    | |    >3
    # T-----a----4/
    # |4<-2/| |
    # |     |/
    # C-----T----1
    rdm= rdm.permute(0,1,3,6,4,7,2,5).contiguous().view(rdm.size()[0],rdm.size()[1],\
        a.size()[1]**2,a.size()[4]**2,a.size()[0],a.size()[0])

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

    # C--T--0 0---------C
    # |  |              |
    # |  |              |
    # T--a--2 1---------T
    # |  |\34->01(s,s') |
    # |  |              |
    # C--T--1 2---------C
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[0,2,1]))
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
    :type state: IPEPS_C4V
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

def rdm2x1_sl(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
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
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))
    #----- building C2x2_LU ----------------------------------------------------
    # C--1 0--T--1
    # 0       2
    C2x1 = torch.tensordot(C, T, ([1],[0]))

    # see _get_open_C2x2_LU_sl
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))
    C2x2= torch.tensordot(C2x2, a,([1,3],[1,2]))

    # 4iv) fuse (some) pairs of aux indices
    #
    # C------T----0\
    # | 3<-2\|\     \
    # T------a----4\ \ 
    # | \    | |    ->->2
    # |  ------a--7/
    # |      | |\5->4
    # 1->0  (3 6)->1
    # 
    # permute and reshape 01234567->1(36)(047)25->01234
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size()[1],(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2),a.size()[0],a.size()[0])

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

    # normalize and reshape and move to original device
    if verbosity>0: env.log(f"Tr(rdm2x1): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x2_NN_lowmem(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of 2 sites 
    that are nearest neighbours using strategy:

        1. compute upper left corner using double-layer on-site tensor
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2--C2x2c
        |  |     |     |   |     |
        T--A^+A--A^+A--T   C2x2--C2x2c
        |  |     |     |
        T--A^+A--A^+A--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        s1 c

    """
    return _rdm2x2_NN_lowmem(state,env,_get_open_C2x2_LU_dl,force_cpu=force_cpu,verbosity=verbosity)

def rdm2x2_NN_lowmem_sl(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of 2 sites 
    that are nearest neighbours using strategy:

        1. compute upper left corner using layer-by-layer contraction of on-site tensor 
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2--C2x2c
        |  |     |     |   |     |
        T--a^+a--a^+a--T   C2x2--C2x2c
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        s1 c

    """
    return _rdm2x2_NN_lowmem(state,env,_get_open_C2x2_LU_sl,force_cpu=force_cpu,verbosity=verbosity)

def _rdm2x2_NN_lowmem(state,env,f_c2x2,force_cpu=False,verbosity=0):
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))

    #----- building C2x2_LU ----------------------------------------------------
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    def lin_ten_size(t):
        dims=t.size()
        c=1
        for d in dims:
            c=c*d
        return c

    # C2x2--1
    # |\23
    # 0
    C2x2= f_c2x2(C, T, a, verbosity=verbosity)

    # C2x2c--1
    # |
    # 0
    C2x2c= torch.einsum('abii->ab',C2x2)
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")

    #----- build upper part C2x2 -- C2x2c -----------------------------------
    # C2x2c--1 0--C2x2    C2x2c--C2x2
    # |           | \     |      | \
    # 0           1  23   0      1 (23)->2
    C2x2= C2x2.view(T.size()[1]*(a.size()[2]**2),T.size()[1]*(a.size()[3]**2),a.size()[0]**2)
    C2x2= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    # C2x2c= torch.matmul(C2x2c,C2x2)
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")

    # C2x2c----C2x2--2
    # |        |
    # 0        1
    # 0        1 
    # |        |
    # C2x2c----C2x2--2
    rdm= torch.einsum('abi,abj->ij',C2x2,C2x2)
    #C2x2= C2x2c.permute(2,1,0).contiguous().view(dimsA[0]*dimsA[0],(T.size()[1]*a.size()[3])**2)
    #C2x2c= C2x2c.view((T.size()[1]*a.size()[3])**2,dimsA[0]*dimsA[0])
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(rdm)+lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")

    rdm= rdm.view(tuple([a.size()[0] for i in range(4)]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    rdm= 0.5*(rdm+rdm.t())
    eps=0.0
    with torch.no_grad():
        D, U= torch.symeig(rdm)
        if D.min() < 0:
            eps= D.min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize and reshape and move to original device
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x2_NNN_lowmem(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}_{diag}` of 2 sites 
    that are next-nearest neighbours (along diagonal) using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2---C2x2c
        |  |     |     |   |      |
        T--A^+A--A^+A--T   C2x2c--C2x2
        |  |     |     |
        T--A^+A--A^+A--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        c  s1

    """
    return _rdm2x2_NNN_lowmem(state,env,_get_open_C2x2_LU_dl,force_cpu=force_cpu,verbosity=verbosity)

def rdm2x2_NNN_lowmem_sl(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}_{diag}` of 2 sites 
    that are next-nearest neighbours (along diagonal) using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2---C2x2c
        |  |     |     |   |      |
        T--a^+a--A^+a--T   C2x2c--C2x2
        |  |     |     |
        T--a^+a--A^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        c  s1

    """
    return _rdm2x2_NNN_lowmem(state,env,_get_open_C2x2_LU_sl,force_cpu=force_cpu,verbosity=verbosity)

def _rdm2x2_NNN_lowmem(state,env,f_c2x2,force_cpu=False,verbosity=0):
    if force_cpu:
        # move to cpu
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = next(iter(state.sites.values())).cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = next(iter(state.sites.values()))

    #----- building C2x2_LU ----------------------------------------------------
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    def lin_ten_size(t):
        dims=t.size()
        c=1
        for d in dims:
            c=c*d
        return c

    # C2x2--1
    # |\23
    # 0
    C2x2= f_c2x2(C, T, a, verbosity=verbosity)

    # C2x2c--1
    # |
    # 0
    C2x2c= torch.einsum('abii->ab',C2x2)
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")

    #----- build upper part C2x2 -- C2x2c -----------------------------------
    # C2x2c--1 0--C2x2    C2x2c--C2x2
    # |           | \     |      | \
    # 0           1  23   0      1 (23)->2
    C2x2= C2x2.view(T.size()[1]*(a.size()[2]**2),T.size()[1]*(a.size()[3]**2),a.size()[0]**2)
    # rdm= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    C2x2= torch.einsum('ab,bci->aci',C2x2c,C2x2)
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")

    #  ---C2x2----
    # |          /|
    # 0   2=s0s0' 1 
    # 1 2=s1s1'   0
    # |/          | 
    #  ---C2x2----
    rdm= torch.einsum('abi,baj->ij',C2x2,C2x2)
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}"\
            + f" C2x2,C2x2c: {8*(lin_ten_size(rdm)+lin_ten_size(C2x2)+lin_ten_size(C2x2c))}")


    # rdm= torch.einsum('abi,abj->ij',C2x2,rdm)
    rdm= rdm.view(tuple([a.size()[0] for i in range(4)]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = rdm.permute(0,2,1,3).contiguous()

    # symmetrize
    dimsRDM= rdm.size()
    rdm= rdm.view(dimsRDM[0]**2,dimsRDM[0]**2)
    #print( f"rdm_antisym: {torch.norm(rdm-rdm.t())}")
    rdm= 0.5*(rdm+rdm.t())
    eps=0.0
    with torch.no_grad():
        D, U= torch.symeig(rdm)
        if D.min() < 0:
            eps= D.min().abs()
    rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize and reshape and move to original device
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM).to(env.device)

    return rdm

def rdm2x2(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
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
    loc_device=C.device
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--a--3 
    # 2->1    2\45
    C2x2 = torch.tensordot(C2x2, a, ([1,3],[0,1]))
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2 = C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1]*a.size()[2],T.size()[1]*a.size()[3],dimsA[0],dimsA[0])
    if verbosity>2: env.log(f"C2X2 {C2x2.size()}\n")
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2--1 0--C2x2                 C2x2------C2x2
    # |\23->12   |\23->45   & permute |\12->23  |\45
    # 0          1->3                 0         3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    upper_half = torch.tensordot(C2x2, C2x2, ([1],[0]))
    upper_half = upper_half.permute(0,3,1,2,4,5)
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2------C2x2
    # |\23->01  |\45->23
    # 0         1
    # 0         1
    # |/23->45  |/45->67
    # C2x2------C2x2_RD
    rdm = torch.tensordot(upper_half,upper_half,([0,1],[0,1]))
    if verbosity>0:
        print(f"GPU-MEM RDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

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
    # eps=0.0
    # with torch.no_grad():
    #     D, U= torch.eig(rdm)
    #     # check only real part
    #     if D[:,0].min() < 0:
    #         eps= D[:,0].min().abs()
    # rdm+= eps*torch.eye(rdm.size()[0],dtype=rdm.dtype,device=rdm.device)

    # normalize and reshape
    if verbosity>0: env.log(f"Tr(rdm2x2): {torch.trace(rdm)}\n")
    rdm= rdm / torch.trace(rdm)
    rdm= rdm.view(dimsRDM)

    return rdm

def _get_open_C2x2_LU_sl(C, T, a, verbosity=0):
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # 4i) untangle the fused D^2 indices
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])

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
    # |    |/4->2  |
    # T----a----------6->4 
    # | |  |       1/0->5
    # |  -----3 2--a--4->7
    # |    |       3->6
    # |    |
    # 2->1 5->3
    C2x2= torch.tensordot(C2x2, a,([1,3],[1,2]))

    # 4iv) fuse pairs of aux indices
    #
    #  C------T----0\
    #  |    2\|\     \
    #  T------a----4\ \ 
    #  | \    | |    ->->1
    #  |  ------a--7/
    #  |      | |\5->3
    # (1     (3 6))->0
    # 
    # permute and reshape 01234567->(136)(047)25->0123
    C2x2= C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(C2x2.size()[1]*(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2),a.size()[0],a.size()[0])

    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    return C2x2

def test_symm_open_C2x2_LU(C, T, a):
    C2x2= _get_open_C2x2_LU_sl(C,T,a)
    tC2x2= C2x2.permute(1,0,2,3)
    print(f"open C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    C2x2= torch.einsum('ijaa->ij',C2x2)
    chi= C.size()[0]
    C2x2= C2x2.reshape(chi,a.size()[3]**2,chi,a.size()[4]**2)
    # |C2x2|--2    |C2x2|--1
    # |____|--3 => |____|--3
    # 0  1         0  2
    C2x2= C2x2.permute(0,2,1,3).contiguous()

    tC2x2= C2x2.permute(1,0,3,2)
    print(f"C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(1,0,2,3)
    print(f"C2x2-C2x2^t(env) {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(0,1,3,2)
    print(f"C2x2-C2x2^t(aux) {torch.norm(C2x2-tC2x2)}")

def _get_open_C2x2_LU_dl(C, T, a, verbosity=0):
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    dimsa = a.size()
    A = torch.einsum('mefgh,nabcd->eafbgchdmn',a,a).contiguous()\
        .view(dimsa[1]**2, dimsa[2]**2, dimsa[3]**2, dimsa[4]**2, dimsa[0], dimsa[0])
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3
    # 2->1    2\45
    C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2= C2x2.permute(1,2,0,3,4,5).contiguous().view(\
        T.size()[1]*A.size()[2],T.size()[1]*A.size()[3],dimsa[0],dimsa[0])
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")
    
    return C2x2

def partial_rdm2x2(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
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
             0     3
             |     |
          C--T-----T-----C    = C2x2--C2x2
          |  |/2   |/5   |      |     |
       1--T--a^+---a^+---T--4   C2x2--C2x2
          |  |/8   |/11  |
       6--T--a^+---a^+---T--10
          |  |     |     |
          C--T-----T-----C
             |     |
             7     9

    The physical indices `s` of on-sites tensors :math:`a^\dagger` 
    are left uncontracted and given in the following order::
        
        s0 s1
        s2 s3

    """
    #----- building pC2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a = next(iter(state.sites.values()))
    dimsA = a.size()
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    pC2x2= _get_partial_C2x2_LU(C, T, a, verbosity=verbosity)

    #----- build upper part pC2x2_LU--pC2x2_RU -----------------------------------
    # pC2x2----1 1----pC2x2
    # |___|\         /|___|
    # | | \ 3->2 6<-3 / | |
    # 0 2  4->3   7<-4  2 0
    #   V               V V
    #   1               5 4
    pC2x2 = torch.tensordot(pC2x2, pC2x2, ([1],[1]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # construct reduced density matrix by contracting lower and upper halfs
    #  __________________         __________________
    # |_______pC2x2______|       |__________________|
    # | | | \      / | | |       | | | \      / | | |
    # 0 1 2  3    7  6 5 4       | 0 1  2    5  4 3 |
    # 0                  4  =>   |                  |
    # | 1 2  3    7  6 5 |       | 6 7  8   11 10 9 |
    # |_|_|_/______\_|_|_|       |_|_|_/______\_|_|_|
    # |_______pC2x2______|       |_______pC2x2______|
    pC2x2 = torch.tensordot(pC2x2,pC2x2,([0,4],[0,4]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute such, that the index of aux indices for every on-site tensor
    # increases from "up" in the anti-clockwise
    # 01234567891011->012345768910111
    # and normalize
    pC2x2 = pC2x2.permute(0,1,2,3,4,5,7,6,8,9,10,11).contiguous()
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    return pC2x2

def fidelity_rdm2x2(prdm0, state1, force_cpu=False, verbosity=0):
    r"""
    :param prdm0: partial reduced density matrix (without ket part) of 2x2 subsystem
    :param state1: 1-site C4v symmetric wavefunction
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type prdm0: torch.tensor
    :type state1: IPEPS_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: fidelity
    :rtype: torch.tensor

    Contracts 4-site partial reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem 
    with the 2x2 "ket" part given by state1 on-site tensors

    :: 
        prmd0

             0     3
             |     |
          C--T-----T-----C
          |  |/2   |/5   |
       1--T--a^+---a^+---T--4
          |  |/8   |/11  |
       6--T--a^+---a^+---T--10
          |  |     |     |
          C--T-----T-----C
             |     |
             7     9

    """
    a= next(iter(state1.sites.values()))
    dimsA= a.size()
    loc_device=prdm0.device
    is_cpu= loc_device==torch.device('cpu')

    # contract prdm0 with 2x2 section of state1 ipeps
    # 
    #      0       0          0    4
    #    1/      1/         1/   5/
    # 2--a--4 2--a--4 => 2--a----a--7
    #    3       3          3    6
    aa= torch.tensordot(a,a,([4],[2]))

    # contract aa with prmd0
    #
    #                     C--T-------T-------C
    #  _3_____6__         T--a^+a----a^+a----T
    # |____aa____|        |  |   \   |   \   |
    # 1 2 0 5 7 4         |  |/2  6  |/5  7  |
    # 0_1_2_3_4_5_     0--T--a^+-----a^+-----T--4
    # |__prdm0____| =>    C--T-------T-------C
    # 6 7 8 9 10 11          1       3
    fid= torch.tensordot(prdm0,aa,([0,1,2,3,4,5],[1,2,0,5,7,4]))

    # contract rest of the 2x2 section
    #  _____________
    # |_____aa______|
    # 2 3 0 6 7 4 1 5
    # 0_1_2_3_4_5_6_7
    # |____fid______|
    fid= torch.tensordot(fid,aa,([0,1,2,3,4,5,6,7],[2,3,0,6,7,4,1,5]))
    return fid

def _get_partial_C2x2_LU(C, T, a, verbosity=0):
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # 4i) untangle the fused D^2 indices
    #
    # C------T--0
    # 0      1->1,2
    # 0
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],a.size()[2],\
        a.size()[2])

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

    # 4iv) fuse pairs of aux indices
    #
    #  C------T----0    C----T---1
    #  |      |\---1    |    |\--2
    #  |      |         |    |
    #  T------a----6 => T----a
    #  | \    |\        |\    \
    #  2 3    5 4       0 3    4
    # 
    # permute and reshape 0123456->(25)(06)134 ->01234
    C2x2= C2x2.permute(2,5,0,6,1,3,4).contiguous().view(C2x2.size()[2]*a.size()[3],\
        C2x2.size()[0]*a.size()[4],a.size()[1],a.size()[2],a.size()[0])

    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    return C2x2

def test_symm_partial_C2x2_LU(C, T, a):
    C2x2= _get_partial_C2x2_LU(C,T,a)

    tC2x2= C2x2.permute(1,0,3,2,4)
    print(f"C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(1,0,2,3,4)
    print(f"C2x2-C2x2^t(env) {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(0,1,3,2,4)
    print(f"C2x2-C2x2^t(aux) {torch.norm(C2x2-tC2x2)}")

def aux_rdm2x2(state, env, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param force_cpu: execute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 4-site auxilliary reduced density matrix
    :rtype: torch.tensor

    Computes 4-site aux reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem 
    without the on-site tensors (leaving corresponding aux indices open)
    using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           auxilliary density matrix
           
    TODO try single torch.einsum ?

    :: 

          C----T----T----C    = C2x2--C2x2
          |              |      |     |
          T--          --T      C2x2--C2x2
          |              |
          T--          --T
          |              |
          C----T----T----C

    """
    #----- building pC2x2_LU ----------------------------------------------------
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    a = next(iter(state.sites.values()))
    dimsA = a.size()
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    aC2x2= _get_aux_C2x2_LU(C, T, verbosity=verbosity)

    #----- build upper part pC2x2_LU--pC2x2_RU -----------------------------------
    # aC2x2----1 1----aC2x2
    # |___|\         /|___|
    # | |   3->2 5<-3   | |
    # 0 2               2 0
    #   V               V V
    #   1               4 3 
    pC2x2 = torch.tensordot(pC2x2, pC2x2, ([1],[1]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # construct reduced density matrix by contracting lower and upper halfs
    #  __________________         __________________
    # |_______pC2x2______|       |__________________|
    # | | | \      / | | |       | | | \      / | | |
    # 0 1 2  3    7  6 5 4       | 0 1  2    5  4 3 |
    # 0                  4  =>   |                  |   
    # | 1 2  3    7  6 5 |       | 6 7  8   11 10 9 |
    # |_|_|_/______\_|_|_|       |_|_|_/______\_|_|_|
    # |_______pC2x2______|       |_______pC2x2______|
    pC2x2 = torch.tensordot(pC2x2,pC2x2,([0,4],[0,4]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # permute such, that the index of aux indices for every on-site tensor
    # increases from "up" in the anti-clockwise
    # 01234567891011->012345768910111
    # and normalize
    pC2x2 = pC2x2.permute(0,1,2,3,4,5,7,6,8,9,10,11).contiguous()
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM pRDM2X2 MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    return pC2x2

def _get_aux_C2x2_LU(C, T, verbosity=0):
    loc_device=C.device
    is_cpu= loc_device==torch.device('cpu')

    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))
    if not is_cpu and verbosity>0:
        print(f"GPU-MEM RDM2X1_diag MAX:{torch.cuda.max_memory_allocated(loc_device)}"\
            + f" CURRENT:{torch.cuda.memory_allocated(loc_device)}")

    # |C2x2|--0    |C2x2|--1  
    # |____|--1 => |____|--3
    #  2  3         0  2
    C2x2= C2x2.permute(2,0,3,1).contiguous()

    return C2x2

def test_symm_aux_C2x2_LU(C, T):
    C2x2= _get_aux_C2x2_LU(C,T)

    tC2x2= C2x2.permute(1,0,3,2)
    print(f"C2x2-C2x2^t {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(1,0,2,3)
    print(f"C2x2-C2x2^t(env) {torch.norm(C2x2-tC2x2)}")

    tC2x2= C2x2.permute(0,1,3,2)
    print(f"C2x2-C2x2^t(aux) {torch.norm(C2x2-tC2x2)}")