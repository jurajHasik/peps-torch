import logging
from tn_interface_abelian import contract, permute, conj
from .ctm_components_c4v import c2x2_dl
log = logging.getLogger(__name__)

def _sym_pos_def_matrix(rdm, sym_pos_def=False, verbosity=0, who="unknown"):
    rdm_asym= 0.5*(rdm-rdm.transpose().conj())
    rdm= 0.5*(rdm+rdm.transpose().conj())
    if verbosity>0: 
        log.info(f"{who} norm(rdm_sym) {rdm.norm()} norm(rdm_asym) {rdm_asym.norm()}")
    # if sym_pos_def:
    #     with torch.no_grad():
    #         D, U= torch.symeig(rdm, eigenvectors=True)
    #         if D.min() < 0:
    #             log.info(f"{who} max(diag(rdm)) {D.max()} min(diag(rdm)) {D.min()}")
    #             D= torch.clamp(D, min=0)
    #             rdm_posdef= U@torch.diag(D)@U.t()
    #             rdm.copy_(rdm_posdef)
    rdm = rdm / rdm.trace().to_number()
    return rdm

def _sym_pos_def_rdm(rdm, sym_pos_def=False, verbosity=0, who=None):
    assert rdm.ndim%2==0, "invalid rank of RDM"
    nsites= rdm.ndim//2
    rdm, lo_bra= rdm.group_legs(tuple(nsites+i for i in range(nsites)), new_s=1)
    rdm, lo_ket= rdm.group_legs(tuple(i for i in range(nsites)), new_s=-1)
    rdm= _sym_pos_def_matrix(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.ungroup_leg(1, lo_bra)
    rdm= rdm.ungroup_leg(0, lo_ket)
    return rdm

# CONVENTION:
#
# when grouping indices, environment index always preceeds aux-indices of 
# double-layer on-site tensor

# ----- components -------------------------------------------------------------
def aux_C2x2_LU(a,C,T, verbosity=0):
    # C--1(-1)(+1)0--T--1(+1)
    # 0(-1)      (-1)2
    c2x2= contract(C, T, ([1],[0]))

    # C------T--1->2(+) => C------T--2(+1)->1(+)
    # 0(-1)  2->3(-1)      |      3->2(-)
    # 0(+1)                |
    # T--2->1(-)           T--1(-)->3(-)
    # 1->0(+1)             0(+)
    c2x2= contract(T, c2x2, ([0],[0]))
    c2x2= permute(c2x2,(0,2,3,1))

    # Open indices connecting Ts to on-site tensor. The unmerged index pairs are ordered 
    # as ket,bra
    #
    # C-------T--1(+1)
    # |       2->2(+1),3(-1)
    # T--3->3(+1),4(-1)->4(+1),5(-1)
    # |
    # 0(+1)
    c2x2= c2x2.ungroup_leg(3, T._leg_fusion_data[2])
    c2x2= c2x2.ungroup_leg(2, T._leg_fusion_data[2])

    return c2x2

def open_C2x2_LU(a,C,T, verbosity=0):
    # C-------T--1(+1)
    # |       2(+1),3(-1)
    # T--4(+1),5(-1)
    # |
    # 0(+1)
    c2x2= aux_C2x2_LU(a,C,T,verbosity=verbosity)

    # C--------------T--1(+1)     => C-----------T_|--1(+1)
    # |           (-)2 \3->2(+)      |   (+)2<-4\|  \
    # |           (+)1               T-----------a-------6->4(+)
    # T----4(-)(+)2--a--4->6(+)      |\          |   2(+)
    # |\5->3(+)      |\0->4(+)       | \         |   1(-) /0->5(-) 
    # |              3->5(+)         |  \(+)3 2(-)---a*--4->7(-)
    # 0(+1)                          |           |   3->6(-)
    #                                0(+)  (+)3<-5
    #
    c2x2= contract(c2x2, a, ([2,4], [1,2]))
    c2x2= contract(c2x2, a.conj(), ([2,3], [1,2]))


    # C-----T--1->3(+)                             C----T--3(+)       /2(+)
    # |     |                                  =>  |    |
    # T----a*a--4(+),7(-)->4(+),5(-)->4->3(+1)     T---a*a--4(+),5(-) /3(+)
    # |     |\2(+),5(-)->6,7->5,6->4(+),5(-)       |    |\6(+),7(-)   /4(+),5(-)
    # 0(+)  3(+),6(-)->1(+),2(-)->2(+1)            0(+) 1(+),2(-)     /1(+)
    c2x2= permute(c2x2, (0,3,6,1,4,7,2,5))
    # c2x2, lo1= c2x2.group_legs((4,5), new_s=1)
    # c2x2, lo3= c2x2.group_legs((1,2), new_s=1)
    # c2x2._leg_fusion_data[1]= lo1
    # c2x2._leg_fusion_data[3]= lo3

    return c2x2

def closed_C2x2_LU(a,C,T, verbosity=0):
    # C-------T--1(+1)
    # |       2(+1),3(-1)
    # T--4(+1),5(-1)
    # |
    # 0(+1)
    c2x2= aux_C2x2_LU(a,C,T,verbosity=verbosity)

    # C--------------T--1(+)     =>  C--------------T__|--1(+)
    # |           (-)2 \3->2(+)      |              |  \
    # |           (+)1               T--------------a-------6->4(+)
    # T----4(-)(+)2--a--4->6(+)      |\        (+)4/|   2(+)
    # |\5->3(+)      |\0->4(+)       | \       (-)0-|--\1(-)
    # |              3->5(+)         |  \(+)3 2(-)------a*--4->7(-)
    # 0(+)                           |              |   3->6(-) 
    #                                0(+)     (+)3<-5
    #
    c2x2= contract(c2x2, a, ([2,4], [1,2]))
    c2x2= contract(c2x2, a.conj(), ([2,3,4], [1,2,0]))

    # C-----T--1->3(+)                        C----T---3(+)      /2(+)
    # |     |                             =>  |    |
    # T----a*a--3(+),5(-)->4,5->4->3(+1)      T---a*a--4(+),5(-) /3(+)
    # |     |                                 |    |
    # 0(+)  2(+),4(-)->1,2->2(+1)             0(+) 1(+),2(-)     /1(+) 
    c2x2= permute(c2x2, (0,2,4,1,3,5))
    # c2x2, lo= c2x2.group_legs((3,5), new_s=1)
    # c2x2, lo= c2x2.group_legs((2,4), new_s=1)

    return c2x2

# ----- density matrices in physical space -------------------------------------
def rdm1x1(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: yamps.Tensor

    Computes 1-site reduced density matrix :math:`\rho_{1x1}` centered on vertex ``coord`` by 
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--a^+a--T
        |  |     |
        C--T-----C

    where the physical indices `s` and `s'` of on-site tensor :math:`A` at vertex ``coord`` 
    and it's hermitian conjugate :math:`A^\dagger` are left uncontracted
    """
    who= "rdm1x1"
    a= state.site()
    C= env.get_C()
    T= env.get_T()

    # C----T--3(+)       /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |\6(+),7(-)   /4(+),5(-)
    # 0(+) 1(+),2(-)     /1(+)
    rdm= open_C2x2_LU(a,C,T, verbosity=verbosity)

    # 0(-)       (-)2           0(-)   2(-)->2(-),3(+)
    # C--(-)1 (+)0--T--(+)1  => C------T--1(+)
    #
    C2x1_LU= contract(C, T,([1],[0]))
    C2x1_LU= C2x1_LU.ungroup_leg(2, T._leg_fusion_data[2])

    # |rdm     |--3(+)->1(+)
    # |________|--4(+),5(-)->2(+),3(-)
    # |      | \--6(+),7(-)->4(+),5(-)
    # 0(+)   1(+),2(-)
    # 0(-)   2(-),3(+)
    # C------T--1->0(+)
    rdm= contract(C2x1_LU, rdm, ([0,2,3],[0,1,2]))
    if verbosity>0:
        print("rdm=CTCTaT "+str(rdm))

    #       (-)0--|C2x1|         (-)0--|C2x1|
    #  2(-),3(+)--|____| => 1(-),2(+)--|____|
    #             (+)1                      |
    #             (-)0                      |
    #          (-)1--C                (-)3--C
    C2x1_LU= contract(C2x1_LU, C, ([1],[0]))
    if verbosity>0:
        print("rdm=CTC "+str(E))

    # C--T---------------------1(+) (-)0-------C
    # |  | /4(+),5(-)                          |
    # |  |/  ->0(+),1(-)(s,s')                 |
    # T--a----------------2(+),3(-) 1(-),2(+)--T
    # |  |                                     |
    # |  |                                     |
    # C--T---------------------0(+) (-)3-------C
    # rdm = contract(rdm,E,([0,1,2],[2,0,1]))
    rdm = contract(C2x1_LU,rdm,([3,0,1,2],[0,1,2,3]))
    if verbosity>0:
        print("rdm=CTCTaTCTC "+str(rdm))

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm2x1(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: yamps.Tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of a horizontal 
    2x1 subsystem using following strategy:
    
        1. compute upper left 2x2 corner and lower left 2x1 corner
        2. construct left half of the network
        3. contract left and right half (identical to the left) to obtain final 
           reduced density matrix

    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |  
        T--a^+a--a^+a--T   C2x1--C2x1
        |  |     |     |
        C--T-----T-----C 

    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1

    The resulting reduced density matrix is identical to the one of vertical 1x2 subsystem
    due to the C4v symmetry. 
    """
    #----- building C2x2_LU ----------------------------------------------------
    who= "rdm2x1"
    a= state.site()
    C= env.get_C()
    T= env.get_T()

    # C----T--3(+)       /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |\6(+),7(-)   /4(+),5(-)
    # 0(+) 1(+),2(-)     /1(+)
    left_half= open_C2x2_LU(a,C,T, verbosity=verbosity)

    # 0(-)       (-)2           0(-)   2(-)->2(-),3(+)
    # C--(-)1 (+)0--T--(+)1  => C------T--1(+)
    #
    C2x1_LU= contract(C, T,([1],[0]))
    C2x1_LU= C2x1_LU.ungroup_leg(2, T._leg_fusion_data[2])

    # |rdm     |--3(+)->1(+)
    # |________|--4(+),5(-)->2(+),3(-)
    # |      | \--6(+),7(-)->4(+),5(-)
    # 0(+)   1(+),2(-)
    # 0(-)   2(-),3(+)
    # C------T--1->0(+)
    left_half= contract(C2x1_LU, left_half, ([0,2,3],[0,1,2]))

    # construct reduced density matrix by contracting left and right halfs
    #
    # C2x2---------1(+) 1(-)---------C2x2
    # |__|----2(+),3(-) 2(-),3(+)----|  |
    # |\4(+),5(-)             4(-),5(+)/|
    # | ->0(+),1(-)         ->2(-),3(+) |
    # C2x1---------0(+) 0(+)---------C2x1
    rdm = contract(left_half,left_half.negate_signature(),([0,1,2,3],[0,1,2,3]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    rdm= permute(rdm,(0,2,1,3))

    # symmetrize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    return rdm

def rdm2x2_NN(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: yamps.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}` of 2 sites 
    that are nearest neighbours using strategy:

        1. compute upper left corner using double-layer on-site tensor
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2c--C2x2--s0,s'0
        |  |     |     |   |      |
        T--a^+a--a^+a--T   C2x2c--C2x2--s1,s'1
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`a` (and :math:`a^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        c s0,s'0
        c s1,s'1

    """
    who= "rdm2x2_NN"
    if force_cpu:
        C = env.C[env.keyC].to('cpu')
        T = env.T[env.keyT].to('cpu')
        a = state.site().to('cpu')
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = state.site()

    # C----T--3(+)       /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |\6(+),7(-)   /4(+),5(-)
    # 0(+) 1(+),2(-)     /1(+)
    C2x2= open_C2x2_LU(a,C,T, verbosity=verbosity)

    # C----T---3(+)      /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |
    # 0(+) 1(+),2(-)     /1(+)
    C2x2c= closed_C2x2_LU(a,C,T, verbosity=verbosity)

    # build upper part C2x2 -- C2x2c
    #
    # C--------T-------------3(+->-) 3(+)--------T-----C
    # |        |                                 |/----|--6(+),7(-)
    # T-------a*a--4(+1->-),5(-1->+) 4(+),5(-)--a*a----T
    # |        |                                 |     |
    # 0(+1->-) 1(+1->-),2(-1->+)            1(+),2(-)  0(+)
    #                                     ->4(+),5(-)  ->3(+)
    C2x2= contract(C2x2c.negate_signature(), C2x2, ([3,4,5],[3,4,5]))

    # C2x2______________________|--6(+),7(-)->0(+),1(-)
    # |                |
    # 0(-),1(-),2(+)   3(+),4(+),5(-)
    # 0(+),1(+),2(-)   3(-),4(-),5(+)
    # |________________|________
    # C2x2______________________|--6(-),7(+)->2(-),3(+)
    rdm= contract(C2x2, C2x2.negate_signature(), ([0,1,2,3,4,5],[0,1,2,3,4,5]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = permute(rdm,(0,2,1,3))
    
    # normalize and symmetrize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    if force_cpu:
        rdm= rdm.to(env.device)

    return rdm

def rdm2x2_NNN(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param force_cpu: compute on cpu
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{2x1}^{diag}` of 2 sites 
    that are next-nearest neighbours (along diagonal) using strategy:

        1. compute upper left corner using double-layer on-site tensor
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix

    ::

        C--T-----T-----C = C2x2---C2x2c
        |  |     |     |   |      |
        T--a^+a--a^+a--T   C2x2c--C2x2
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    inside C2x2 tensors are left uncontracted and given in the following order::
        
        s0 c
        c  s1

    """
    who= "rdm2x2_NNN"
    if force_cpu:
        C = env.C[env.keyC].to('cpu')
        T = env.T[env.keyT].to('cpu')
        a = state.site().to('cpu')
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = state.site()

    # C----T--3(+)       /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |\6(+),7(-)   /4(+),5(-)
    # 0(+) 1(+),2(-)     /1(+)
    C2x2= open_C2x2_LU(a,C,T, verbosity=verbosity)

    # C----T---3(+)      /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |
    # 0(+) 1(+),2(-)     /1(+)
    C2x2c= closed_C2x2_LU(a,C,T, verbosity=verbosity)

    # build upper part C2x2 -- C2x2c
    #
    # C--------T-------------3(+->-) 3(+)--------T-----C
    # |        |                                 |/----|--6(+),7(-)
    # T-------a*a--4(+1->-),5(-1->+) 4(+),5(-)--a*a----T
    # |        |                                 |     |
    # 0(+1->-) 1(+1->-),2(-1->+)            1(+),2(-)  0(+)
    #                                     ->4(+),5(-)  ->3(+)
    C2x2= contract(C2x2c.negate_signature(), C2x2, ([3,4,5],[3,4,5]))

    #                        C2x2_____________|--6(+),7(-)->0(+),1(-)
    #                        |                |
    #                        0(-),1(-),2(+)   3(+),4(+),5(-)
    #                        3(+),4(+),5(-)   0(-),1(-),2(+)
    #                        |________________|
    #  2(+),3(-)<-6(+),7(-)--|____________C2x2|
    rdm= contract(C2x2, C2x2, ([0,1,2,3,4,5],[3,4,5,0,1,2]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # and normalize
    rdm = permute(rdm,(0,2,1,3))

    # normalize and symmetrize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    if force_cpu:
        rdm= rdm.to(env.device)

    return rdm

def rdm2x2(state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param sym_pos_def: make final density matrix symmetric and non-negative (default: False)
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type verbosity: int
    :return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem using strategy:

        1. compute upper left corner
        2. construct upper half of the network
        3. contract upper and lower half (identical to upper) to obtain final reduced 
           density matrix
    
    ::

        C--T-----T-----C = C2x2--C2x2
        |  |     |     |   |     |
        T--a^+a--a^+a--T   C2x2--C2x2
        |  |     |     |
        T--a^+a--a^+a--T
        |  |     |     |
        C--T-----T-----C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    are left uncontracted and given in the following order::
        
        s0 s1
        s2 s3

    """
    who= "rdm2x2"
    if force_cpu:
        C = env.C[env.keyC].cpu()
        T = env.T[env.keyT].cpu()
        a = state.site().cpu()
    else:
        C = env.C[env.keyC]
        T = env.T[env.keyT]
        a = state.site()

    # C----T--3(+)       /2(+)
    # |    |
    # T---a*a--4(+),5(-) /3(+)
    # |    |\6(+),7(-)   /4(+),5(-)
    # 0(+) 1(+),2(-)     /1(+)
    C2x2= open_C2x2_LU(a,C,T, verbosity=verbosity)

    # build upper part C2x2 -- C2x2
    #
    #             /6(+),7(-)->3(+),4(-)  
    # C--------T-------------3(+) 3(+->-)--------T-----C
    # |        |/                                |/----|--8(+->-),9(-1->+)
    # T-------a*a--4(+),5(-)  4(+->-),5(-1->+)--a*a----T
    # |        |                                 |     |
    # 0(+)     1(+),2(-)             1(+->-),2(-1->+)  0(+->-)
    #                                     ->6(-),7(+)  ->5(-)
    C2x2= contract(C2x2, C2x2.negate_signature(), ([3,4,5],[3,4,5]))

    # construct reduced density matrix by contracting lower and upper halfs
    #
    #  0(+),1(-)<-3(+),4(-)--C2x2_____________|--8(-),9(+)->2(-),3(+)
    #                        |                |
    #                        0(+),1(+),2(-)   5(-),6(-),7(+)
    #                        5(-),6(-),7(+)   0(+),1(+),2(-)
    #                        |________________|
    #  6(-),7(+)<-8(-),9(+)--|____________C2x2|--0(+),1(-)->4(+),5(-)
    rdm= contract(C2x2, C2x2, ([0,1,2,5,6,7],[5,6,7,0,1,2]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # and normalize
    rdm = permute(rdm,(0,2,6,4, 1,3,7,5))

    # normalize and symmetrize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    return rdm