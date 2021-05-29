import logging
from tn_interface_abelian import contract, permute, conj

log= logging.getLogger('peps.ctm.generic_abelian.rdm')

def _sym_pos_def_matrix(rdm, sym_pos_def=False, verbosity=0, who="unknown"):
    rdm_asym= 0.5*(rdm-rdm.transpose((1,0)).conj())
    rdm= 0.5*(rdm+rdm.transpose((1,0)).conj())
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
    assert rdm.get_ndim()%2==0, "invalid rank of RDM"
    nsites= rdm.get_ndim()//2
    # rdm, lo_bra= rdm.group_legs(tuple(nsites+i for i in range(nsites)), new_s=1)
    # rdm, lo_ket= rdm.group_legs(tuple(i for i in range(nsites)), new_s=-1)
    rdm= rdm.fuse_legs(axes=(tuple(nsites+i for i in range(nsites)),\
        tuple(i for i in range(nsites))) )
    rdm= _sym_pos_def_matrix(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    # rdm= rdm.ungroup_leg(1, lo_bra)
    # rdm= rdm.ungroup_leg(0, lo_ket)
    rdm= rdm.unfuse_legs(axes=(0,1), inplace=True)
    return rdm

# CONVENTION:
#
# when grouping indices, environment index always preceeds aux-indices of 
# double-layer on-site tensor

# ----- COMPONENTS ------------------------------------------------------------

def open_C2x2_LU(coord, state, env, verbosity=0):
    r= state.vertexToSite(coord)
    C = env.C[(state.vertexToSite(r),(-1,-1))]
    T1 = env.T[(state.vertexToSite(r),(0,-1))]
    T2 = env.T[(state.vertexToSite(r),(-1,0))]

    # C--10--T1--2
    # 0      1
    c2x2= contract(C, T1, ([1],[0]))

    # C------T1--2->3
    # 0      1->2
    # 0
    # T2--2->1
    # 1->0
    c2x2= contract(T2, c2x2, ([0],[0]))

    # Open indices connecting Ts to on-site tensor. The unmerged index pairs are ordered 
    # as ket,bra
    #
    # C-------T1--3->4->5
    # |       2->2,3->3,4
    # T2--1->1,2
    # |
    # 0
    c2x2= c2x2.unfuse_legs(axes=(1,2),inplace=True)

    # C----------T1--5->3     C--------T1--3->1
    # |          3 \4->2      |   2<-4\| \
    # |          1            T--------a-------6->4
    # T2----1 2--a--4->6      |\       |  2
    # |\2->1     |\0->4       | \      |  1 /0->5 
    # |          3->5         |  \1 2-----a*--4->7
    # 0                       |        |  3->6 
    #                         0     3<-5
    #                         
    #
    c2x2= contract(c2x2, state.site(r), ([1,3], [2,1]))
    c2x2= contract(c2x2, conj(state.site(r)), ([1,2], [2,1]))


    # C----T--1->3                   C----T--2
    # |    |                     =>  |    |
    # T---a*a--4,7->4,5->4->3        T---a*a--3
    # |    |\2,5->6,7->5,6->4,5      |    |\4,5
    # 0    3,6->1,2->2               0    1 
    c2x2= permute(c2x2, (0,3,6,1,4,7,2,5))
    # c2x2, lo= c2x2.group_legs((4,5), new_s=-1)
    # c2x2, lo= c2x2.group_legs((1,2), new_s=-1)
    c2x2= c2x2.fuse_legs(axes=(0,(1,2),3,(4,5),6,7))

    return c2x2

def _group_legs_C2x2_LU(C2x2_LU):
    # C----T--2  => C----T---2
    # |    |        |    |    >--2->1(-1)
    # T---a*a--3    T---a*a--3 
    # |    |\4,5     \ /  \4,5->3,4->2,3
    # 0    1          0(-1)
    C2x2_LU, lo= C2x2_LU.group_legs((2,3), new_s=-1)
    C2x2_LU, lo= C2x2_LU.group_legs((0,1), new_s=-1)
    return C2x2_LU

def open_C2x2_LD(coord, state, env, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type verbosity: int
    :return: left-down enlarged corner with open physical indices
    :rtype: torch.tensor

    Computes lower-down enlarged corner centered on vertex ``coord`` by contracting 
    the following tensor network::

              s,s'
        |  | /
        T--a^+a--
        |  |
        C--T-----

    The physical indices `s` and `s'` of on-site tensor :math:`a` at vertex ``coord`` 
    and its hermitian conjugate :math:`a^\dagger` are left uncontracted
    """
    r= state.vertexToSite(coord)
    # 0
    # |
    # T(-1,0)--2->1
    # 1
    # 0
    # C(-1,1)--1->2
    c2x2 = contract(env.T[(r,(-1,0))],env.C[(r,(-1,1))],([1],[0]))
    if verbosity>0:
        print("c2x2=TC "+str(c2x2))
    # 0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T(0,1)--2->3
    c2x2 = contract(c2x2,env.T[(r,(0,1))],([2],[1]))
    if verbosity>0:
        print("c2x2=TCT "+str(c2x2))
    
    # Open indices connecting Ts to on-site tensor. The unmerged index pairs are ordered 
    # as ket,bra
    #
    # 0
    # T--1->1,2
    # |       2->2,3->3,4
    # C-------T--3->4->5
    # c2x2= c2x2.ungroup_leg(2, env.T[(r,(0,1))]._leg_fusion_data[0])
    # c2x2= c2x2.ungroup_leg(1, env.T[(r,(-1,0))]._leg_fusion_data[2])
    c2x2= c2x2.unfuse_legs(axes=(1,2),inplace=True)

    # 0            0->4       0      5->3/4->2
    # |       5<-1/           T------a-------6->4
    # T-----1 2--a--4->6      |\     |     1->6/0->5
    # |\2->1     3            | \1 2-------a*--4->7
    # |          3/--4->2     |      |     3
    # C----------T--5->3      |      \----\2
    #                         C------------T--3->1
    #
    c2x2= contract(c2x2, state.site(r), ([1,3], [2,3]))
    c2x2= contract(c2x2, conj(state.site(r)), ([1,2], [2,3]))
    
    # 0    3,6->1,2->1                 0    1  4,5
    # |    |/--2,5->6,7->5,6->4,5  =>  |    | /
    # T---a*a--4,7->4,5->4->3          T---a*a--3
    # |    |                           |    |
    # C----T--1->3                     C----T---2
    c2x2= permute(c2x2, (0,3,6,1,4,7,2,5))
    # c2x2, lo= c2x2.group_legs((4,5), new_s=-1)
    # c2x2, lo= c2x2.group_legs((1,2), new_s=1)
    c2x2= c2x2.fuse_legs(axes=(0,(1,2),3,(4,5),6,7))
    if verbosity>0:
        print("c2x2=TCTa*a "+str(c2x2))

    return c2x2

def _group_legs_C2x2_LD(C2x2_LD):
    # 0    1  4,5 => (+1)0   4,5->3,4->2,3
    # |    | /          / \ /
    # T---a*a--3       T-a*a--3
    # |    |           |  |    >2->1(-1)
    # C----T---2       C--T---2
    C2x2_LD, lo= C2x2_LD.group_legs((2,3), new_s=-1)
    C2x2_LD, lo= C2x2_LD.group_legs((0,1), new_s=1)
    return C2x2_LD

def open_C2x2_RU(coord, state, env, verbosity=0):
    r= state.vertexToSite(coord)
    C = env.C[(r,(1,-1))]
    T1 = env.T[(r,(1,0))]
    T2 = env.T[(r,(0,-1))]

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    c2x2 =contract(C, T1, ([1],[0]))

    # 0--T2--2 0--C
    #    1        |
    #       2<-1--T1
    #           3<-2
    c2x2 =contract(T2, c2x2, ([2],[0]))

    #   0--T2-------C
    # 1,2<-1        |
    #  3,4<-2,3<-2--T1
    #         5<-4<-3
    # c2x2= c2x2.ungroup_leg(2, T1._leg_fusion_data[1])
    # c2x2= c2x2.ungroup_leg(1, T2._leg_fusion_data[1])
    c2x2= c2x2.unfuse_legs(axes=(1,2),inplace=True)

    #     0--T2------C          0--T2--------C
    # 1<-2--/1       |      3<-5--/-a------\ |
    # 4<-0--\1  2<-4\|           1  |\4->2  \T1 
    #  5<-2--a--4 3--T1     5<-0\1  |       /|
    #     6<-3    3<-5     6<-2--a*--4 2---/ |
    #                         7<-3  6->4  1<-3
    c2x2 =contract(c2x2, state.site(r), ([1,3],[1,4]))
    c2x2 =contract(c2x2, conj(state.site(r)), ([1,2],[1,4]))

    #          0--T2----C  =>              0--T2----C
    #             |     |                     |     |
    #  1,2<-3,6--a*a----T1       (+1)1<-1,2<-a*a----T1
    #    6,7<-2,5/|     |     4,5<-5,6<-6,7<-/|     |
    #      4,5<-4,7  3<-1         (-1)3<-4<-4,5  2<-3
    #
    c2x2= permute(c2x2, (0,3,6,1,4,7,2,5))
    # c2x2, lo= c2x2.group_legs((4,5), new_s=-1)
    # c2x2, lo= c2x2.group_legs((1,2), new_s=1)
    c2x2= c2x2.fuse_legs(axes=(0,(1,2),3,(4,5),6,7))

    return c2x2

def _group_legs_C2x2_RU(C2x2_RU):
    #  0--T-----C  =>           0---T----C
    #     |     |       (+1)0--<    |    |
    # 1--a*a----T               1--a*a---T
    # 4,5/|     |     2,3<-3,4<-4,5/  \ /
    #     3     2               (-1)1<-2
    C2x2_RU, lo= C2x2_RU.group_legs((2,3), new_s=-1)
    C2x2_RU, lo= C2x2_RU.group_legs((0,1), new_s=1)
    return C2x2_RU

def open_C2x2_RD(coord, state, env, verbosity=0):
    r= state.vertexToSite(coord)
    C = env.C[(r,(1,1))]
    T1 = env.T[(r,(0,1))]
    T2 = env.T[(r,(1,0))]

    #    1<-0        0
    # 2<-1--T1--2 1--C
    c2x2 = contract(C, T1, ([1],[2]))

    #            0
    #         1--T2
    #            2
    #    2<-1    0
    # 3<-2--T1---C
    c2x2 = contract(T2, c2x2, ([2],[0]))

    #                0   
    #        1,2<-1--T2
    # 3,4<-2,3<-2    | 
    #  5<-4<-3--T1---C
    # c2x2= c2x2.ungroup_leg(2, T1._leg_fusion_data[0])
    # c2x2= c2x2.ungroup_leg(1, T2._leg_fusion_data[1])
    c2x2= c2x2.unfuse_legs(axes=(1,2),inplace=True)

    #    5<-1          0         6<-1          0
    # 6<-2--a--4 1-----T2     7<-2--a*--4 1----T2
    #  4<-0/3     1<-2/|       5<-0/3         /| 
    #       3/4->2     |      3<-5  2        / |
    # 3<-5--T1---------C   4<-6--a--|-------/  |
    #                       2<-4/| /           |
    #                      1<-3--T1------------C
    c2x2 = contract(c2x2, state.site(r), ([1,3],[4,3]))
    c2x2 = contract(c2x2, conj(state.site(r)), ([1,2],[4,3]))

    #      1,2<-3,6    0  =>          (+1)1<-1,2   0
    #  6,7<-2,5\  |    |        4,5<-5,6<-6,7\|    |
    #  4,5<-4,7--a*a---T2     (+1)3<-4<-4,5--a*a---T2
    #             |    |                      |    |
    #       3<-1--T1---C                2<-3--T1---C
    c2x2= permute(c2x2, (0,3,6,1,4,7,2,5))
    # c2x2, lo= c2x2.group_legs((4,5), new_s=1)
    # c2x2, lo= c2x2.group_legs((1,2), new_s=1)
    c2x2= c2x2.fuse_legs(axes=(0,(1,2),3,(4,5),6,7))

    return c2x2

def _group_legs_C2x2_RD(C2x2_RD):
    #     1     0  =>              (+1)0
    # 4,5\|     |     2,3<-3,4<-4,5\ /  \
    # 3--a*a----T               3--a*a---T
    #     |     |    (+1)1<-2--<    |    |
    #  2--T-----C               2---T----C
    C2x2_RD, lo= C2x2_RD.group_legs((2,3), new_s=1)
    C2x2_RD, lo= C2x2_RD.group_legs((0,1), new_s=1)
    return C2x2_RD

# ----- 1-site RDM ------------------------------------------------------------

def rdm1x1(coord, state, env, sym_pos_def=False, verbosity=0):
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
    who= "rdm1x1"
    r= state.vertexToSite(coord)
    rdm= open_C2x2_LD(r, state, env, verbosity=verbosity)
    # rdm= _group_legs_C2x2_LD(rdm)


    # C(-1,-1)--1 0--T(0,-1)--2  => C---T--2->1(-1)
    # 0              1               \ /
    #                                 0(-1)
    C2x1_LU= contract(env.C[(r,(-1,-1))], env.T[(r,(0,-1))],([1],[0]))
    # C2x1_LU, lo= C2x1_LU.group_legs((0,1), new_s=-1) 

    # C2x1_LU--1->0 => C2x1_LU--0
    # |                |         \0(-1)
    # 0      2,3       |___      /
    # 0__ _/           |rdm|----1
    # |    |               \2,3->1,2
    # |rdm_|--1 <-NOTE: contains both env index and double layer aux-indices)
    #rdm= contract(C2x1_LU, rdm, ([0],[0]))
    #rdm, lo= rdm.group_legs((0,1), new_s=-1)
    # C----T--2->0
    # 0    1
    # 0    1  4,5->3,4
    # |    | /
    # T---a*a--3->2
    # |    |
    # C----T---2->1
    rdm= contract(C2x1_LU, rdm, ([0,1],[0,1]))

    if verbosity>0:
        print("rdm=CTCTaT "+str(rdm))

    #    1<-0       =>      
    # 2<-1--T(1,0)            1
    #       2              2--T(1,0)
    #       0          0--<   |
    # 0<-1--C(1,1)         0--C(1,1)
    E= contract(env.C[(r,(1,1))], env.T[(r,(1,0))], ([0],[2]))
    # E, lo= E.group_legs((0,2), new_s=1)

    #    0--C(1,-1) =>          0--C
    #       1           (+1)0--<   |
    #       1                   1--E
    # 1<-0--E
    # E= contract(env.C[(r,(1,-1))], E, ([1],[1]))
    # E, lo= E.group_legs((0,1), new_s=1)
    #    0--C
    #       1 
    #       1  
    #    2--T(1,0)
    #       |
    #       |
    # 1<-0--C(1,1)
    E= contract(env.C[(r,(1,-1))], E, ([1],[1]))

    if verbosity>0:
        print("rdm=CTC "+str(E))

    # C(-1,-1)--T(0,-1)--------\       /--C(1,-1)
    # |         |              |       |  |
    # |         |/23->01(s,s') |       |  |
    # T(-1,0)---a--------------|--0 0--|--T(1,0) 
    # |         |              |       |  | 
    # |         |              |       |  |
    # C(-1,1)---T(0,1)---------/       \--C(1,1)
    # rdm = contract(rdm,E,([0],[0]))
    # C----T------0 0--C
    # |    |           |
    # |    |  3,4      | 
    # |    | /         |
    # T---a*a-----2 2--T
    # |    |           |
    # C----T------1 1--C
    rdm = contract(rdm,E,([0,2,1],[0,2,1]))

    if verbosity>0:
        print("rdm=CTCTaTCTC "+str(rdm))

    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

# ----- 2-site RDM ------------------------------------------------------------

def rdm2x1(coord, state, env, sym_pos_def=False, verbosity=0):
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
    who="rdm2x1"
    #----- building C2x2_LU ----------------------------------------------------
    C2x2_LU= open_C2x2_LU(coord, state, env, verbosity=verbosity)
    # C2x2_LU= _group_legs_C2x2_LU(C2x2_LU)

    if verbosity>0:
        print(f"C2X2 LU {coord} -> f{state.vertexToSite(coord)} (-1,-1): {C2x2_LU}")

    #----- building C2x1_LD ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord),(-1,1))]
    T2 = env.T[(state.vertexToSite(coord),(0,1))]

    #                    0(+1)
    # 0       0->1      / \
    # C--1 1--T2--2 => C---T--2->1
    C2x1_LD= contract(C, T2, ([1],[1]))
    # C2x1_LD, lo= C2x1_LD.group_legs((0,1), new_s=1)

    if verbosity>0:
        print(f"C2X1 LD {coord} -> {state.vertexToSite(coord)} (-1,1): {C2x1_LD}")

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2_LU--1 
    # |\23
    # 0
    # 0
    # C2x1_LD--1->0
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    # left_half= contract(C2x1_LD, C2x2_LU, ([0],[0]))
    # C----T--2->1  
    # |    |     
    # T---a*a--3->2  
    # |    |\4,5->3,4 
    # 0    1     
    # 0    1
    # C----T2--2->0
    left_half= contract(C2x1_LD, C2x2_LU, ([0,1],[0,1]))

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RU= open_C2x2_RU(shift_r, state, env, verbosity=verbosity)
    # C2x2_RU= _group_legs_C2x2_RU(C2x2_RU)
    
    if verbosity>0:
        print(f"C2X2 RU {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            f"(1,-1): {C2x2_RU}")

    #----- building C2x1_RD ----------------------------------------------------
    C = env.C[(shift_r,(1,1))]
    T1 = env.T[(shift_r,(0,1))]

    #                             0(+1)
    #    1<-0        0           / \  
    # 2<-1--T1--2 1--C => 1<-2--T1--C
    C2x1_RD= contract(C, T1, ([1],[2]))
    # C2x1_RD, lo= C2x1_RD.group_legs((0,1), new_s=1)

    if verbosity>0:
        print(f"C2X1 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shitf_coord} "\
            +f"(1,1): {C2x1_RD}")

    #----- build right part C2x2_RU--C2x1_RD -----------------------------------
    # 1<-0--C2x2_RU
    #       |\23
    #       1
    #       0
    # 0<-1--C2x1_RD
    # right_half =contract(C2x1_RD, C2x2_RU, ([0],[1]))
    #       1<-0--T2----C
    #             |     |
    #       2<-1-a*a----T1
    #   3,4<-4,5-/|     |
    #         (-1)3     2
    #             1     0
    #       0<-2--T1----C
    right_half =contract(C2x1_RD, C2x2_RU, ([0,1],[2,3]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2_LU--1 1----C2x2_RU
    # |\23->01        |\23
    # |               |    
    # C2x1_LD--0 0----C2x1_RD
    # rdm =contract(left_half,right_half,([0,1],[0,1]))
    # C2x2_LU--1 1----C2x2_RU
    # |     \--2 2--/ |
    # |\34->01        |\34->23
    # |               |    
    # C2x1_LD--0 0----C2x1_RD
    rdm =contract(left_half,right_half,([0,1,2],[0,1,2]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "bra"
    # 0123->0213
    # and normalize
    rdm= permute(rdm, (0,2,1,3))
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    return rdm

def rdm1x2(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies position of 1x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
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
    who="rdm1x2"
    #----- building C2x2_LU ----------------------------------------------------
    C2x2_LU= open_C2x2_LU(coord, state, env, verbosity=verbosity)
    # C2x2_LU= _group_legs_C2x2_LU(C2x2_LU)
    # C2x2_LU--1
    # | \2,3 
    # 0 

    if verbosity>0:
        print(f"C2X2 LU {coord} -> {state.vertexToSite(coord)} (-1,-1): {C2x2_LU}")

    #----- building C1x2_RU ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord),(1,-1))]
    T1 = env.T[(state.vertexToSite(coord),(1,0))]

    # 0--C   =>         0--C
    #    1      (+1)0--<   |
    #    0              1--T1
    # 1--T1             1<-2
    #    2
    C1x2_RU= contract(C, T1, ([1],[0]))
    # C1x2_RU, lo= C1x2_RU.group_legs((0,1), new_s=1)

    if verbosity>0:
        print(f"C1X2 RU {coord} -> {state.vertexToSite(coord)} (1,-1): {C1x2_RU}")

    #----- build upper part C2x2_LU--C1x2_RU -----------------------------------
    # C2x2_LU--1 0--C1x2_RU
    # |\23          |
    # 0->1          1->0
    # upper_half =contract(C1x2_RU, C2x2_LU, ([0],[1]))
    # C----T--------2 0--C  
    # |    |             |
    # T---a*a-------3 1--T1
    # |    |\4,5->3,4    2->0
    # 0->1 1->2
    upper_half =contract(C1x2_RU, C2x2_LU, ([0,1],[2,3]))

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_LD= open_C2x2_LD(shift_r, state, env, verbosity=verbosity)
    # C2x2_LD= _group_legs_C2x2_LD(C2x2_LD)

    if verbosity>0:
        print(f"C2X2 LD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(-1,1): {C2x2_LD}")

    #----- building C2x2_RD ----------------------------------------------------
    C = env.C[(shift_r,(1,1))]
    T2 = env.T[(shift_r,(1,0))]

    #       0   =>
    #    1--T2               (+1)0  
    #       2              2<-1--T2
    #       0      (+1)1--<      |
    # 2<-1--C              1<-2--C
    C1x2_RD= contract(T2, C, ([2],[0]))
    C1x2_RD= permute(C1x2_RD, (0,2,1))
    # C1x2_RD, lo= C1x2_RD.group_legs((1,2), new_s=1)

    if verbosity>0:
        print(f"C1X2 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(1,1): {C1x2_RD}")

    #----- build lower part C2x2_LD--C1x2_RD -----------------------------------
    # 0->1(+1)      0
    # |/23          |
    # C2x2_LD--1 1--C1x2_RD 
    # lower_half =contract(C1x2_RD, C2x2_LD, ([1],[1]))
    # 0->1 1->2 4,5->3,4
    # |    | --/          0
    # T---a*a--------3 2--T2
    # |    |              |
    # C----T---------2 1--C
    lower_half =contract(C1x2_RD, C2x2_LD, ([1,2],[2,3]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C1x2_RU
    # |\23->01     |
    # 1            0
    # 1            0
    # |/23         |
    # C2x2_LD------C1x2_RD
    # rdm =contract(upper_half,lower_half,([0,1],[0,1]))
    # C2x2_LU------C1x2_RU
    # | | \34->01     |
    # 1 2             0
    # 1 2             0
    # | | /34         |
    # C2x2_LD------C1x2_RD
    rdm =contract(upper_half,lower_half,([0,1,2],[0,1,2]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "bra"
    # 0123->0213
    # and normalize
    rdm= permute(rdm, (0,2,1,3))
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    return rdm

# ----- 2x2-cluster RDM -------------------------------------------------------

def rdm2x2(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem 
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
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
    who= "rdm2x2"
    #----- building C2x2_LU ----------------------------------------------------
    C2x2_LU= open_C2x2_LU(coord, state, env, verbosity=verbosity)
    # C2x2_LU= _group_legs_C2x2_LU(C2x2_LU)

    if verbosity>0:
        print(f"C2X2 LU {coord} -> {state.vertexToSite(coord)} (-1,-1):\n{C2x2_LU}")

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RU= open_C2x2_RU(shift_r, state, env, verbosity=verbosity)
    # C2x2_RU= _group_legs_C2x2_RU(C2x2_RU)
    
    if verbosity>0:
        print(f"C2X2 RU {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            f"(1,-1):\n{C2x2_RU}")

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU              C2x2_LU------C2x2_RU
    # |\23->12      |\23->45   & permute |\12->23      |\45
    # 0             1->3                 0             3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?  
    # upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))
    # upper_half = permute(upper_half, (0,3,1,2,4,5))
    # C2x2_LU--2 0--C2x2_RU                    C2x2_LU------C2x2_RU
    # |_____|--3 1--|_____|                    |_____|       |____|
    # |  |  \45->23  |    |\45->67  & permute  |  |  \23->45 |    |\67
    # 0  1           3->5 2->4                 0  1          5->3 4->2
    upper_half = contract(C2x2_LU, C2x2_RU, ([2,3],[0,1]))
    upper_half = permute(upper_half, (0,1,4,5,2,3,6,7))

    #----- building C2x2_RD ----------------------------------------------------
    vec = (1,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RD= open_C2x2_RD(shift_r, state, env, verbosity=verbosity)
    # C2x2_RD= _group_legs_C2x2_RD(C2x2_RD)

    #    0
    #    |/23
    # 1--C2x2
    if verbosity>0:
        print(f"C2X2 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(1,1):\n{C2x2_RD}")

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_LD= open_C2x2_LD(shift_r, state, env, verbosity=verbosity)
    # C2x2_LD= _group_legs_C2x2_LD(C2x2_LD)

    if verbosity>0:
        print(f"C2X2 LD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(-1,1): {C2x2_LD}")

    #----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->3                 0             3->1
    # |/23->12      |/23->45   & permute |/12->23      |/45
    # C2x2_LD--1 1--C2x2_RD              C2x2_LD------C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?  
    # lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))
    # lower_half = permute(lower_half, (0,3,1,2,4,5))
    # 0   1          1->5 0->4               0   1          5->3 4->2
    # |___|_/45->23  |____|/23->67 & permute |___|_/23->45  |____|/67
    # |     |--3 3--|     |                  |     |       |     |
    # C2x2_LD--2 2--C2x2_RD                  C2x2_LD-------C2x2_RD
    lower_half = contract(C2x2_LD, C2x2_RD, ([2,3],[2,3]))
    lower_half = permute(lower_half, (0,1,4,5,2,3,6,7))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |  |\45->01     |  |\67->23
    # 0  1            3  2
    # 0  1            3  2  
    # |  |/45         |  |/67->67
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half,lower_half,([0,1,2,3],[0,1,2,3]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "bra"
    # 01234567->02461357
    # and normalize
    rdm= permute(rdm, (0,2,4,6,1,3,5,7))
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm