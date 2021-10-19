import torch
from ctm.generic.env import ENV
from tn_interface import contract, einsum
from tn_interface import contiguous, view, permute
from tn_interface import conj


def rdm2x2_id_overlap(coord, state, state2, env, force_cpu=False, verbosity=0):
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

    Special case where the operator = the identity. This funciton thus computes the norm of the wavefunction < psi | psi >
    """
    who = "rdm2x2"
    # ----- building C2x2_LU ----------------------------------------------------
    if force_cpu:
        C = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(coord), (0, -1))].cpu()
        T2 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        a_1layer = state.site(coord).cpu()
        a_1layer2 = state2.site(coord).cpu()
    else:
        C = env.C[(state.vertexToSite(coord), (-1, -1))]
        T1 = env.T[(state.vertexToSite(coord), (0, -1))]
        T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
        a_1layer = state.site(coord)
        a_1layer2 = state2.site(coord)
    dimsA = a_1layer.size()
    # contract all physical sites of this unit cell (index m)
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer2)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

    # C--10--T1--2
    # 0      1
    C2x2_LU = contract(C, T1, ([1], [0]))

    # C------T1--2->1
    # 0      1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2_LU = contract(C2x2_LU, T2, ([0], [0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 a--3
    # 2->1    2
    C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    # C2x2--1
    # |
    # 0
    C2x2_LU = contiguous(permute(C2x2_LU, (1, 2, 0, 3)))
    C2x2_LU = view(C2x2_LU, (T2.size(1) * a.size(2), T1.size(2) * a.size(3)))
    if verbosity > 0:
        print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shitf_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    if force_cpu:
        C = env.C[(state.vertexToSite(shitf_coord), (1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(shitf_coord), (1, 0))].cpu()
        T2 = env.T[(state.vertexToSite(shitf_coord), (0, -1))].cpu()
        a_1layer = state.site(shitf_coord).cpu()
        a_1layer2 = state2.site(shitf_coord).cpu()
    else:
        C = env.C[(state.vertexToSite(shitf_coord), (1, -1))]
        T1 = env.T[(state.vertexToSite(shitf_coord), (1, 0))]
        T2 = env.T[(state.vertexToSite(shitf_coord), (0, -1))]
        a_1layer = state.site(shitf_coord)
        a_1layer2 = state2.site(shitf_coord)
    dimsA = a_1layer.size()
    # contract all physical sites of this unit cell (index m)
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer2)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C2x2_RU = contract(C, T1, ([1], [0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2_RU = contract(C2x2_RU, T2, ([0], [2]))

    # 1<-2--T2------C
    #       3       |
    #       0       |
    # 2<-1--a--3 0--T1
    #    3<-2    0<-1
    C2x2_RU = contract(C2x2_RU, a, ([0, 3], [3, 0]))

    # permute 012334->1203
    # reshape (12)(03)->01
    # 0--C2x2
    #    |
    #    1
    C2x2_RU = contiguous(permute(C2x2_RU, (1, 2, 0, 3)))
    C2x2_RU = view(C2x2_RU, (T2.size(0) * a.size(1), T1.size(2) * a.size(2)))
    if verbosity > 0:
        print("C2X2 RU " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,-1): " + str(
            C2x2_RU.size()))

    # ----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    #
    # C2x2_LU--1 0--C2x2_RU
    # |              |
    # 0              1
    #
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?
    upper_half = contract(C2x2_LU, C2x2_RU, ([1], [0]))

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 1)
    shitf_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    if force_cpu:
        C = env.C[(state.vertexToSite(shitf_coord), (1, 1))].cpu()
        T1 = env.T[(state.vertexToSite(shitf_coord), (0, 1))].cpu()
        T2 = env.T[(state.vertexToSite(shitf_coord), (1, 0))].cpu()
        a_1layer = state.site(shitf_coord).cpu()
        a_1layer2 = state2.site(shitf_coord).cpu()
    else:
        C = env.C[(state.vertexToSite(shitf_coord), (1, 1))]
        T1 = env.T[(state.vertexToSite(shitf_coord), (0, 1))]
        T2 = env.T[(state.vertexToSite(shitf_coord), (1, 0))]
        a_1layer = state.site(shitf_coord)
        a_1layer2 = state2.site(shitf_coord)
    dimsA = a_1layer.size()
    # contract all physical sites of this unit cell (index m)
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer2)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

    #    1<-0       0
    # 2<-1--T1--2 1--C
    C2x2_RD = contract(C, T1, ([1], [2]))

    #         2<-0
    #      3<-1--T2
    #            2
    #    0<-1    0
    # 1<-2--T1---C
    C2x2_RD = contract(C2x2_RD, T2, ([0], [2]))

    #    2<-0    1<-2
    # 3<-1--a--3 3--T2
    #       2       |
    #       0       |
    # 0<-1--T1------C
    C2x2_RD = contract(C2x2_RD, a, ([0, 3], [2, 3]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2_RD = contiguous(permute(C2x2_RD, (1, 2, 0, 3)))
    C2x2_RD = view(C2x2_RD, (T2.size(0) * a.size(0), T1.size(1) * a.size(1)))

    #    0
    #    |
    # 1--C2x2
    if verbosity > 0:
        print("C2X2 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,1): " + str(
            C2x2_RD.size()))

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shitf_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    if force_cpu:
        C = env.C[(state.vertexToSite(shitf_coord), (-1, 1))].cpu()
        T1 = env.T[(state.vertexToSite(shitf_coord), (-1, 0))].cpu()
        T2 = env.T[(state.vertexToSite(shitf_coord), (0, 1))].cpu()
        a_1layer = state.site(shitf_coord).cpu()
        a_1layer2 = state2.site(shitf_coord).cpu()
    else:
        C = env.C[(state.vertexToSite(shitf_coord), (-1, 1))]
        T1 = env.T[(state.vertexToSite(shitf_coord), (-1, 0))]
        T2 = env.T[(state.vertexToSite(shitf_coord), (0, 1))]
        a_1layer = state.site(shitf_coord)
        a_1layer2 = state2.site(shitf_coord)
    dimsA = a_1layer.size()
    # contract all physical sites of this unit cell (index m)
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer2)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

    # 0->1
    # T1--2
    # 1
    # 0
    # C--1->0
    C2x2_LD = contract(C, T1, ([0], [1]))

    # 1->0
    # T1--2->1
    # |
    # |       0->2
    # C--0 1--T2--2->3
    C2x2_LD = contract(C2x2_LD, T2, ([0], [1]))

    # 0        0->2
    # T1--1 1--a--3
    # |        2
    # |        2
    # C--------T2--3->1
    C2x2_LD = contract(C2x2_LD, a, ([1, 2], [1, 2]))

    # permute 0123->0213
    # reshape (02)(13)->01
    # 0
    # |
    # C2x2--1
    C2x2_LD = contiguous(permute(C2x2_LD, (0, 2, 1, 3)))
    C2x2_LD = view(C2x2_LD, (T1.size(0) * a.size(1), T2.size(1) * a.size(1)))
    if verbosity > 0:
        print("C2X2 LD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (-1,1): " + str(
            C2x2_LD.size()))

    # ----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->1
    # |             |
    # C2x2_LD--1 1--C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |             |
    # 0             1
    # 0             1
    # |             |
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    rdm = rdm.to(env.device)
    return rdm


def rdm1x1_id_overlap(coord, state, state2, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param state: 1-site operator to contract with the two physical indices of the rdm
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`. If an operator was provided,
    returns the expectation value of this operator (not normalized by the norm of the wavefunction).
    :rtype: torch.tensor

    Computes 1-site reduced density matrix :math:`\rho_{1x1}` centered on vertex ``coord`` by
    contracting the following tensor network::

        C--T-----C
        |  |     |
        T--A^+A--T
        |  |     |
        C--T-----C

    If no operator was provided, the physical indices `s` and `s'` of on-site tensor :math:`A`
    at vertex ``coord`` and it's hermitian conjugate :math:`A^\dagger` are left uncontracted.
    Else, they are contracted with the operator.
    """
    who = "rdm1x1"
    # C(-1,-1)--1->0
    # 0
    # 0
    # T(-1,0)--2
    # 1
    if force_cpu:
        C = env.C[(coord, (-1, -1))].cpu()
        T = env.T[(coord, (-1, 0))].cpu()
    else:
        C = env.C[(coord, (-1, -1))]
        T = env.T[(coord, (-1, 0))]
    rdm = contract(C, T, ([0], [0]))
    if verbosity > 0:
        print("rdm=CT " + str(rdm.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--2->1
    # 1
    # 0
    # C(-1,1)--1->2
    if force_cpu:
        C = env.C[(coord, (-1, 1))].cpu()
    else:
        C = env.C[(coord, (-1, 1))]
    rdm = contract(rdm, C, ([1], [0]))
    if verbosity > 0:
        print("rdm=CTC " + str(rdm.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T(0,1)--2->3
    if force_cpu:
        T = env.T[(coord, (0, 1))].cpu()
    else:
        T = env.T[(coord, (0, 1))]
    rdm = contract(rdm, T, ([2], [1]))
    if verbosity > 0:
        print("rdm=CTCT " + str(rdm.size()))
    # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
    #       Possibly reshape indices 1,2 of rdm, which are to be contracted with
    #       on-site tensor and contract bra,ket in two steps instead of creating
    #       double layer tensor
    #    /
    # --A1--
    #  /|s
    #
    # s'|/
    # --A2--
    #  /
    #
    if force_cpu:
        a_1layer = state.site(coord).cpu()
        a_1layer2 = state2.site(coord).cpu()
    else:
        a_1layer = state.site(coord)
        a_1layer2 = state2.site(coord)
    dimsA = a_1layer.size()
    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer2)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))
    # C(-1,-1)--0
    # |
    # |             0->2
    # T(-1,0)--1 1--a--3
    # |             2\45(s,s')
    # |             2
    # C(-1,1)-------T(0,1)--3->1
    rdm = contract(rdm, a, ([1, 2], [1, 2]))
    if verbosity > 0:
        print("rdm=CTCTa " + str(rdm.size()))
    # C(-1,-1)--0 0--T(0,-1)--2->0
    # |              1
    # |              2
    # T(-1,0)--------a--3->2
    # |              |\45->34(s,s')
    # |              |
    # C(-1,1)--------T(0,1)--1
    if force_cpu:
        T = env.T[(coord, (0, -1))].cpu()
    else:
        T = env.T[(coord, (0, -1))]
    rdm = contract(T, rdm, ([0, 1], [0, 2]))
    if verbosity > 0:
        print("rdm=CTCTaT " + str(rdm.size()))
    # C(-1,-1)--T(0,-1)--0 0--C(1,-1)
    # |         |             1->0
    # |         |
    # T(-1,0)---a--2
    # |         |\34(s,s')
    # |         |
    # C(-1,1)---T(0,1)--0->1
    if force_cpu:
        C = env.C[(coord, (1, -1))].cpu()
    else:
        C = env.C[(coord, (1, -1))]
    rdm = contract(C, rdm, ([0], [0]))
    if verbosity > 0:
        print("rdm=CTCTaTC " + str(rdm.size()))
    # C(-1,-1)--T(0,-1)-----C(1,-1)
    # |         |               0
    # |         |               0
    # T(-1,0)---a--2 1------T(1,0)
    # |         |\34->23(s,s')  2->0
    # |         |
    # C(-1,1)---T(0,1)--1
    if force_cpu:
        T = env.T[(coord, (1, 0))].cpu()
    else:
        T = env.T[(coord, (1, 0))]
    rdm = contract(T, rdm, ([0, 1], [0, 2]))
    if verbosity > 0:
        print("rdm=CTCTaTCT " + str(rdm.size()))
    # C(-1,-1)--T(0,-1)--------C(1,-1)
    # |         |                |
    # |         |                |
    # T(-1,0)---a--------------T(1,0)
    # |         |\23->12(s,s')   0
    # |         |                0
    # C(-1,1)---T(0,1)--1 1----C(1,1)
    if force_cpu:
        C = env.C[(coord, (1, 1))].cpu()
    else:
        C = env.C[(coord, (1, 1))]
    rdm = contract(rdm, C, ([0, 1], [0, 1]))
    if verbosity > 0:
        print("rdm=CTCTaTCTC " + str(rdm.size()))

    return rdm