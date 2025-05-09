import warnings
from functools import lru_cache
from itertools import product
import torch
from math import prod
from config import _torch_version_check
import config as cfg
from ctm.generic.env import ENV
from ctm.generic.ctm_components import c2x2_LU, c2x2_LD, c2x2_RU, c2x2_RD
from ctm.generic.ctm_projectors import ctm_get_projectors_from_matrices
import ctm.generic.corrf as corrf
try:
    import opt_einsum as oe
    from oe_ext.oe_ext import get_contraction_path, contract_with_unroll
except:
    oe=False
    warnings.warn("opt_einsum not available.")
from tn_interface import contract, einsum
from tn_interface import contiguous, view, permute
from tn_interface import conj
import logging

log = logging.getLogger(__name__)


def _cast_to_real(t, fail_on_check=False, warn_on_check=True, imag_eps=1.0e-8,\
    who="unknown", **kwargs):
    if t.is_complex():
        if abs(t.imag)/(abs(t.real)+1.0e-8) > imag_eps:
            if warn_on_check:
                log.warning(f"Unexpected imaginary part "+who+" "+str(t))
            if fail_on_check: 
                raise RuntimeError("Unexpected imaginary part "+who+" "+str(t))
        return t.real
    return t


def _sym_pos_def_matrix(rdm, sym_pos_def=False, verbosity=0, who="unknown", **kwargs):
    rdm_asym = 0.5 * (rdm - rdm.conj().t())
    rdm = 0.5 * (rdm + rdm.conj().t())
    if verbosity > 0:
        log.info(f"{who} norm(rdm_sym) {rdm.norm()} norm(rdm_asym) {rdm_asym.norm()}")
    if sym_pos_def:
        with torch.no_grad():
            if _torch_version_check("1.8.1"):
                D, U=  torch.linalg.eigh(rdm)
            else:
                D, U= torch.symeig(rdm, eigenvectors=True)
            if D.min() < 0:
                log.info(f"{who} max(diag(rdm)) {D.max()} min(diag(rdm)) {D.min()}")
                D = torch.clamp(D, min=0)
                rdm_posdef = U @ torch.diag(D) @ U.conj().t()
                rdm.copy_(rdm_posdef)
    norm= _cast_to_real(rdm.diagonal().sum(),who=who,**kwargs)
    rdm = rdm / norm
    return rdm


def _sym_pos_def_rdm(rdm, sym_pos_def=False, verbosity=0, who=None,  **kwargs):
    assert len(rdm.size()) % 2 == 0, "invalid rank of RDM"
    nsites = len(rdm.size()) // 2

    orig_shape = rdm.size()
    rdm = rdm.reshape(torch.prod(torch.as_tensor(rdm.size())[:nsites]), -1)

    rdm = _sym_pos_def_matrix(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm = rdm.reshape(orig_shape)
    return rdm


def rdm1x1(coord, state, env, mode='sl', operator=None, sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param mode: use double-layer ('dl') or layer-by-layer ('sl') approach when adding on-site tensors
    :param operator: 1-site operator to contract with the two physical indices of the rdm
    :param sym_pos_def: enforce hermiticity (always) and positive definiteness if ``True`` 
    :param force_cpu: compute on CPU
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type mode: str
    :type operator: torch.tensor
    :type sym_pos_def: bool
    :type force_cpu: bool
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
    Otherwise, they are contracted with the operator returning a scalar without enforcing 
    hermiticity or positive definiteness of reduced density matrix.
    """
    if mode=='sl':
        return rdm1x1_sl(coord, state, env, operator=operator, sym_pos_def=sym_pos_def,\
            force_cpu=force_cpu, verbosity=verbosity)
    else:
        return rdm1x1_dl(coord, state, env, operator=operator, sym_pos_def=sym_pos_def,\
            force_cpu=force_cpu, verbosity=verbosity)

def rdm1x1_dl(coord, state, env, operator=None, sym_pos_def=False, force_cpu=False, verbosity=0):
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
    # --A--
    #  /|s
    #
    # s'|/
    # --A--
    #  /
    #
    if force_cpu:
        a_1layer = state.site(coord).cpu()
    else:
        a_1layer = state.site(coord)
    dimsA = a_1layer.size()
    if operator == None:
        a = contiguous(einsum('mefgh,nabcd->eafbgchdmn', a_1layer, conj(a_1layer)))
        a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))
    else:
        a = contiguous(einsum('mefgh,nm,nabcd->eafbgchd', a_1layer, operator, conj(a_1layer)))
        a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

    # C(-1,-1)--0
    # |
    # |             0->2
    # T(-1,0)--1 1--a--3
    # |             2[\45(s,s')]
    # |             2
    # C(-1,1)-------T(0,1)--3->1
    rdm = contract(rdm, a, ([1, 2], [1, 2]))
    if verbosity > 0:
        print("rdm=CTCTa " + str(rdm.size()))

    # C(-1,-1)--0 0--T(0,-1)--2->0
    # |              1
    # |              2
    # T(-1,0)--------a--3->2
    # |              |[\45->34(s,s')]
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
    # |         |[\34(s,s')]
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

    # symmetrize and normalize
    if operator == None:
        rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        rdm = rdm.to(env.device)
    return rdm

def rdm1x1_sl(coord, state, env, operator=None, sym_pos_def=False, force_cpu=False, verbosity=0):
    # C1--(1)1 1(0)----T1--(3)9 9(0)----C2
    # 0(0)            (1,2)             8(1)
    # 0(0)          4  2  5             8(0)
    # |              \ 2  5              |
    # T4--(2)3 3-------a--|---10 10(1)---T2
    # |                |  |              |
    # |   (3)6 6----------a*--11 11(2)   |
    # 14(1)           15 16 \7          17(3)  
    # 14(0)           (0,1)             17(0)
    # C4--(1)12 12(2)--T3--(3)13 13(1)--C3
    #
    who="rdm1x1_sl"
    C1, C2, C3, C4, T1, T2, T3, T4= env.get_site_env_t(coord,state)
    a= state.site(coord)
    a_op= a if operator is None else torch.tensordot(operator,a,([1],[0]))
    t= C1, C2, C3, C4, T1, T2, T3, T4, a, a_op
    if force_cpu:
        t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T2= T2.view(T2.size(0),a.size(4),a.size(4),T2.size(2))
    T3= T3.view(a.size(3),a.size(3),T3.size(1),T3.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))


    contract_tn= C1,[0,1],T1,[1,2,5,9],T4,[0,14,3,6],a_op,[4,2,3,15,10],a.conj(),[7,5,6,16,11],\
        C2,[9,8],T2,[8,10,11,17],\
        C3,[17,13],T3,[15,16,12,13],C4,[14,12],[4,7]
    names= tuple(x.strip() for x in "C1, T1, T4, a_op, a*, C2, T2, C3, T3, C4".split(','))

    path, path_info= get_contraction_path(*contract_tn,names=names,path=None,who=who,\
        optimizer="default" if env.chi>1 else "auto")
    R= oe.contract(*contract_tn,optimize=path,backend='torch')

    # symmetrize and normalize
    if operator is None:
        R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    else:
        R= R.trace()
    if force_cpu:
        R = R.to(env.device)
    return R


def rdm2x1(coord, state, env, mode='sl', sym_pos_def=False, force_cpu=False, 
    unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False,
    verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies position of 2x1 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param mode: use double-layer ('dl') or layer-by-layer ('sl') approach when adding on-site tensors
    :param sym_pos_def: enforce hermiticity (always) and positive definiteness if ``True`` 
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type mode: str
    :type sym_pos_def: bool
    :param force_cpu: compute on CPU
    :type force_cpu: bool
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
    if mode=='sl':
        return rdm2x1_sl(coord,state,env,sym_pos_def=sym_pos_def,force_cpu=force_cpu,\
            unroll=unroll, 
            checkpoint_unrolled=checkpoint_unrolled, checkpoint_on_device=checkpoint_on_device,
            verbosity=verbosity)
    else:
        return rdm2x1_dl(coord,state,env,sym_pos_def=sym_pos_def,force_cpu=force_cpu,\
            verbosity=verbosity)

def rdm2x1_dl(coord, state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    who = "rdm2x1"
    # ----- building C2x2_LU ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (-1, -1))]
    T1 = env.T[(state.vertexToSite(coord), (0, -1))]
    T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
    dimsA = state.site(coord).size()
    a = einsum('mefgh,nabcd->eafbgchdmn', state.site(coord), conj(state.site(coord)))
    a = view(contiguous(a), \
             (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

    # C--10--T1--2
    # 0      1
    C2x2_LU = contract(C, T1, ([1], [0]))

    # C------T1--2->1
    # 0   1->0
    # 0
    # T2--2->3
    # 1->2
    C2x2_LU = contract(C2x2_LU, T2, ([0], [0]))

    # C-------T1--1->0
    # |       0
    # |       0
    # T2--3 1 a--3
    # 2->1    2\45
    C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = permute(C2x2_LU, (1, 2, 0, 3, 4, 5))
    C2x2_LU = view(contiguous(C2x2_LU), \
                   (T2.size(1) * a.size(2), T1.size(2) * a.size(3), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))

    # ----- building C2x1_LD ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (-1, 1))]
    T2 = env.T[(state.vertexToSite(coord), (0, 1))]

    # 0    0->1
    # C--1 1--T2--2
    C2x1_LD = contract(C, T2, ([1], [1]))

    # reshape (01)2->(0)1
    # 0
    # |
    # C2x1--1
    C2x1_LD = view(contiguous(C2x1_LD), (C.size(0) * T2.size(0), T2.size(2)))
    if verbosity > 0:
        print("C2X1 LD " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,1): " + str(C2x1_LD.size()))

    # ----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2_LU--1
    # |\23
    # 0
    # 0
    # C2x1_LD--1->0
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?
    left_half = contract(C2x1_LD, C2x2_LU, ([0], [0]))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C = env.C[(shift_coord, (1, -1))]
    T1 = env.T[(shift_coord, (1, 0))]
    T2 = env.T[(shift_coord, (0, -1))]
    dimsA = state.site(shift_coord).size()
    a = einsum('mefgh,nabcd->eafbgchdmn', state.site(shift_coord), conj(state.site(shift_coord)))
    a = view(contiguous(a), \
             (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

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
    #    45\0       |
    # 2<-1--a--3 0--T1
    #    3<-2    0<-1
    C2x2_RU = contract(C2x2_RU, a, ([0, 3], [3, 0]))

    # permute 012334->120345
    # reshape (12)(03)45->0123
    # 0--C2x2
    # 23/|
    #    1
    C2x2_RU = permute(C2x2_RU, (1, 2, 0, 3, 4, 5))
    C2x2_RU = view(contiguous(C2x2_RU), \
                   (T2.size(0) * a.size(1), T1.size(2) * a.size(2), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 RU " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (1,-1): " + str(
            C2x2_RU.size()))

    # ----- building C2x1_RD ----------------------------------------------------
    C = env.C[(shift_coord, (1, 1))]
    T1 = env.T[(shift_coord, (0, 1))]

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x1_RD = contract(C, T1, ([1], [2]))

    # reshape (01)2->(0)1
    C2x1_RD = view(contiguous(C2x1_RD), (C.size(0) * T1.size(0), T1.size(1)))

    #    0
    #    |
    # 1--C2x1
    if verbosity > 0:
        print("C2X1 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (1,1): " + str(
            C2x1_RD.size()))

    # ----- build right part C2x2_RU--C2x1_RD -----------------------------------
    # 1<-0--C2x2_RU
    #       |\23
    #       1
    #       0
    # 0<-1--C2x1_RD
    right_half = contract(C2x1_RD, C2x2_RU, ([0], [1]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2_LU--1 1----C2x2_RU
    # |\23->01        |\23
    # |               |
    # C2x1_LD--0 0----C2x1_RD
    rdm = contract(left_half, right_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 1, 3)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm2x1_sl(coord, state, env, sym_pos_def=False, force_cpu=False, unroll=False, 
    checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0):
    # C1--(1)1 1(0)----T1--(3)24 24(0)--T1_x--(3)12 12(0)----C2_x
    # 0(0)            (1,2)             (1,2)                13(1)
    # 0(0)          4  2  5         16  14 17                13(0)
    # |              \ 2  5           \ 14 17                |
    # T4--(2)3 3-------a--|----26 26----a_x-------15 15(1)---T2_x
    # |                |  |              |  |                |
    # |   (3)6 6----------a*---27 27-------a_x*---18 18(2)   |
    # 9(1)            10 11 \7          22 23 \19            21(3)
    # 9(0)            (0,1)             (0,1)                21(0)
    # C4--(1)8 8(2)----T3--(3)25 25(2)---T3_x---(3)20 20(1)--C3_x
    #
    who="rdm2x1_sl"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    C1, C2_x, C3_x, C4= env.C[(state.vertexToSite(coord),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,1))],\
        env.C[(state.vertexToSite(coord), (-1,1))]
    T1, T1_x, T2_x, T3, T3_x, T4= env.T[(state.vertexToSite(coord),(0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite(coord), (0,1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,1))],\
        env.T[(state.vertexToSite(coord), (-1,0))]
    t= C1, C2_x, C3_x, C4, T1, T1_x, T2_x, T3, T3_x, T4, a, a_x
    if force_cpu:
        t=(x.cpu() for x in t)
    
    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T2_x= T2_x.view(T2_x.size(0),a_x.size(4),a_x.size(4),T2_x.size(2))
    T3= T3.view(a.size(3),a.size(3),T3.size(1),T3.size(2))
    T3_x= T3_x.view(a_x.size(3),a_x.size(3),T3_x.size(1),T3_x.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))

    contract_tn= C1,[0,1],T1,[1,2,5,24],T4,[0,9,3,6],a,[4,2,3,10,26],a.conj(),[7,5,6,11,27],\
        C4,[9,8],T3,[10,11,8,25],\
        T1_x,[24,14,17,12],C2_x,[12,13],T2_x,[13,15,18,21],a_x,[16,14,26,22,15],a_x.conj(),[19,17,27,23,18],\
        C3_x,[21,20],T3_x,[22,23,25,20],[4,16,7,19]
    names= tuple(x.strip() for x in "C1, T1, T4, a, a*, C4, T3, T1_x, C2_x, T2_x, a_x, a_x*, C3_x, T3_x".split(','))
    #
    # This (typical) strategy is optimal, when X >> D^2 >> phys_dim
    #
    # path=((2, 5), (0, 12), (3, 11), (2, 10), (1, 9), (0, 8), (2, 5), (1, 6), (3, 5), (2, 4), (1, 3), (0, 2), (0, 1))
    path, path_info= get_contraction_path(*contract_tn,names=names,path=None,unroll=unroll,who=who,\
                                          optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,optimize=path,backend='torch',unroll=unroll,
        checkpoint_unrolled=checkpoint_unrolled,checkpoint_on_device=checkpoint_on_device,
        who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R


def rdm1x2(coord, state, env, mode='sl', sym_pos_def=False, force_cpu=False, 
    unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False,
    verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies position of 1x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param mode: use double-layer ('dl') or layer-by-layer ('sl') approach when adding on-site tensors
    :param sym_pos_def: enforce hermiticity (always) and positive definiteness if ``True`` 
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type mode: str
    :type sym_pos_def: bool
    :param force_cpu: compute on CPU
    :type force_cpu: bool
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
    if mode=='sl':
        return rdm1x2_sl(coord, state, env, sym_pos_def=sym_pos_def, force_cpu=force_cpu,
            unroll=unroll, 
            checkpoint_unrolled=checkpoint_unrolled, checkpoint_on_device=checkpoint_on_device,
            verbosity=verbosity)
    else:
        return rdm1x2_dl(coord, state, env, sym_pos_def=sym_pos_def, force_cpu=force_cpu,\
            verbosity=verbosity)

def rdm1x2_dl(coord, state, env, sym_pos_def=False, force_cpu=False, verbosity=0): 
    who = "rdm1x2"
    # ----- building C2x2_LU ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (-1, -1))]
    T1 = env.T[(state.vertexToSite(coord), (0, -1))]
    T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
    dimsA = state.site(coord).size()
    a = einsum('mefgh,nabcd->eafbgchdmn', state.site(coord), conj(state.site(coord)))
    a = view(contiguous(a), \
             (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

    # C--10--T1--2
    # 0   1
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
    # 2->1    2\45
    C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = permute(C2x2_LU, (1, 2, 0, 3, 4, 5))
    C2x2_LU = view(contiguous(C2x2_LU), \
                   (T2.size(1) * a.size(2), T1.size(2) * a.size(3), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))

    # ----- building C1x2_RU ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (1, -1))]
    T1 = env.T[(state.vertexToSite(coord), (1, 0))]

    # 0--C
    #    1
    #    0
    # 1--T1
    #    2
    C1x2_RU = contract(C, T1, ([1], [0]))

    # reshape (01)2->(0)1
    # 0--C1x2
    # 23/|
    #    1
    C1x2_RU = view(contiguous(C1x2_RU), (C.size(0) * T1.size(1), T1.size(2)))
    if verbosity > 0:
        print("C1X2 RU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (1,-1): " + str(C1x2_RU.size()))

    # ----- build upper part C2x2_LU--C1x2_RU -----------------------------------
    # C2x2_LU--1 0--C1x2_RU
    # |\23          |
    # 0->1          1->0
    upper_half = contract(C1x2_RU, C2x2_LU, ([0], [1]))

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C = env.C[(shift_coord, (-1, 1))]
    T1 = env.T[(shift_coord, (-1, 0))]
    T2 = env.T[(shift_coord, (0, 1))]
    dimsA = state.site(shift_coord).size()
    a = einsum('mefgh,nabcd->eafbgchdmn', state.site(shift_coord), conj(state.site(shift_coord)))
    a = view(contiguous(a), \
             (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

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
    # |        2\45
    # |        2
    # C--------T2--3->1
    C2x2_LD = contract(C2x2_LD, a, ([1, 2], [1, 2]))

    # permute 012345->021345
    # reshape (02)(13)45->0123
    # 0
    # |/23
    # C2x2--1
    C2x2_LD = permute(C2x2_LD, (0, 2, 1, 3, 4, 5))
    C2x2_LD = view(contiguous(C2x2_LD), \
                   (T1.size(0) * a.size(0), T2.size(2) * a.size(3), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 LD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (-1,1): " + str(
            C2x2_LD.size()))

    # ----- building C2x2_RD ----------------------------------------------------
    C = env.C[(shift_coord, (1, 1))]
    T2 = env.T[(shift_coord, (1, 0))]

    #       0
    #    1--T2
    #       2
    #       0
    # 2<-1--C
    C1x2_RD = contract(T2, C, ([2], [0]))

    # permute 012->021
    # reshape 0(12)->0(1)
    C1x2_RD = view(contiguous(permute(C1x2_RD, (0, 2, 1))), \
                   (T2.size()[0], C.size()[1] * T2.size()[1]))

    #    0
    #    |
    # 1--C1x2
    if verbosity > 0:
        print("C1X2 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (1,1): " + str(
            C1x2_RD.size()))

    # ----- build lower part C2x2_LD--C1x2_RD -----------------------------------
    # 0->1          0
    # |/23          |
    # C2x2_LD--1 1--C1x2_RD
    lower_half = contract(C1x2_RD, C2x2_LD, ([1], [1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C1x2_RU
    # |\23->01     |
    # 1            0
    # 1            0
    # |/23         |
    # C2x2_LD------C1x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 1, 3)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm1x2_sl(coord, state, env, sym_pos_def=False, force_cpu=False, unroll=False, 
            checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0):
    # C1--(1)1 1(0)----T1--(3)9 9(0)--------C2
    # 0(0)            (1,2)                 8(1)
    # 0(0)          4  2  5                 8(0)
    # |              \ 2  5                 |
    # T4----(2)3 3-----a--|---10 10(1)------T2
    # |                |  |                 |
    # |     (3)6 6--------a*--11 11(2)      |
    # 24(1)           25 26 \7             27(3)
    # 24(0)       16  25 26                27(0)
    # |             \ |   |                 |
    # T4_y(2)14 14---a_y--------22 22(1)----T2_y 
    # |               |   |                 |
    # |   (3)17 17-------a_y*---23 23(2)    |
    # |               15 18 \19             |
    # 13(1)           15 18                20(3)
    # 13(0)           (0,1)                20(0)
    # C4_y--(1)12 12(2)--T3_y--(3)21 21(1)--C3_y
    #
    who="rdm1x2_sl"
    a= state.site(coord)
    a_y= state.site( (coord[0],coord[1]+1) )
    C1, C2, C3_y, C4_y= env.C[(state.vertexToSite(coord),(-1,-1))],\
        env.C[(state.vertexToSite(coord), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+1) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,1))]
    T1, T2, T2_y, T3_y, T4, T4_y= env.T[(state.vertexToSite(coord),(0,-1))],\
        env.T[(state.vertexToSite(coord), (1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite(coord), (-1,0))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))]
    t= C1, C2, C3_y, C4_y, T1, T2, T2_y, T3_y, T4, T4_y, a, a_y
    if force_cpu:
        t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T2= T2.view(T2.size(0),a.size(4),a.size(4),T2.size(2))
    T2_y= T2_y.view(T2_y.size(0),a_y.size(4),a_y.size(4),T2_y.size(2))
    T3_y= T3_y.view(a_y.size(3),a_y.size(3),T3_y.size(1),T3_y.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))
    
    contract_tn= C1,[0,1],T1,[1,2,5,9],T4,[0,24,3,6],a,[4,2,3,25,10],a.conj(),[7,5,6,26,11],\
        T2,[8,10,11,27],C2,[9,8],\
        T4_y,[24,13,14,17],C4_y,[13,12],T3_y,[15,18,12,21],a_y,[16,25,14,15,22],a_y.conj(),[19,26,17,18,23],\
        C3_y,[20,21],T2_y,[27,22,23,20],[4,16,7,19]
    names= tuple(x.strip() for x in "C1, T1, T4, a, a*, T2, C2, T4_y, C4_y, T3_y, a_y, a_y*, C3_y, T2_y".split(','))
    #
    # This (typical) strategy is optimal, when X >> D^2 >> phys_dim
    #
    # path= ((1, 6), (0, 12), (3, 11), (2, 10), (1, 9), (0, 8), (2, 5), (1, 6), (3, 5), (2, 4), (1, 3), (0, 2), (0, 1))
    path, path_info= get_contraction_path(*contract_tn,names=names,path=None,unroll=unroll,who=who,\
                                          optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,optimize=path,backend='torch',unroll=unroll,
        checkpoint_unrolled=checkpoint_unrolled,checkpoint_on_device=checkpoint_on_device,
        who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R


def rdm2x2_NNN_11(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{NNN,11}` of two-site subsystem 
    across (1,1) diagonal specified by the vertex ``coord`` of its upper left corner using strategy:

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
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        s0 x
        x  s1
    """
    who = "rdm2x2_NNN_11"
    # ----- building C2x2_LU ----------------------------------------------------
    C2X2_LU= c2x2_LU(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2X2_RU ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord, state, env, mode='sl',verbosity=verbosity) 

    # ----- build upper part C2x2_LU--C2X2_RU -----------------------------------
    # C2x2_LU--1 0--C2X2_RU           C2x2_LU------C2X2_RU
    # |\23->12      |       & permute |\12->23     |
    # 0             1->3              0            1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2X2_RU ?
    upper_half = contract(C2X2_LU, C2X2_RU, ([1], [0]))
    upper_half = permute(upper_half, (0, 3, 1, 2))

    # ----- building C2X2_RD ----------------------------------------------------
    vec = (1, 1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LD= c2x2_LD(shift_coord, state, env, mode='sl', verbosity=verbosity)

    # ----- build lower part C2X2_LD--C2X2_RD -----------------------------------
    # 0             0->1
    # |             |/23
    # C2X2_LD--1 1--C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2X2_LD,C2x2_RD ?
    lower_half = contract(C2X2_LD, C2X2_RD, ([1], [1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |\23->01     |
    # 0            1
    # 0            1
    # |            |/23
    # C2X2_LD------C2x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 1, 3)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm


def rdm2x2_NNN_1n1(coord, state, env, sym_pos_def=False, force_cpu=False, 
    unroll=False, checkpoint_unrolled=False, checkpoint_on_device=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param sym_pos_def: enforce hermiticity (always) and positive definiteness if ``True``
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :param force_cpu: compute on CPU
    :type force_cpu: bool
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{NNN,1n1}` of two-site subsystem 
    across (1,-1) diagonal specified by the vertex ``coord`` of its lower left corner using strategy:

        1. compute four individual corners
        2. construct upper and lower half of the network
        3. contract upper and lower half to obtain final reduced density matrix

    ::

        C--T------------------T------------------C = C2x2_LU(coord+(0,-1))-C2x2(coord+(1,-1))
        |  |                  |                  |   |                     |
        T--A^+A(coord+(0,-1))-A^+A(coord+(1,-1))-T   C2x2_LD(coord)--------C2x2(coord+(1,0))
        |  |                  |                  | 
        T--A^+A(coord)--------A^+A(coord+(1,0))--T
        |  |                  |                  |
        C--T------------------T------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,-1)`` are left uncontracted and given in the same order::

        x  s1
        s0 x

    """
    if oe:
        return rdm2x2_NNN_1n1_oe(coord, state, env, sym_pos_def=sym_pos_def, force_cpu=force_cpu,\
            unroll=unroll, checkpoint_unrolled=checkpoint_unrolled, checkpoint_on_device=checkpoint_on_device,
            verbosity=verbosity)
    else:
        return rdm2x2_NNN_1n1_legacy(coord, state, env, sym_pos_def=sym_pos_def, force_cpu=force_cpu,\
            verbosity=verbosity)

def rdm2x2_NNN_1n1_legacy(coord, state, env, sym_pos_def=False, force_cpu=False, verbosity=0):
    who = "rdm2x2_NNN_1n1_legacy"
    # ----- building C2X2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord, state, env, mode='sl-open', verbosity=verbosity)

    # ----- build upper part C2x2_LU--C2X2_RU -----------------------------------
    # C2x2_LU--1 0--C2X2_RU
    # |             |\23
    # 0             1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2X2_RU ?
    upper_half = contract(C2X2_LU, C2X2_RU, ([1], [0]))

    # ----- building C2X2_RD ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)    

    # ----- building C2X2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build lower part C2X2_LD--C2X2_RD -----------------------------------
    # 0             0->3                 0            3->1
    # |/23->12      |          & permute |/12->23     |
    # C2X2_LD--1 1--C2X2_RD              C2X2_LD------C2X2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2X2_LD,C2X2_RD ?
    lower_half = contract(C2X2_LD, C2X2_RD, ([1], [1]))
    lower_half = permute(lower_half, (0, 3, 1, 2))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2X2_LU------C2X2_RU
    # |            |\23
    # 0            1
    # 0            1
    # |/23->01     |
    # C2X2_LD------C2X2_RD
    rdm = contract(lower_half, upper_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 1, 3)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm2x2_NNN_1n1_oe(coord, state, env, sym_pos_def=False, force_cpu=False, 
    unroll=False, checkpoint_unrolled=False, checkpoint_on_device=False,
    verbosity=0):
    # C1_ny--(1)1 1(0)----T1_ny--(3)36 36(0)----T1_xny--(3)18 18(0)----C2_xny
    # 0(0)               (1,2)                 (1,2)                   19(1)
    # 0(0)                2  5              22 20 23                   19(0)
    # |                   2  5               \ |  |                    |  
    # T4_ny----(2)3 3-- a_ny-|------37 37----a_xny6(1)----21 21(1)-----T2_xny
    # |                   |4 |                 |  |                    |
    # |        (3)6 6-------a*_ny---38 38-------a*_xny----24 24(2)     |
    # 15(1)               16 17               34 35 \25                33(3)
    # 15(0)           11  16 17               34 35                    33(0)
    # |                 \ |   |                |  |                    |
    # T4----(2)9 9--------a---------39 39-----a_x---------28 28(1)-----T2_x 
    # |                   |   |                |30|                    |
    # |     (3)12 12----------a*----40 40------- a*_x-----31 31(2)     |
    # |                   10 13 \14            29 32                   |
    # 8(1)                10 13                29 32                  26(3)
    # 8(0)                (0,1)                (0,1)                  26(0)
    # C4-----(1)7 7(2)-----T3----(3)41 41(2)----T3_x----(3)27 27(1)----C3_x
    who="rdm2x2_NNN_1n1_oe"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_ny= state.site( (coord[0],coord[1]-1) )
    a_xny= state.site( (coord[0]+1,coord[1]-1) )
    C1_ny, C2_xny, C3_x, C4= env.C[(state.vertexToSite( (coord[0],coord[1]-1) ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]-1) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,1))],\
        env.C[(state.vertexToSite( coord ), (-1,1))]
    T1_ny, T4_ny, T1_xny, T2_xny, T2_x, T3_x, T3, T4= \
        env.T[(state.vertexToSite( (coord[0],coord[1]-1) ),(0,-1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]-1) ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]-1) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]-1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,1))],\
        env.T[(state.vertexToSite(coord), (0,1))],\
        env.T[(state.vertexToSite(coord), (-1,0))]
    t= C1_ny, C2_xny, C3_x, C4, T1_ny, T4_ny, T1_xny, T2_xny, T2_x, T3_x, T3, T4, a, a_x, a_ny, a_xny
    if force_cpu:
       t=(x.cpu() for x in t)

    T1_ny= T1_ny.view(T1_ny.size(0),a_ny.size(1),a_ny.size(1),T1_ny.size(2))
    T1_xny= T1_xny.view(T1_xny.size(0),a_xny.size(1),a_xny.size(1),T1_xny.size(2))
    T2_xny= T2_xny.view(T2_xny.size(0),a_xny.size(4),a_xny.size(4),T2_xny.size(2))
    T2_x= T2_x.view(T2_x.size(0),a_x.size(4),a_x.size(4),T2_x.size(2))
    T3_x= T3_x.view(a_x.size(3),a_x.size(3),T3_x.size(1),T3_x.size(2))
    T3= T3.view(a.size(3),a.size(3),T3.size(1),T3.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_ny= T4_ny.view(T4_ny.size(0),T4_ny.size(1),a_ny.size(2),a_ny.size(2))

    contract_tn= C1_ny,[0,1],T1_ny,[1,2,5,36],T4_ny,[0,15,3,6],a_ny,[4,2,3,16,37],a_ny.conj(),[4,5,6,17,38],\
        T4,[15,8,9,12],C4,[8,7],T3,[10,13,7,41],a,[11,16,9,10,39],a.conj(),[14,17,12,13,40],\
        T1_xny,[36,20,23,18],C2_xny,[18,19],T2_xny,[19,21,24,33],a_xny,[22,20,37,34,21],a_xny.conj(),[25,23,38,35,24],\
        T2_x,[33,28,31,26],C3_x,[26,27],T3_x,[29,32,41,27],a_x,[30,34,39,29,28],a_x.conj(),[30,35,40,32,31],[11,22,14,25]
    names= tuple(x.strip() for x in ("C1_ny, T1_ny, T4_ny, a_ny, a_ny*, T4, C4, T3, a, a*, T1_xny, C2_xny, T2_xny, a_xny, a_xny*,"\
        +"T2_x, C3_x, T3_x, a_x, a_x*").split(','))
    #
    # This (typical) strategy is optimal, when X >> D^2 >> phys_dim
    # 
    # path= ((6, 7), (5, 18), (6, 17), (5, 16), (0, 2), (0, 14), (1, 13), (0, 12), (1, 2), (0, 10),\
    #     (1, 9), (0, 8), (1, 2), (0, 6), (1, 5), (0, 4), (2, 3), (1, 2), (0, 1))
    if type(unroll)==bool and unroll:
        unroll= [11,14,22,25]
    path, path_info= get_contraction_path(*contract_tn,names=names,path=None,\
        unroll=unroll if unroll else [],who=who,optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,optimize=path,unroll=unroll if unroll else [],\
        checkpoint_unrolled=checkpoint_unrolled,
        checkpoint_on_device=checkpoint_on_device,
        backend='torch',who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R


def rdm2x2(coord, state, env, open_sites=[0,1,2,3],\
        unroll=[], checkpoint_unrolled=False, checkpoint_on_device=False, 
        sym_pos_def=False, force_cpu=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param open_sites: which sites to keep open
    :type open_sites: list(int)
    :param sym_pos_def: enforce hermiticity (always) and positive definiteness if ``True``
    :type sym_pos_def: bool
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :param force_cpu: compute on CPU
    :type force_cpu: bool
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

    If ``open_sites`` is provided with smaller number of sites to be left open, the returned
    density matrix indices still follow the order above.
    """
    if oe:
        return rdm2x2_oe(coord, state, env, open_sites=open_sites, unroll=unroll, 
            checkpoint_unrolled=checkpoint_unrolled, checkpoint_on_device=checkpoint_on_device,
            sym_pos_def=sym_pos_def, force_cpu=force_cpu, verbosity=verbosity)
    else:
        return rdm2x2_legacy(coord, state, env, sym_pos_def=sym_pos_def, force_cpu=force_cpu,\
            verbosity=verbosity)

def rdm2x2_legacy(coord, state, env, sym_pos_def=False, verbosity=0):
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
    who = "rdm2x2"
    # ----- building C2x2_LU ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (-1, -1))]
    T1 = env.T[(state.vertexToSite(coord), (0, -1))]
    T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
    dimsA = state.site(coord).size()
    a = contiguous(einsum('mefgh,nabcd->eafbgchdmn', state.site(coord), conj(state.site(coord))))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

    # C--10--T1--2
    # 0   1
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
    # 2->1    2\45
    C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    # C2x2--1
    # |\23
    # 0
    C2x2_LU = contiguous(permute(C2x2_LU, (1, 2, 0, 3, 4, 5)))
    C2x2_LU = view(C2x2_LU, (T2.size(1) * a.size(2), T1.size(2) * a.size(3), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C = env.C[(shift_coord, (1, -1))]
    T1 = env.T[(shift_coord, (1, 0))]
    T2 = env.T[(shift_coord, (0, -1))]
    dimsA = state.site(shift_coord).size()
    a = contiguous(einsum('mefgh,nabcd->eafbgchdmn', state.site(shift_coord), conj(state.site(shift_coord))))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

    # 0--C
    #    1
    #    0
    # 1--T1
    #     2
    C2x2_RU = contract(C, T1, ([1], [0]))

    # 2<-0--T2--2 0--C
    #    3<-1        |
    #          0<-1--T1
    #             1<-2
    C2x2_RU = contract(C2x2_RU, T2, ([0], [2]))

    # 1<-2--T2------C
    #       3       |
    #    45\0       |
    # 2<-1--a--3 0--T1
    #    3<-2    0<-1
    C2x2_RU = contract(C2x2_RU, a, ([0, 3], [3, 0]))

    # permute 012334->120345
    # reshape (12)(03)45->0123
    # 0--C2x2
    # 23/|
    #    1
    C2x2_RU = contiguous(permute(C2x2_RU, (1, 2, 0, 3, 4, 5)))
    C2x2_RU = view(C2x2_RU, (T2.size(0) * a.size(1), T1.size(2) * a.size(2), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 RU " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (1,-1): " + str(
            C2x2_RU.size()))

    # ----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU           C2x2_LU------C2x2_RU
    # |\23->12   |\23->45   & permute |\12->23     |\45
    # 0          1->3                 0            3->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?
    upper_half = contract(C2x2_LU, C2x2_RU, ([1], [0]))
    upper_half = permute(upper_half, (0, 3, 1, 2, 4, 5))

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C = env.C[(shift_coord, (1, 1))]
    T1 = env.T[(shift_coord, (0, 1))]
    T2 = env.T[(shift_coord, (1, 0))]
    dimsA = state.site(shift_coord).size()
    a = contiguous(einsum('mefgh,nabcd->eafbgchdmn', state.site(shift_coord), conj(state.site(shift_coord))))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

    #   1<-0        0
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
    #       2\45    |
    #       0       |
    # 0<-1--T1------C
    C2x2_RD = contract(C2x2_RD, a, ([0, 3], [2, 3]))

    # permute 012345->120345
    # reshape (12)(03)45->0123
    C2x2_RD = contiguous(permute(C2x2_RD, (1, 2, 0, 3, 4, 5)))
    C2x2_RD = view(C2x2_RD, (T2.size(0) * a.size(0), T1.size(1) * a.size(1), dimsA[0], dimsA[0]))

    #    0
    #    |/23
    # 1--C2x2
    if verbosity > 0:
        print("C2X2 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (1,1): " + str(
            C2x2_RD.size()))

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C = env.C[(shift_coord, (-1, 1))]
    T1 = env.T[(shift_coord, (-1, 0))]
    T2 = env.T[(shift_coord, (0, 1))]
    dimsA = state.site(shift_coord).size()
    a = contiguous(einsum('mefgh,nabcd->eafbgchdmn', state.site(shift_coord), conj(state.site(shift_coord))))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2, dimsA[0], dimsA[0]))

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
    # |        2\45
    # |        2
    # C--------T2--3->1
    C2x2_LD = contract(C2x2_LD, a, ([1, 2], [1, 2]))

    # permute 012345->021345
    # reshape (02)(13)45->0123
    # 0
    # |/23
    # C2x2--1
    C2x2_LD = contiguous(permute(C2x2_LD, (0, 2, 1, 3, 4, 5)))
    C2x2_LD = view(C2x2_LD, (T1.size(0) * a.size(0), T2.size(2) * a.size(3), dimsA[0], dimsA[0]))
    if verbosity > 0:
        print("C2X2 LD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shift_coord) + " (-1,1): " + str(
            C2x2_LD.size()))

    # ----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->3                 0            3->1
    # |/23->12      |/23->45   & permute |/12->23     |/45
    # C2x2_LD--1 1--C2x2_RD              C2x2_LD------C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))
    lower_half = permute(lower_half, (0, 3, 1, 2, 4, 5))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |\23->01     |\45->23
    # 0            1
    # 0            1
    # |/23->45     |/45->67
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 4, 6, 1, 3, 5, 7)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm

def rdm2x2_oe(coord, state, env, open_sites=[0,1,2,3], unroll=False, 
    checkpoint_unrolled=False, checkpoint_on_device=False,
    sym_pos_def=False, force_cpu=False, verbosity=0):
    # C1------(1)1 1(0)----T1----(3)36 36(0)----T1_x----(3)18 18(0)----C2_x
    # 0(0)               (1,2)                 (1,2)                   19(1)
    # 0(0)           100  2  5             102 20 23                   19(0)
    # |                 \ 2  5               \ |  |                    |  
    # T4-------(2)3 3-----a--|------37 37----a_x--6(1)----21 21(1)-----T2_x
    # |                   |  |                 |  |                    |
    # |        (3)6 6-------a*------38 38--------a*_x-----24 24(2)     |
    # 15(1)               16 17 \101          34 35 \103               33(3)
    # 15(0)          104  16 17           106 34 35                    33(0)
    # |                 \ |   |              \ |  |                    |
    # T4_y--(2)9 9--------a_y-------39 39-----a_xy--------28 28(1)-----T2_xy
    # |                   |   |                |30|                    |
    # |     (3)12 12---------a*_y---40 40------- a*_xy----31 31(2)     |
    # |                   10 13 \105           29 32 \107              |
    # 8(1)                10 13                29 32                  26(3)
    # 8(0)                (0,1)                (0,1)                  26(0)
    # C4_y---(1)7 7(2)-----T3_y--(3)41 41(2)----T3_xy---(3)27 27(1)----C3_xy
    ind_os= set(sorted(open_sites))
    assert len(ind_os)==len(open_sites),"contains repeated elements"
    assert ind_os <= {0,1,2,3},"allowed site labels are 0,1,2, and 3"
    I= sum([[100+2*x,100+2*x+1] if x in ind_os else [100+2*x]*2 for x in [0,1,2,3]],[])
    I_out= [100+2*x for x in ind_os]+[100+2*x+1 for x in ind_os]

    who=f"rdm2x2_oe_{ind_os}"
    a= state.site(coord)
    a_x= state.site( (coord[0]+1,coord[1]) )
    a_y= state.site( (coord[0],coord[1]+1) )
    a_xy= state.site( (coord[0]+1,coord[1]+1) )
    C1, C2_x, C3_xy, C4_y= env.C[(state.vertexToSite( coord ),(-1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,-1))],\
        env.C[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (1,1))],\
        env.C[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,1))]
    T1, T4, T1_x, T2_x, T2_xy, T3_xy, T3_y, T4_y= \
        env.T[(state.vertexToSite( coord ),(0,-1))],\
        env.T[(state.vertexToSite( coord ),(-1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (0,-1))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (1,0))],\
        env.T[(state.vertexToSite( (coord[0]+1,coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (0,1))],\
        env.T[(state.vertexToSite( (coord[0],coord[1]+1) ), (-1,0))]
    t= C1, C2_x, C3_xy, C4_y, T1, T4, T1_x, T2_x, T2_xy, T3_xy, T3_y, T4_y, a, a_x, a_y, a_xy
    if force_cpu:
       t=(x.cpu() for x in t)

    T1= T1.view(T1.size(0),a.size(1),a.size(1),T1.size(2))
    T1_x= T1_x.view(T1_x.size(0),a_x.size(1),a_x.size(1),T1_x.size(2))
    T2_xy= T2_xy.view(T2_xy.size(0),a_xy.size(4),a_xy.size(4),T2_xy.size(2))
    T2_x= T2_x.view(T2_x.size(0),a_x.size(4),a_x.size(4),T2_x.size(2))
    T3_xy= T3_xy.view(a_xy.size(3),a_xy.size(3),T3_xy.size(1),T3_xy.size(2))
    T3_y= T3_y.view(a_y.size(3),a_y.size(3),T3_y.size(1),T3_y.size(2))
    T4= T4.view(T4.size(0),T4.size(1),a.size(2),a.size(2))
    T4_y= T4_y.view(T4_y.size(0),T4_y.size(1),a_y.size(2),a_y.size(2))

    contract_tn= C1,[0,1],T1,[1,2,5,36],T4,[0,15,3,6],a,[I[0],2,3,16,37],a.conj(),[I[1],5,6,17,38],\
        T4_y,[15,8,9,12],C4_y,[8,7],T3_y,[10,13,7,41],a_y,[I[4],16,9,10,39],a_y.conj(),[I[5],17,12,13,40],\
        T1_x,[36,20,23,18],C2_x,[18,19],T2_x,[19,21,24,33],a_x,[I[2],20,37,34,21],a_x.conj(),[I[3],23,38,35,24],\
        T2_xy,[33,28,31,26],C3_xy,[26,27],T3_xy,[29,32,41,27],a_xy,[I[6],34,39,29,28],a_xy.conj(),[I[7],35,40,32,31],I_out
    names= tuple(x.strip() for x in ("C1, T1, T4, a, a*, T4_y, C4_y, T3_y, a_y, a_y*, T1_x, C2_x, T2_x, a_x, a_x*,"\
        +"T2_xy, C3_xy, T3_xy, a_xy, a_xy*").split(','))
    #
    # This (typical) strategy is optimal, when X >> D^2 >> phys_dim
    # 
    # path= ((6, 7), (5, 18), (6, 17), (5, 16), (0, 2), (0, 14), (1, 13), (0, 12), (1, 2), (0, 10),\
    #     (1, 9), (0, 8), (1, 2), (0, 6), (1, 5), (0, 4), (2, 3), (1, 2), (0, 1))
    #
    if type(unroll)==bool and unroll:
        unroll= I_out
    path, path_info= get_contraction_path(*contract_tn,unroll=unroll,\
        names=names,path=None,who=who,optimizer="default" if env.chi>1 else "auto")
    R= contract_with_unroll(*contract_tn,unroll=unroll,optimize=path,backend='torch',
        checkpoint_unrolled=checkpoint_unrolled,checkpoint_on_device=checkpoint_on_device,
        who=who,verbosity=verbosity)

    R = _sym_pos_def_rdm(R, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    if force_cpu:
        R= R.to(env.device)
    return R


def rdm2x3_trglringex(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 2x3 patch using strategy:

        1. compute left edge of the network
        2. add extra T-tensor and on-site tensor to the bottom of the left edge
        3. analogously for the right edge, attaching extra T-tensor
           and on-site tensor to the top of the right edge
        4. contract left and right half to obtain final reduced density matrix

    ::

        C--T------------------T-----T-------------------C = C2x2_LU(coord+(0,-1))--T-----C2x2(coord+(2,-1))
        |  |                  |     |                   |   |___________________|--A^+A--|_______________|
        T--A^+A(coord+(0,-1))-A^+A--A^+A(coord+(2,-1))--T   |                   |--A^+A--|               |
        |  |                  |     |                   |   C2x2_LD(coord)---------T-----C2x2(coord+(2,0))
        T--A^+A(coord)--------A^+A--A^+A(coord+(2,0))---T
        |  |                  |     |                   |
        C--T------------------T-----T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s3 s2
        s0 s1 x 

    """
    who="rdm2x3"
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build left part C2x2_LU--C2X2_LD ------------------------------------
    # C2x2_LU--1->0
    # |
    # 0
    # 0/23
    # C2x2_LD--1
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_LD, ([0],[0]))

    # attach extra T-tensor and open double-layer tensor
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_10= env.T[(shift_coord,(0,1))]
    T_10= T_10.view([state.site((shift_coord)).size(3)]*2+[T_10.size(1),T_10.size(2)])
    # C2x2_LU--0
    # |
    # |     ->3,4    ->1,2
    # |/23->4,5   /->2,3  0,1->5,6
    # C2x2_LD-----1->1 2--T[coord+(1,0),(0,1)]--3->7
    C2X2_LU= C2X2_LU.view([C2X2_LU.size(0)]+[T_10.size(2)]+[state.site(coord).size(4)]*2\
        +[state.site(coord).size(0)]*2)
    C2X2_LU= torch.tensordot(C2X2_LU, T_10, ([1],[2]))

    #               ->7
    # C2x2LU---0    1 0->6
    # |_____ /-2->1 |/
    # |3,4  |--1 2--a--4->8
    # |->2,3|       3
    # |     |       5,6->4
    # C2x2_LD-------T---7->5
    C2X2_LU= torch.tensordot(C2X2_LU, state.site(shift_coord), ([1,5],[2,3]))

    #             ->5   ->8      
    #               7 ->4 |   ->7
    # C2x2LU---0    | 6   |/--0       left|--0->0,1,2
    # |_____ /-1 2--|/----a*---4->9   |   |     
    # |2,3  |-------a--8--|->6        |   |______5 8->7 10
    # |->1,2|       | 4 3/            | ->3,4   ->6,9  --9->11
    # |     |       |/                |1,2     4,7     --6->8
    # C2x2_LD-------T---5->3          |________________--3->5
    C2X2_LU= torch.tensordot(C2X2_LU, state.site(shift_coord).conj(), ([1,4],[2,3]))
    vec = (1, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= C2X2_LU.view( [env.T[shift_coord, (0,-1)].size(0)]+\
        [state.site(shift_coord).size(2)]*2 + list(C2X2_LU.size()[1:]) )

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (2, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

     # ----- building C2x2_RD ----------------------------------------------------
    vec = (2, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build right part C2X2_RU--C2X2_RD -----------------------------------
    #            0--C2x2_RU--1,2 
    #               1
    #               0
    #         3<-1--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,C2X2_RD,([1],[0]))

    # attach extra T-tensor and open double-layer tensor
    vec = (1, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_11= env.T[(shift_coord,(0,-1))]
    T_11= T_11.view([T_11.size(0)]+[state.site((shift_coord)).size(1)]*2+[T_11.size(2)])

    #     0--T(coord+(1,1),(0,-1))--3  0<-0--C2x2_RU--1,2->3,4->5,6 
    #        1,2                 3,4<-1,2/   | 
    #                                        |
    #                               7<-5<-3--C2x2_RD
    C2X2_RU= C2X2_RU.view([T_11.size(3)]+[state.site(shift_coord).size(4)]*2\
        +[C2X2_RU.size(1)]*2+[C2X2_RU.size(3)])
    C2X2_RU= torch.tensordot(T_11,C2X2_RU,([3],[0]))

    #     0--T-------C2x2_RU--5,6->3,4 
    #   1<-2,1       |
    #        1       |
    #  7<-2--a--4 3--|
    #   6<-0/3    4--|
    #     8<-   2<-  |
    #          5<-7--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,state.site(shift_coord),([1,3],[1,4]))

    #     0--T----------C2x2_RU--3,4->1,2         0--|right |--1,2
    #        | 1 ->7    |                     10<-8--|      |--4,7->6,9 
    #        | 1/0      |                      7<-5--|   ___
    #  8<-2--|-a*--4 2--|                          8<-6 9   |
    #  5<-7--a----------|                           11<-    |
    #   4<-6/8 |        |                         3,4,5<-3--|
    #      6<- 3        |
    #       9<-   3<-5--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RU,state.site(shift_coord).conj(),([1,2],[1,4]))
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= C2X2_RU.view(list(C2X2_RU.size()[:3])+[env.T[shift_coord,(0,1)].size(2)]\
        +[state.site(shift_coord).size(4)]*2+list(C2X2_RU.size()[4:]))

    #
    # x   6,7 4,5
    # 0,1 2,3 x
    rdm= torch.tensordot(C2X2_LU,C2X2_RU,([0,1,2, 5,8,11, 7,10],[0,7,10, 3,4,5, 8,11]))

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 4, 6, 1, 3, 5, 7)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm

def rdm2x3_trglringex_compressed(coord,state,env,compressed_chi=None,sym_pos_def=False,\
    ctm_args=cfg.ctm_args,global_args=cfg.global_args,verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 2x3 patch using strategy:

        1. compute left edge of the network with compression on the top-left corner
        2. add extra T-tensor and on-site tensor to the bottom of the left edge
        3. analogously for the right edge, attaching extra T-tensor
           and on-site tensor to the top of the right edge and compression 
           on the bottom right corner
        4. contract left and right half to obtain final reduced density matrix

    The isometries performing the compresion are obtained as CTMRG projectors
    using full (:attr:`CTMARGS.projector_method` = ``"4X4"``) method.

    ::

        C--T-------------------|\    /|--T---------------T-------------------C
        |  |                   | >--< |  |               |                   |
        T--A^+A(coord+(0,-1))--|/    \|--A^+A------------A^+A(coord+(2,-1))--T
        |  |                             |               |                   |
        T--A^+A(coord)-------------------A^+A--|\    /|--A^+A(coord+(2,0))---T
        |  |                             |     | >--< |  |                   |
        C--T-----------------------------T-----|/    \|--T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s3 s2
        s0 s1 x

    """ 
    who="rdm2x3_compressed"
    if not compressed_chi: compressed_chi= env.chi
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build left part C2x2_LU--C2X2_LD ------------------------------------
    # C2x2_LU--1->0
    # |
    # 0
    # 0/23
    # C2x2_LD--1
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_LD, ([0],[0]))

    # construct projector for UP move between coord+(0,-1) and coord+(1,-1)
    # see :meth:`ctm.generic.ctm_components.halves_of_4x4_CTM_MOVE_UP_c`
    #
    #        _0 0_          --|\__/|--
    #       |     |         --|/  \|--
    # C2X2_LU     half1 =>    P    Pt
    #       |_1 1_|
    vec = (1, -1)
    shift_coord_1n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    vec = (1, 0)
    shift_coord_10 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    half1= torch.tensordot(c2x2_RU(shift_coord_1n1,state,env,mode='sl',verbosity=verbosity),\
        c2x2_RD(shift_coord_10,state,env,mode='sl',verbosity=verbosity),([1],[0]))

    P_up, Pt_up= ctm_get_projectors_from_matrices(half1, torch.einsum('ijss->ij',C2X2_LU),\
        compressed_chi, ctm_args, global_args)

    # compress C2X2_LU
    #  
    #      |C2X2_LU--0 0--P--1->0  
    #      |      |
    # 2,3--|______|--1
    C2X2_LU= torch.tensordot(P_up,C2X2_LU,([0],[0]))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (2, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

     # ----- building C2x2_RD ----------------------------------------------------
    vec = (2, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build right part C2X2_RU--C2X2_RD -----------------------------------
    #         1<-0--C2x2_RU--1,2->2,3
    #               1
    #               0
    #         0<-1--C2x2_RD
    C2X2_RU= torch.tensordot(C2X2_RD,C2X2_RU,([0],[1]))

    # construct projector for DOWN move between coord+(1,0) and coord+(2,0)
    # see :meth:`ctm.generic.ctm_components.halves_of_4x4_CTM_MOVE_DOWN_c`
    #
    #        _1 1_            --|\__/|--
    #       |     |           --|/  \|--
    #   half1     C2X2_RU =>    Pt   P
    #       |_0 0_|
    half1= torch.tensordot(c2x2_LD(shift_coord_10,state,env,mode='sl',verbosity=verbosity),\
        c2x2_LU(shift_coord_1n1,state,env,mode='sl',verbosity=verbosity),([0],[0]))
    
    P_down, Pt_down= ctm_get_projectors_from_matrices(half1, torch.einsum('ijss->ij',C2X2_RU),\
        compressed_chi, ctm_args, global_args)

    # compress C2X2_RU
    #  
    #            1--|C2X2_RU--2,3 
    #               |      |
    # 0<-1--P--0 0--|______|
    C2X2_RU= torch.tensordot(P_down,C2X2_RU,([0],[0]))

    # contract bottom T-tensor of central column with projector and on-site tensors
    # mem \chi^2 D^4 p^2
    #
    #     9 7  10
    #      \|  |
    #    8--a------2   2--|\
    #   11--|--a*--3   3--| Pt_down--0 
    #       |  |\         |
    #       5  6 12       |
    #                     |
    #       5,6(0,1)      |
    # 4(2)--T---1(3) 1----|/
    #
    T_10= env.T[shift_coord_10,(0,1)]
    T_10= T_10.view([state.site(shift_coord_10).size(3)]*2+list(T_10.size())[1:]) 
    Pt_down= Pt_down.view([T_10.size(3)]+[state.site(shift_coord_10).size(4)]*2+[Pt_down.size(1)])
    T_10aa_open= torch.einsum(Pt_down,[1,2,3,0],T_10,[5,6,4,1],\
        state.site(shift_coord_10),[9,7,8,5,2],\
        state.site(shift_coord_10).conj(),[12,10,11,6,3],[4,8,11, 0, 7,10, 9,12])
    

    # contract T_10aa_open with compressed left edge
    # mem \chi^2 D^2 p^4
    #
    #           |C2X2_LU--P--0                    7,8(4,5)
    #           |      |                          |
    #           |      |                         |aa|--9,10(6,7)
    # 4,5<-2,3--|______|--1->1,2,3 1,2,3(0,1,2)--T_10--6(3)
    C2X2_LU= C2X2_LU.view([C2X2_LU.size(0)]+list(T_10aa_open.size()[:3])\
        + [C2X2_LU.size(2), C2X2_LU.size(3)])
    C2X2_LU= torch.einsum(C2X2_LU,[0, 1,2,3, 4,5], T_10aa_open, [1,2,3, 6, 7,8, 9,10],\
        [0,6,7,8, 4,5,9,10])

    # contract top T-tensor of central column with projector and on-site tensors
    # mem \chi^2 D^4 p^2
    #    
    #         /|--1   1--T------4
    #          |         7, 10         
    #          |       9 7  10         
    #          |        \|  |         
    # 0--Pt_up |--2   2--a------8
    #         \|--3   3--|--a*--11 
    #                    |  |\         
    #                    5  6 12       
    #
    T_1n1= env.T[shift_coord_1n1,(0,-1)]
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(1)]*2+[T_1n1.size(2)])
    Pt_up= Pt_up.view([T_1n1.size(3)]+[state.site(shift_coord_1n1).size(4)]*2+[Pt_up.size(1)])
    T_1n1aa_open= torch.einsum(Pt_up,[1,2,3,0],T_1n1,[1,7,10,4],\
        state.site(shift_coord_1n1),[9,7,2,5,8],\
        state.site(shift_coord_1n1).conj(),[12,10,3,6,11],[4,8,11, 0, 5,6, 9,12])

    # contract T_1n1aa_open with compressed right edge
    # mem \chi^2 D^2 p^4
    #
    #      6(3)--T_1n1--1,2,3(0,1,2)  1,2,3<-1--|C2X2_RU--2,3->4,5 
    # 9,10(6,7)--|aa |                          |      |
    #             7,8(4,5)                      |      |
    #                                  0<--P----|______|
    C2X2_RU= C2X2_RU.view([C2X2_RU.size(0)]+list(T_1n1.size()[:3])\
        +[C2X2_RU.size(2),C2X2_RU.size(3)])
    C2X2_RU= torch.einsum(C2X2_RU,[0, 1,2,3, 4,5], T_1n1aa_open, [1,2,3, 6, 7,8, 9,10],\
        [0,6,7,8,4,5,9,10])

    # contract two parts
    #                               
    #   C2X2_LU-------0   0(1)--|10,11(6,7) 8,9(4,5)|
    #   | x      |--2,3    2,3--|                x  | 
    #   | 4,5 6,7|----1   1(0)----------------C2X2_RU
    #
    rdm= torch.einsum(C2X2_LU,[0,1,2,3, 4,5,6,7],C2X2_RU,[1,0,2,3, 8,9,10,11],\
        [4,5,6,7,8,9,10,11])

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 4, 6, 1, 3, 5, 7)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm


def rdm3x2_trglringex(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 2x3 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 3x2 patch using strategy:

        1. compute top edge of the network
        2. add extra T-tensor and on-site tensor to the right of the top edge
        3. analogously for the bottom edge, attaching extra T-tensor
           and on-site tensor to the left of the bottom edge
        4. contract top and bottom half to obtain final reduced density matrix

    ::

        C--T-------------------T-------------------C
        |  |                   |                   |
        T--A^+A(coord+(0,-2))--A^+A(coord+(1,-2))--T
        |  |                   |                   |
        T--A^+A(coord+(0,-1))--A^+A(coord+(1,-1))--T
        |  |                   |                   |
        T--A^+A(coord)---------A^+A(coord+(1,0))---T
        |  |                   |                   |
        C--T-------------------T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::

        x  s2
        s3 s1  
        s0 x 

    """
    who="rdm3x2"
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build top part C2x2_LU--C2X2_RU ------------------------------------
    # C2x2_LU--1 0--C2x2_RU--2,3->4,5
    # |                  |
    # 0                  1->1,2,3
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_RU, ([1],[0]))

    vec = (1, -1)
    shift_coord_1n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_1n1= env.T[shift_coord_1n1,(1,0)]
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(4)]*2+[T_1n1.size(2)]) 

    C2X2_LU= C2X2_LU.view([C2X2_LU.size(0)]+[T_1n1.size(0)]\
        +[state.site(shift_coord_1n1).size(1)]*2+[C2X2_LU.size(2), C2X2_LU.size(3)])

    # contract right T-tensor of central row and on-site tensors
    # mem \chi^2 D^4 p^2
    #
    # C2x2_LU--------------13(4),14(5)  =>   |C2x2_LU   3,4 
    # 0     7(2) 10(3)  1(1)                 |       |  1,2
    #                                        |    2--|____|
    #     9 7  10       1                    |            |
    #      \|  |        |                    0            1
    #    8--a------2 2  |
    #   11--|--a*--3 3--T   
    #       |  |\       |
    #       5  6 12     4
    #
    C2X2_LU= torch.einsum(C2X2_LU,[0,1,7,10,13,14],T_1n1,[1,2,3,4],\
        state.site(shift_coord_1n1),[9,7,8,5,2],\
        state.site(shift_coord_1n1).conj(),[12,10,11,6,3],\
        [0, 4,5,6, 8,11, 9,12, 13,14]).contiguous()
    C2X2_LU= C2X2_LU.view([C2X2_LU.size(0)]+[prod(C2X2_LU.size()[1:4])]\
        +[prod(C2X2_LU.size()[4:6])]+list(C2X2_LU.size()[6:]))

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build bottom part C2X2_LD--C2X2_RD -----------------------------------
    #        
    #            0->1->1,2,3   0
    #  4,5<-2,3--C2x2_LD--1 1--C2x2_RD
    C2X2_RD= torch.tensordot(C2X2_RD,C2X2_LD,([1],[1]))

    vec = (0, -1)
    shift_coord_0n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    T_0n1= env.T[shift_coord_0n1,(-1,0)]
    T_0n1= T_0n1.view([T_0n1.size(0)]+[T_0n1.size(1)]+[state.site(shift_coord_0n1).size(2)]*2)

    C2X2_RD= C2X2_RD.view([C2X2_RD.size(0)]+[T_0n1.size(1)]\
        +[state.site(shift_coord_0n1).size(3)]*2+[C2X2_RD.size(2),C2X2_RD.size(3)])

    # contract left T-tensor of central row and on-site tensors
    # mem \chi^2 D^4 p^2
    #             
    #               1      9 7  10                <=>  1            0
    #               |       \|  |                      |______      |
    #               T--2  2--a------8                  |      |--2  | 
    #               |  3  3--|--a*--11                 |3,4   |     |
    #               |        |  |\                     |1,2___C2x2_RD
    #               |        |  | 12 
    #               4        5  6  
    #               4        5(2) 6(3)        0
    #  13(4),14(5)--C2x2_LD--------------C2x2_RD              
    #
    C2X2_RD= torch.einsum(C2X2_RD,[0,4,5,6,13,14],T_0n1,[1,4,2,3],\
        state.site(shift_coord_0n1),[9,7,2,5,8],\
        state.site(shift_coord_0n1).conj(),[12,10,3,6,11],\
        [0, 1,7,10, 8,11, 13,14, 9,12]).contiguous()
    C2X2_RD= C2X2_RD.view([C2X2_RD.size(0)]+[prod(C2X2_RD.size()[1:4])]\
        +[prod(C2X2_RD.size()[4:6])]+list(C2X2_RD.size()[6:]))

    # contract two parts
    #          __________  
    #   C2X2_LU  7,8(3,4)|     <=>  x  s2
    #   |_x______5,6(1,2)|          s3 s1
    #   |           |____|          s0 x 
    #   0           2    1
    #   0(1)        2    1(0)
    #   |9,10_______|____|
    #   |3,4        x    |
    #   |C2x2_RD_________|
    rdm= torch.einsum(C2X2_LU,[0,1,2, 5,6,7,8],C2X2_RD,[1,0,2, 3,4,9,10],\
        [3,4,5,6,7,8,9,10]).contiguous()

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 4, 6, 1, 3, 5, 7)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm

def rdm3x2_trglringex_compressed(coord,state,env,compressed_chi=None,sym_pos_def=False,\
    ctm_args=cfg.ctm_args,global_args=cfg.global_args,verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies lower left site of 3x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param ctm_args: CTM algorithm configuration
    :param global_args: global configuration
    :type ctm_args: CTMARGS
    :type global_args: GLOBALARGS
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 4-site reduced density matrix with indices 
             :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho` of four-site subsystem, 
    a parallelogram, specified by the vertex ``coord`` of its lower-left 
    and upper-right corner within 3x2 patch using strategy:

        1. compute top edge of the network with compression on the top-left corner
        2. add extra T-tensor and on-site tensor to the right of the top edge
        3. analogously for the botton edge with attaching extra T-tensor
           and on-site tensor to the left of the bottom edge and compression 
           on the bottom right corner
        4. contract top and left half to obtain final reduced density matrix

    The isometries performing the compresion are obtained as CTMRG projectors
    using full (:attr:`CTMARGS.projector_method` = ``"4X4"``) method.

    ::

        C--T-------------------T-------------------C
        |  |                   |                   |
        T--A^+A(coord+(0,-2))--A^+A(coord+(1,-2))--T
        |  |                   |                   |
         \/                    |                   |
         |                     |                   |
         /\                    |                   |
        |  |                   |                   |
        T--A^+A(coord+(0,-1))--A^+A(coord+(1,-1))--T
        |  |                   |                   |
        |  |                    \_________________/
        |  |                     ________|________ 
        |  |                    /                 \
        |  |                   |                   |
        T--A^+A(coord)---------A^+A(coord+(1,0))---T
        |  |                   |                   |
        C--T-------------------T-------------------C

    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`)
    at vertices ``coord`` and ``coord+(1,1)`` are left uncontracted and given in the same order::
    
        x  s2
        s3 s1
        s0 x

    """
    who="rdm3x2_compressed"
    if not compressed_chi: compressed_chi= env.chi
    # ----- building C2x2_LU ----------------------------------------------------
    vec = (0, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= c2x2_LU(shift_coord,state,env,mode='sl',verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, -2)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= c2x2_RU(shift_coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- build top part C2x2_LU--C2X2_RU ------------------------------------
    # C2x2_LU--1 0--C2x2_RU--2,3
    # |                  |
    # 0                  1
    C2X2_LU= torch.tensordot(C2X2_LU, C2X2_RU, ([1],[0]))

    # construct projector for LEFT move between coord+(0,-2) and coord+(0,-1)
    # see :meth:`ctm.generic.ctm_components.halves_of_4x4_CTM_MOVE_LEFT_c`
    #
    #        _C2X2_LU_          
    #       0         1        \__/ Pt
    #       0         1  =>    /  \ P
    #       |_half2___|
    vec = (0, -1)
    shift_coord_0n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    vec = (1, -1)
    shift_coord_1n1 = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    half2= torch.tensordot(c2x2_LD(shift_coord_0n1,state,env,mode='sl',verbosity=verbosity),\
        c2x2_RD(shift_coord_1n1,state,env,mode='sl',verbosity=verbosity),([1],[1]))

    P_left, Pt_left= ctm_get_projectors_from_matrices(torch.einsum('ijss->ij',C2X2_LU),half2,\
        compressed_chi, ctm_args, global_args)

    # compress C2X2_LU
    #  
    #      |C2X2_LU--2,3  
    #      0      |
    #      0      1
    #      Pt
    #      1->0
    C2X2_LU= torch.tensordot(Pt_left,C2X2_LU,([0],[0]))

    # ----- building C2x2_LD ----------------------------------------------------
    C2X2_LD= c2x2_LD(coord,state,env,mode='sl-open',verbosity=verbosity)

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= c2x2_RD(shift_coord,state,env,mode='sl',verbosity=verbosity)
    
    # ----- build bottom part C2X2_LD--C2X2_RD -----------------------------------
    #        
    #       0->1          0
    #  2,3--C2x2_LD--1 1--C2x2_RD
    C2X2_RD= torch.tensordot(C2X2_RD,C2X2_LD,([1],[1]))

    # construct projector for RIGHT move between coord+(1,-1) and coord+(1,0)
    # see :meth:`ctm.generic.ctm_components.halves_of_4x4_CTM_MOVE_RIGHT_c`
    #
    #        _half2___     
    #       1         0      \___/ P
    #       1         0  =>  /   \ Pt
    #       |_C2x2_RD_|
    half2= torch.tensordot(c2x2_RU(shift_coord_1n1,state,env,mode='sl',verbosity=verbosity),\
        c2x2_LU(shift_coord_0n1,state,env,mode='sl',verbosity=verbosity),([0],[1]))
    
    P_right, Pt_right= ctm_get_projectors_from_matrices(torch.einsum('ijss->ij',C2X2_RD),half2,\
        compressed_chi, ctm_args, global_args)

    # compress C2X2_RD
    #  
    #                 1->0
    #                 Pt
    #                 0
    #       1         0
    #  2,3--|_C2x2_RD_|
    C2X2_RD= torch.tensordot(Pt_right,C2X2_RD,([0],[0]))

    # contract right T-tensor of central row with projector and on-site tensors
    # mem \chi^2 D^4 p^2
    #
    #     9 7  10       4
    #      \|  |        |
    #    8--a------2 2  |
    #   11--|--a*--3 3--T   
    #       |  |\       |
    #       5  6 12     1
    #       5__6________1     
    #       \   P_right/ 
    #           0

    T_1n1= env.T[shift_coord_1n1,(1,0)]
    T_1n1= T_1n1.view([T_1n1.size(0)]+[state.site(shift_coord_1n1).size(4)]*2+[T_1n1.size(2)]) 
    P_right= P_right.view([T_1n1.size(3)]+[state.site(shift_coord_1n1).size(3)]*2+[P_right.size(1)])
    T_1n1aa_open= torch.einsum(P_right,[1,5,6,0],T_1n1,[4,2,3,1],\
        state.site(shift_coord_1n1),[9,7,8,5,2],\
        state.site(shift_coord_1n1).conj(),[12,10,11,6,3],[4,7,10, 0, 8,11, 9,12])

    # contract T_1n1aa_open with compressed top edge
    # mem \chi^2 D^2 p^4
    #
    #      |C2X2_LU--2,3->4,5  
    #      |      |
    #      |      1->1,2,3
    #      Pt        1,2,3(0,1,2)
    #      0         | 
    #   7,8(5,6)--|aaT_1n1|--9,10(7,8)
    #                |
    #                6(4)
    C2X2_LU= C2X2_LU.view([C2X2_LU.size(0)]+list(T_1n1aa_open.size()[:3])\
        + [C2X2_LU.size(2), C2X2_LU.size(3)])
    C2X2_LU= torch.einsum(C2X2_LU,[0, 1,2,3, 4,5], T_1n1aa_open, [1,2,3, 6, 7,8, 9,10],\
        [0,6,7,8, 4,5,9,10])

    # contract left T-tensor of central column with projector and on-site tensors
    # mem \chi^2 D^4 p^2
    #    
    #                  0
    #            /P_left_____\
    #            1       7  10             
    #            1     9 7  10         
    #            |       \|  |         
    #            T--2  2--a------8
    #            |  3  3--|--a*--11 
    #            |        |  |\         
    #            4        5  6 12       
    #
    T_0n1= env.T[shift_coord_0n1,(-1,0)]
    T_0n1= T_0n1.view([T_0n1.size(0)]+[T_0n1.size(1)]+[state.site(shift_coord_0n1).size(2)]*2)
    P_left= P_left.view([T_0n1.size(0)]+[state.site(shift_coord_0n1).size(1)]*2+[P_left.size(1)])
    T_0n1aa_open= torch.einsum(P_left,[1,7,10,0],T_0n1,[1,4,2,3],\
        state.site(shift_coord_0n1),[9,7,2,5,8],\
        state.site(shift_coord_0n1).conj(),[12,10,3,6,11],[4,5,6, 0, 8,11, 9,12])

    # contract T_0n1aa_open with compressed bottom edge
    # mem \chi^2 D^2 p^4
    #
    #            6(4)
    #            |                 
    # 9,10(7,8)--T_0n1aa|--7,8(5,6)          
    #            |          
    #            1,2,3(0,1,2)  
    #          ->1,2,3            0
    #            1                Pt
    #  4,5<-2,3--|_C2x2_RD________|
    C2X2_RD= C2X2_RD.view([C2X2_RD.size(0)]+list(T_0n1aa_open.size()[:3])\
        +[C2X2_RD.size(2),C2X2_RD.size(3)])
    C2X2_RD= torch.einsum(C2X2_RD,[0, 1,2,3, 4,5], T_0n1aa_open, [1,2,3, 6, 7,8, 9,10],\
        [0,6,7,8,4,5,9,10])

    # contract two parts
    #          __________  
    #   C2X2_LU          |     
    #   |_x_________  4,5|    
    #   |           |_6,7|
    #   0           2,3  1
    #   0(1)        2,3  1(0)
    #   |10,11(6,7) |____|
    #   |8,9(4,5)   x    |
    #   |C2x2_RD_________|
    rdm= torch.einsum(C2X2_LU,[0,1,2,3, 4,5,6,7],C2X2_RD,[1,0,2,3, 8,9,10,11],\
        [8,9,6,7,4,5,10,11])

    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3' where primed indices
    # represent "ket"
    # 01234567->02461357
    # symmetrize and normalize
    rdm = contiguous(permute(rdm, (0, 2, 4, 6, 1, 3, 5, 7)))
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    return rdm


# ----- auxiliary rdms -----
def norm_C4(coord,state,env):
    #
    # C[coord,(-1,-1)]-----------1 1(0)--C[coord+(-1,0),(1,-1)]
    # 0                                  2(1)
    # 0                                  2(0)
    # C[coord+(0,-1),(-1,1)]--(1)3 3(1)--C[coord+(-1,-1),(1,1)]
    #
    norm= torch.einsum(env.C[coord,(-1,-1)],[0,1],\
        env.C[state.vertexToSite((coord[0]-1,coord[1])),(1,-1)],[1,2],\
        env.C[state.vertexToSite((coord[0]-1,coord[1]-1)),(1,1)],[2,3],\
        env.C[state.vertexToSite((coord[0],coord[1]-1)),(-1,1)],[0,3])
    return norm

def norm_3x3(coord,state,env,verbosity=0):
    E= corrf.get_edge(coord, (-1,0), state, env, verbosity=verbosity)
    E= corrf.apply_TM_1sO(coord, (0,1), state, env, E, op=None, verbosity=verbosity)
    E= corrf.apply_edge(coord, (0,1), state, env, E, verbosity=verbosity)
    return E

def _CTCT_LD(coord,state,env):
    C1, C2, C3, C4, T1, T2, T3, T4= env.get_site_env_t(coord,state)
    #  C1--1->0
    #  0
    #  0
    #  T4--2
    #  1
    CTC_LD = torch.tensordot(C1,T4,([0],[0]))
    #  C1--0
    #  |
    #  T4--2->1
    #  1
    #  0
    #  C4--1->2
    CTC_LD = torch.tensordot(CTC_LD,C4,([1],[0]))
    # C1--0
    # |
    # T4--1
    # |        0->2
    # C4--2 1--T3--2->3
    # 
    CTC_LD = torch.tensordot(CTC_LD,T3,([2],[1]))
    return CTC_LD

def _Lhalf_1x2(coord,state,env):
    Lhalf= _CTCT_LD(coord,state,env)
    # C1--0 0--T1[coord,(0,-1)]--2->1->0
    # |        1->2->1
    # T4--1->2
    # |     2->3
    # C4----T3--3->4
    T1= env.T[coord,(0,-1)]
    Lhalf= torch.tensordot(T1.permute(0,2,1).contiguous(),Lhalf,([0],[0]))
    return Lhalf

def _CTCT_RU(coord,state,env):
    C1, C2, C3, C4, T1, T2, T3, T4= env.get_site_env_t(coord,state)
    #       0
    #    1--T2
    #       2
    #       0
    # 2<-1--C3
    CTC_RU= torch.tensordot(T2,C3,([2],[0]))
    #    0--C2
    #       1
    #       0
    #    1--T2
    #       |
    #    2--C3
    CTC_RU= torch.tensordot(C2,CTC_RU,([1],[0]))
    #  0--T1--2 0--C2
    #     1        |
    #        2<-1--T2
    #              |
    #        3<-2--C3
    CTC_RU= torch.tensordot(T1,CTC_RU,([2],[0]))
    return CTC_RU

def _Rhalf_1x2(coord,state,env):
    Rhalf= _CTCT_RU(coord,state,env)
    #                     0--T1----C2
    #                        1     |
    #                           2--T2
    #        0->3                  |
    #  4<-1--T3[coord,(0,1)]--2 3--C3
    T3= env.T[coord,(0,1)]
    Rhalf= torch.tensordot(Rhalf,T3,([3],[2]))
    return Rhalf

def aux_rdm1x1(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 1-site auxilliary reduced density matrix
    :rtype: torch.tensor
    
    Builds 1x1 reduced density matrix by 
    contracting the following tensor network::

        C1--T1--C2
        |   |   |
        T4--  --T2
        |   |   |
        C4--T3--C3

    """
    who= "aux_rdm1x1"
    CTC_LD= _CTCT_LD(coord,state,env)
    CTC_RU= _CTCT_RU(coord,state,env)

    #   C1--0  0--T1-------C2
    #   |         1->2     |
    #   T4--1->0     3<-2--T2 
    #   |      1<-2        |
    #   C4--------T3--3 3--C3
    #
    rdm= torch.tensordot(CTC_LD,CTC_RU,([0,3],[0,3]))
    rdm= rdm.permute(2,0,1,3).contiguous()

    # 4i) unfuse the D^2 indices and permute to bra,ket
    #
    #   C----T----C      C------T--------C
    #   |    0    |      |      0,1      |
    #   T--1   3--T  =>  T--2,3     6,7--T
    #   |    2    |      |      4,5      | 
    #   C----T----C      C------T--------C
    #        
    a= state.site(coord)
    rdm= rdm.view([a.size(1)]*2+[a.size(2)]*2+[a.size(3)]*2+[a.size(4)]*2)
    rdm= rdm.permute(0,2,4,6,1,3,5,7).contiguous()

    return rdm

def aux_rdm1x2(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: 1-site auxilliary reduced density matrix
    :rtype: torch.tensor
    
    Builds 1x1 reduced density matrix by 
    contracting the following tensor network::

        C1--T1[(0,0),(0,-1)]--T1[(1,0),(0,-1)]--C2
        |   |                 |                 |
        T4--                                  --T2
        |   |                 |                 |
        C4--T3[(0,0),(0,1)]---T3[(1,0),(0,1)]---C3

    """
    who= "aux_rdm1x1"
    Lhalf= _Lhalf_1x2(coord,state,env)
    Rhalf= _Rhalf_1x2(coord,state,env)

    #   C1----T1--0 0--T1----C2
    #   |     1->0     1->3  |
    #   T4--2->1       4<-2--T2
    #   |     3->2     3->5  |
    #   C4----T3--4 4--T3----C3
    #
    rdm= torch.tensordot(Lhalf,Rhalf,([0,4],[0,4]))
    # take anti-clockwise order
    rdm= rdm.permute(0,1,2,5,4,3).contiguous()

    # 4i) unfuse the D^2 indices and permute to bra,ket
    #
    #   C1----T1-------T1----C2
    #   |     0,1      10,11 |
    #   T4--2,3         8,9--T2
    #   |     4,5      6,7   |
    #   C4----T3-------T3----C3
    #        
    dims00= state.site(coord).size()
    dims10= state.site((coord[0]+1,coord[1])).size()
    rdm= rdm.view([dims00[1]]*2+[dims00[2]]*2+[dims00[3]]*2\
        +[dims10[3]]*2+[dims10[4]]*2+[dims10[1]]*2)
    rdm= rdm.permute(0,2,4,6,8,10, 1,3,5,7,9,11).contiguous()

    return rdm