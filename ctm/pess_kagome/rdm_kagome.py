import torch
from ctm.generic.env import ENV
from ctm.generic.rdm import _sym_pos_def_rdm, _cast_to_real
from tn_interface import contract, einsum
from tn_interface import contiguous, view, permute
from tn_interface import conj

# ----- auxiliary functions -----
def _shift_coord(state,coord,vec):
    return state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))

def _abc_to_012_site(sites_to_keep):
    char_to_int= {'A': 2, 'B': 1, 'C': 0}
    int_list=[]
    if len(sites_to_keep)>0:
        int_list= [ char_to_int[k] for k in sites_to_keep ]
        int_list.sort()
    return int_list

def _expand_perm(n_inds):
    c_sum=0
    expanded_bra, expanded_ket= [], []
    for n in n_inds:
        if n==0: continue
        expanded_bra.extend( list(range(2*c_sum,2*c_sum+n)) )
        expanded_ket.extend( list(range(2*c_sum+n,2*c_sum+2*n)) )
        c_sum+= n
    return expanded_bra+expanded_ket

def double_layer_a(state, coord, open_sites=[], force_cpu=False):
    r"""
    :param state: underlying wavefunction
    :param coord: vertex (x,y) for which the reduced density matrix is constructed
    :param open_sites: a list DoFs to leave open (uncontracted).  
    :param force_cpu: perform on CPU
    :type state: IPEPS_KAGOME
    :type coord: tuple(int,int)
    :type open_sites: list(int)
    :type force_cpu: bool
    :return: result of (partial) contraction of double-layer tensor 
    :rtype: torch.tensor

    Build double-layer tensor of Kagome iPEPS with open, partially, or fully contracted 
    physical space of 3 DoFs on down triangle::

           u
           |                            /
        l--\                         --A*--
            \                         /|\ \            /
            s0--s2--r          ->      | | \  ->   --a*a--
             | /                    s' 0 1 2         / \
             |/   <- DOWN_T            ? ? ?           |s><s'|  
            s1                       s 0 1 2
             |                         | | /
             d                         |///
                                     --A--
                                      /

    Default results in contraction over all 3 DoFs. Physical indices are aggregated into
    a single index with structure :math:`|ket \rangle\langle bra| = s_0,...,s_2;s'_0,...,s'_2`.

    The available choices for ``open_sites`` are: [], [0], [1], [2], [0,1], [0,2], [1,2], and [0,1,2].
    """
    if force_cpu:
        A= state.site(coord).cpu()
    else:
        A= state.site(coord)
    dimsA= A.size()
    dof1_pd= state.get_physical_dim()
    A_reshaped= A.view( [dof1_pd]*3 + list(dimsA[1:]) )
    # TODO use set to ignore order ?
    if open_sites == [0, 1, 2]:
        contraction = 'mikefgh,njlabcd->eafbgchdmiknjl'
    if open_sites == [1, 2]:
        contraction = 'mikefgh,mjlabcd->eafbgchdikjl'
    if open_sites == [0, 2]:
        contraction = 'mikefgh,nilabcd->eafbgchdmknl'
    if open_sites == [0, 1]:
        contraction = 'mikefgh,njkabcd->eafbgchdminj'
    if open_sites == [0]:
        contraction = 'mikefgh,nikabcd->eafbgchdmn'
    if open_sites == [1]:
        contraction = 'mikefgh,mjkabcd->eafbgchdij'
    if open_sites == [2]:
        contraction = 'mikefgh,milabcd->eafbgchdkl'
    if open_sites == []:
        contraction = 'mikefgh,mikabcd->eafbgchd'
    a = contiguous(einsum(contraction, A_reshaped, conj(A_reshaped)))
    a = view(a, tuple([x**2 for x in dimsA[1:]] + [-1]*(len(open_sites)>0)) )
    return a

def enlarged_corner(coord, state, env, corner, open_sites=[], force_cpu=False,
    verbosity=0):
    r"""
    :param coord: vertex (x,y) for which the enlarged corner is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param corner: which corner to construct. The four options are: 'LU', 'RU', 'RD', and 'LD' 
                   for "left up" corner, "right up" corner, "right down" corner, and "left down" corner.
    :param open_sites: a list DoFs to leave open (uncontracted).
    :param force_cpu: perform on CPU
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type corner: str
    :type open_sites: list(int)
    :type force_cpu: bool
    :return: result of (partial) contraction of double-layer tensor 
    :rtype: torch.tensor

    Builds enlarged corner relative to the site at ``coord`` from the environment:: 

        C---T---                            |   |
        |   |                            --a*a--T
        T--a*a--                            |   |
        |   |     for corner='LU', or    ---T---C  for corner='RD'

    The resulting tensor is always reshaped into either rank-2 or rank-3 if some DoFs are left open
    on the double-layer. In the latter case, these open physical indices are aggregated into 
    the last index of the resulting tensor. The index-ordering convention for enlarged corners
    follows convention for corner tensors of the environment ``env``.

    If ``open_sites=0`` returned tensor has rank-2, where env. indices and auxiliary indices
    of double-layer tensor in the same direction were reshaped into a single index. 
    If some DoFs remain open, then returned tensor is rank-3 with extra index carrying 
    all physical DoFs reshaped in `:math:`|ket \rangle\langle bra|` order::

        C---T---\                             C---T---\
        |   |    --1                          |   |    --1   
        T--a*a--/                             T--a*a--/
         \ /                                   \ / \
          |                                     |   2
          0           for open_sites=[], or     0           for non-empty open_sites
    """
    pleg= len(open_sites)>0
    dof1_pd= state.get_physical_dim()
    if corner == 'LU':
        if force_cpu:
            C = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
            T1 = env.T[(state.vertexToSite(coord), (0, -1))].cpu()
            T2 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        else:
            C = env.C[(state.vertexToSite(coord), (-1, -1))]
            T1 = env.T[(state.vertexToSite(coord), (0, -1))]
            T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
        a = double_layer_a(state, coord, open_sites, force_cpu=force_cpu)

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
        # 2->1    2\...
        C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

        # permute 0123...->1203...
        # reshape (12)(03)...->01...
        # C2x2--1
        # |\...
        # 0
        C2x2_LU = contiguous(permute(C2x2_LU, tuple([1, 2, 0, 3]+[4]*pleg)))
        C2x2_LU = view(C2x2_LU, tuple([T2.size(1) * a.size(2), T1.size(2) * a.size(3)]\
            +[-1]*pleg))
        if verbosity > 0:
            print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))
        return C2x2_LU

    if corner == 'RU':
        if force_cpu:
            C = env.C[(coord, (1, -1))].cpu()
            T1 = env.T[(coord, (1, 0))].cpu()
            T2 = env.T[(coord, (0, -1))].cpu()
        else:
            C = env.C[(coord, (1, -1))]
            T1 = env.T[(coord, (1, 0))]
            T2 = env.T[(coord, (0, -1))]
        a = double_layer_a(state, coord, open_sites, force_cpu=force_cpu)

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
        #    .. 3       |
        #      \0       |
        # 2<-1--a--3 0--T1
        #    3<-2    0<-1
        C2x2_RU = contract(C2x2_RU, a, ([0, 3], [3, 0]))

        # permute 0123...->1203...
        # reshape (12)(03)...->01...
        # 0--C2x2
        # ../|
        #    1
        C2x2_RU = contiguous(permute(C2x2_RU, tuple([1, 2, 0, 3]+[4]*pleg)))
        C2x2_RU = view(C2x2_RU, tuple([T2.size(0) * a.size(1), T1.size(2) * a.size(2)]\
            +[-1]*pleg))
        if verbosity > 0:
            print(
                "C2X2 RU " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,-1): " + str(
                    C2x2_RU.size()))
        return C2x2_RU

    if corner == 'RD':
        if force_cpu:
            C = env.C[(coord, (1, 1))].cpu()
            T1 = env.T[(coord, (0, 1))].cpu()
            T2 = env.T[(coord, (1, 0))].cpu()
        else:
            C = env.C[(coord, (1, 1))]
            T1 = env.T[(coord, (0, 1))]
            T2 = env.T[(coord, (1, 0))]
        a = double_layer_a(state, coord, open_sites, force_cpu=force_cpu)

        #    1<-0        0
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
        #       2\...   |
        #       0       |
        # 0<-1--T1------C
        C2x2_RD = contract(C2x2_RD, a, ([0, 3], [2, 3]))

        # permute 0123...->1203...
        # reshape (12)(03)...->01...
        #    0 ...
        #    |/
        # 1--C2x2
        C2x2_RD = contiguous(permute(C2x2_RD, tuple([1, 2, 0, 3]+[4]*pleg)))
        C2x2_RD = view(C2x2_RD, tuple([T2.size(0) * a.size(0), T1.size(1) * a.size(1)]\
            +[-1]*pleg))
        if verbosity > 0:
            print("C2X2 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,1): " + str(
                C2x2_RD.size()))
        return C2x2_RD

    if corner == 'LD':
        if force_cpu:
            C = env.C[(coord, (-1, 1))].cpu()
            T1 = env.T[(coord, (-1, 0))].cpu()
            T2 = env.T[(coord, (0, 1))].cpu()
        else:
            C = env.C[(coord, (-1, 1))]
            T1 = env.T[(coord, (-1, 0))]
            T2 = env.T[(coord, (0, 1))]
        a = double_layer_a(state, coord, open_sites, force_cpu=force_cpu)

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
        # |        2\...
        # |        2
        # C--------T2--3->1
        C2x2_LD = contract(C2x2_LD, a, ([1, 2], [1, 2]))

        # permute 0123...->0213...
        # reshape (02)(13)...->01...
        # 0 ...
        # |/
        # C2x2--1
        C2x2_LD = contiguous(permute(C2x2_LD, tuple([0, 2, 1, 3]+[4]*pleg)))
        C2x2_LD = view(C2x2_LD, tuple([T1.size(0) * a.size(0), T2.size(2) * a.size(3)]\
            +[-1]*pleg))
        if verbosity > 0:
            print(
                "C2X2 LD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (-1,1): " + str(
                    C2x2_LD.size()))
        return C2x2_LD

# ----- main environment contraction functions - 1x1 subsytem -----
def trace1x1_dn_kagome(coord, state, env, op, verbosity=0, force_cpu=False):
    r"""
    :param coord: vertex (x,y) for which the reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op: operator to be contracted. It is expected that the ``op`` is either 
        rank-6 tensor of shape [physical_dim]*6 or rank-2 tensor 
        of shape [physical_dim**3]*2 (fused bra and ket spaces)
    :param verbosity: logging verbosity
    :param force_cpu: perform on CPU
    :type force_cpu: bool
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type op: torch.tensor
    :type verbosity: int
    :return: expectation value of the given on-site observable ``op``
    :rtype: torch.tensor

    ::

        y\x -1 0   1
        -1  C1 T4  C4
         0  T1 a*a T3
         1  C2 T2  C3 

    Evaluate operator ``op`` supported on the three sites of the down triangle
    of Kagome lattice :math:`Tr{\rho_{1x1,ABC} op}` centered on vertex ``coord``.
    """
    assert len(op.size())==2 or len(op.size())==6,"Invalid operator"
    if force_cpu:
        # counter-clockwise
        op= op.cpu()
        A= state.site(coord).cpu()
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))].cpu()
        C3 = env.C[(state.vertexToSite(coord), (1, 1))].cpu()
        C4 = env.C[(state.vertexToSite(coord), (1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        T2 = env.T[(state.vertexToSite(coord), (0, 1))].cpu()
        T3 = env.T[(state.vertexToSite(coord), (1, 0))].cpu()
        T4 = env.T[(state.vertexToSite(coord), (0,-1))].cpu()
    else:
        A= state.site(coord)
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))]
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))]
        C3 = env.C[(state.vertexToSite(coord), (1, 1))]
        C4 = env.C[(state.vertexToSite(coord), (1, -1))]
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))]
        T2 = env.T[(state.vertexToSite(coord), (0, 1))]
        T3 = env.T[(state.vertexToSite(coord), (1, 0))]
        T4 = env.T[(state.vertexToSite(coord), (0,-1))]

    if len(op.size())==6 and len(set(op.size()))==1:
        op= op.view([op.size(0)**3]*2)

    # C1(-1,-1)--1->0
    # 0
    # 0
    # T1(-1,0)--2
    # 1
    trace = contract(C1,T1,([0],[0]))
    if verbosity>0:
        print("rdm=CT "+str(trace.size()))
    # C1(-1,-1)--0
    # |
    # T1(-1,0)--2->1
    # 1
    # 0
    # C2(-1,1)--1->2
    trace = contract(trace,C2,([1],[0]))
    if verbosity>0:
        print("trace=CTC "+str(trace.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T2(0,1)--2->3
    trace = contract(trace,T2,([2],[1]))
    if verbosity>0:
        print("trace=CTCT "+str(trace.size()))
    # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
    #        Possibly reshape indices 1,2 of rdm, which are to be contracted with
    #        on-site tensor and contract bra,ket in two steps instead of creating
    #        double layer tensor
    #    /
    # --A*--
    #  /|
    #   op
    #   |/
    # --A--
    #  /
    #
    dimsA = A.size()
    #   A=|ket>_i(x)|..>  op=|ket>_i(x)<bra|_j  A*=<bra|_j(x)<...|
    a = torch.einsum('iabcd,ji,jefgh->aebfcgdh', A, op, conj(A))
    a = view(contiguous(a), (dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2))

    # C1(-1,-1)--0
    # |
    # |              0->2
    # T1(-1,0)--1 1--a_op--3
    # |              2
    # |              2
    # C2(-1,1)-------T2(0,1)--3->1
    trace = contract(trace,a,([1,2],[1,2]))
    if verbosity>0:
        print("trace=CTCTa "+str(trace.size()))
    # C1(-1,-1)--0 0--T4(0,-1)--2->0
    # |               1
    # |               2
    # T1(-1,0)--------a_op--3->2
    # |               |
    # |               |
    # C2(-1,1)--------T2(0,1)--1
    trace = contract(T4,trace,([0,1],[0,2]))
    if verbosity>0:
        print("trace=CTCTaT "+str(trace.size()))
    # C1(-1,-1)--T4(0,-1)--0 0--C4(1,-1)
    # |          |             1->0
    # |          |
    # T1(-1,0)---a_op--2
    # |          |
    # |          |
    # C2(-1,1)---T2(0,1)--0->1
    trace = contract(C4,trace,([0],[0]))
    if verbosity>0:
        print("trace=CTCTaTC "+str(trace.size()))
    # C1(-1,-1)--T4(0,-1)----C4(1,-1)
    # |          |           0
    # |          |           0
    # T1(-1,0)---a_op--2 1---T3(1,0)
    # |          |           2->0
    # |          |
    # C2(-1,1)---T2(0,1)--1
    trace = contract(T3,trace,([0,1],[0,2]))
    if verbosity>0:
        print("trace=CTCTaTCT "+str(trace.size()))
    # C1(-1,-1)--T(0,-1)--------C4(1,-1)
    # |          |              |
    # |          |              |
    # T1(-1,0)---a_op-----------T3(1,0)
    # |          |              0
    # |          |              0
    # C2(-1,1)---T2(0,1)--1 1---C3(1,1)
    trace = contract(trace,env.C[(coord,(1,1))],([0,1],[0,1]))
    if verbosity>0:
        print("trace=CTCTaTCTC "+str(trace.size()))

    trace= trace.to(env.device)
    return trace

def rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B', 'C'), force_cpu=False, 
    sym_pos_def=False, verbosity=0, **kwargs):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param sites_to_keep: physical degrees of freedom to be kept. Default: "ABC" - keep all the DOF
    :param force_cpu: perform on CPU
    :type force_cpu: bool
    :param sym_pos_def: make reduced density matrix positive-(semi)definite
    :type sym_pos_def: bool
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor

    Compute 1-kagome-site reduced density matrix :math:`\rho_{1x1,\textrm{sites_to_keep}}` centered 
    on vertex ``coord``::

        y\x -1 0   1
        -1  C1 T4  C4
         0  T1 a*a T3
         1  C2 T2  C3

    The physical indices are ordered as :math:`|ket \rangle\langle bra|` from on-site tensor 
    A (`ket`) and then A^\dag (`bra`). 
    """ 
    who= "rdm1x1_kagome"
    assert len(sites_to_keep)>0,"No physical DoFs remain open"
    if force_cpu:
        # counter-clockwise
        A= state.site(coord).cpu()
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))].cpu()
        C3 = env.C[(state.vertexToSite(coord), (1, 1))].cpu()
        C4 = env.C[(state.vertexToSite(coord), (1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        T2 = env.T[(state.vertexToSite(coord), (0, 1))].cpu()
        T3 = env.T[(state.vertexToSite(coord), (1, 0))].cpu()
        T4 = env.T[(state.vertexToSite(coord), (0,-1))].cpu()
    else:
        A= state.site(coord)
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))]
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))]
        C3 = env.C[(state.vertexToSite(coord), (1, 1))]
        C4 = env.C[(state.vertexToSite(coord), (1, -1))]
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))]
        T2 = env.T[(state.vertexToSite(coord), (0, 1))]
        T3 = env.T[(state.vertexToSite(coord), (1, 0))]
        T4 = env.T[(state.vertexToSite(coord), (0,-1))]        
    
    # C1(-1,-1)--1 0--T4--2
    # 0               1
    C2x1 = contract(C1,T4,([1],[0]))

    # 0->1
    # T1--2
    # 1
    # 0
    # C2--1->0
    C2x2_LD = contract(C2, T1, ([0], [1]))

    # 1->0
    # T1--2->1
    # |
    # |        0->2
    # C2--0 1--T2--2->3
    C2x2_LD = contract(C2x2_LD, T2, ([0], [1]))

    #    0/4,5,6
    # 1--A--3
    #    2
    dimsA= A.size()
    dof1_pd= state.get_physical_dim()
    A_reshaped= A.permute(1,2,3,4,0)
    A_reshaped= A.view( list(dimsA[1:]) + [dof1_pd]*3 )

    # 0     
    # T1--1->1,2 
    # |     
    # |        2->3,4
    # C2-------T2--3->5
    C2x2_LD= C2x2_LD.view([T1.size(0)]+[A_reshaped.size(1)]*2+[A_reshaped.size(2)]*2+[T2.size(2)])


    # 0        0->4
    # |/--2    |/4,5,6->6,7,8
    # T1--1 1--A---3->5
    # |        |
    # |        2
    # |        3  4->2
    # |        | /
    # C2-------T2--5->3
    C2x2_LD = contract(C2x2_LD, A_reshaped, ([1,3], [1, 2]))

    # 0     2<-4 x<-0/4,5,6   => 
    # |  1 1--------A*--3->y
    # | /      |    |             0  1<-2 2<-x |ket><bra|
    # |/       |/6,7,8->4...      |      \  | /
    # T1-------A--------5->3      T1------A*A-------3->4
    # |        |    2             |        | -------y->5
    # |        |  2               |        |
    # |        | /                |        |
    # C2-------T2-------3->1      C2-------T2-------1->3
    #
    # where position of x = 3+len(sites_to_keep)+1, and y = 3 + len(sites_to_keep) + 2 
    stk= _abc_to_012_site(sites_to_keep)
    to_contract= tuple(set([0,1,2])-set(stk))
    C2x2_LD= contract(C2x2_LD, A_reshaped.conj(),\
        ([1, 2]+[ 6+i for i in to_contract ], [1, 2]+[ 4+i for i in to_contract ]))
    offset0, offset1= 4, 3+len(stk)+2+1
    C2x2_LD= C2x2_LD.permute([0,2,3+len(stk)+1,1,3,3 + len(stk) + 2]\
        +list(range(offset0,offset0+len(stk)))+list(range(offset1, offset1+len(stk))))\
        .reshape([T1.size(0),A_reshaped.size(0)**2,T2.size(2),A_reshaped.size(3)**2,\
            dof1_pd**(2*len(sites_to_keep))])

    # C1-------T4--2->0
    # 0        1
    # 0        1 |ket><bra|
    # |        |/
    # T1------A*A-------3->2
    # |        |
    # C2-------T2-------2->1
    C2x2_LD = contract(C2x1,C2x2_LD, ([0,1], [0,1]))

    # 0--C4(1,-1)
    #    1
    #    0
    # 1--T3(1,0)
    #    2
    C2x1 = contract(C4,T3,([1],[0]))
    
    # C1(-1,-1)--T4(0,-1)--0 0--C4(1,-1)
    # |          |              |
    # |          |              |
    # T1(-1,0)--A*A--------2 1--T3(1,0)
    # |          |\|ket><bra|   2->0
    # |          |         
    # C2(-1,1)---T2(0,1)--1
    C2x2_LD = contract(C2x1,C2x2_LD,([0,1],[0,2]))

    # C1(-1,-1)--T4(0,-1)-------C4(1,-1)
    # |          |              |
    # |          |              |
    # T1(-1,0)--A*A-------------T3(1,0)
    # |          |\|ket><bra|   0
    # |          |              0
    # C2(-1,1)---T2(0,1)--1 1----C3(1,1)
    rdm = contract(C2x2_LD,C3,([0,1],[0,1]))

    # reshape, symmetrize and normalize
    rdm= rdm.view([dof1_pd]*(2*len(sites_to_keep)))
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who, **kwargs)
    rdm= rdm.to(env.device)

    return rdm

def _old_rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B', 'C'), force_cpu=False, 
    sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param sites_to_keep: physical degrees of freedom to be kept. Default: "ABC" - keep all the DOF
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor

    Compute 1-kagome-site reduced density matrix :math:`\rho{1x1}_{sites_to_keep}` centered on vertex ``coord``.
    Inherited from the rdm1x1() method.
    """
    #    y\x -1 0 1
    # -1  C1 T4 C4
    #  0  T1 A  T3
    #  1  C2 T2 C3 
    who= "rdm1x1_kagome"
    if force_cpu:
        # counter-clockwise
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))].cpu()
        C3 = env.C[(state.vertexToSite(coord), (1, 1))].cpu()
        C4 = env.C[(state.vertexToSite(coord), (1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        T2 = env.T[(state.vertexToSite(coord), (0, 1))].cpu()
        T3 = env.T[(state.vertexToSite(coord), (1, 0))].cpu()
        T4 = env.T[(state.vertexToSite(coord), (0,-1))].cpu()
    else:
        C1 = env.C[(state.vertexToSite(coord), (-1, -1))]
        C2 = env.C[(state.vertexToSite(coord), (-1, 1))]
        C3 = env.C[(state.vertexToSite(coord), (1, 1))]
        C4 = env.C[(state.vertexToSite(coord), (1, -1))]
        T1 = env.T[(state.vertexToSite(coord), (-1, 0))]
        T2 = env.T[(state.vertexToSite(coord), (0, 1))]
        T3 = env.T[(state.vertexToSite(coord), (1, 0))]
        T4 = env.T[(state.vertexToSite(coord), (0,-1))]        
    # C1(-1,-1)--1->0
    # 0
    # 0
    # T1(-1,0)--2
    # 1
    rdm = contract(C1,T1,([0],[0]))
    if verbosity>0:
        print("rdm=CT "+str(rdm.size()))
    # C1(-1,-1)--0
    # |
    # T1(-1,0)--2->1
    # 1
    # 0
    # C2(-1,1)--1->2
    rdm = contract(rdm,C2,([1],[0]))
    if verbosity>0:
        print("rdm=CTC "+str(rdm.size()))
    # C(-1,-1)--0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T2(0,1)--2->3
    rdm = contract(rdm,T2,([2],[1]))
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
    a= double_layer_a(state,coord,_abc_to_012_site(sites_to_keep), force_cpu=force_cpu)

    # C1(-1,-1)--0
    # |
    # |              0->2
    # T1(-1,0)--1 1--a--3
    # |              2\4(s,s')
    # |              2
    # C1(-1,1)-------T2(0,1)--3->1
    rdm = contract(rdm,a,([1,2],[1,2]))
    if verbosity>0:
        print("rdm=CTCTa "+str(rdm.size()))
    # C1(-1,-1)--0 0--T4(0,-1)--2->0
    # |               1
    # |               2
    # T1(-1,0)--------a--3->2
    # |               |\4->3(s,s')
    # |               |
    # C2(-1,1)--------T2(0,1)--1
    rdm = contract(T4,rdm,([0,1],[0,2]))
    if verbosity>0:
        print("rdm=CTCTaT "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)--0 0--C4(1,-1)
    # |         |             1->0
    # |         |
    # T(-1,0)---a--2
    # |         |\3(s,s')
    # |         |
    # C(-1,1)---T(0,1)--0->1
    rdm = contract(C4,rdm,([0],[0]))
    if verbosity>0:
        print("rdm=CTCTaTC "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)-------C4(1,-1)
    # |         |             0
    # |         |             0
    # T(-1,0)---a--2 1--------T3(1,0)
    # |         |\3->2(s,s')  2->0
    # |         |
    # C(-1,1)---T(0,1)--1
    rdm = contract(T3,rdm,([0,1],[0,2]))
    if verbosity>0:
        print("rdm=CTCTaTCT "+str(rdm.size()))
    # C(-1,-1)--T(0,-1)--------C4(1,-1)
    # |         |              |
    # |         |              |
    # T(-1,0)---a--------------T3(1,0)
    # |         |\2->1(s,s')   0
    # |         |              0
    # C(-1,1)---T(0,1)--1 1----C3(1,1)
    rdm = contract(rdm,C3,([0,1],[0,1]))
    if verbosity>0:
        print("rdm=CTCTaTCTC "+str(rdm.size()))

    # reshape, symmetrize and normalize
    dof1_pd= state.get_physical_dim()
    rdm= rdm.view([dof1_pd]*(2*len(sites_to_keep)))
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
    rdm= rdm.to(env.device)

    return rdm

# ----- 2x1 or 1x2 subsystem -----

def rdm2x1_kagome(coord, state, env, sites_to_keep_00=('A', 'B', 'C'),\
    sites_to_keep_10=('A', 'B', 'C'), force_cpu=False, sym_pos_def=False,\
    verbosity=0, **kwargs):
    r"""
    :param coord: vertex (x,y) specifies position of 2x1 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param sites_to_keep_00: physical sites needed for the unit cell at coord + (0, 0)
    :param sites_to_keep_10: physical sites needed for the unit cell at coord + (1, 0)
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
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
    who = "rdm2x1_kagome"
    # ----- building C2x2_LU ----------------------------------------------------
    C2x2_LU = enlarged_corner(coord, state, env, 'LU',open_sites=_abc_to_012_site(\
        sites_to_keep_00),force_cpu=force_cpu, verbosity=verbosity)
    # C2x2--1
    # |\2
    # 0

    # ----- building C2x1_LD ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord), (-1, 1))]
    T2 = env.T[(state.vertexToSite(coord), (0, 1))]

    # 0       0->1
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
    # |\2
    # 0
    # 0
    # C2x1_LD--1->0
    left_half = contract(C2x1_LD, C2x2_LU, ([0], [0]))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_RU= enlarged_corner(shitf_coord, state, env, 'RU',open_sites=_abc_to_012_site(\
        sites_to_keep_10),force_cpu=force_cpu, verbosity=verbosity)
    # 0--C2x2
    #  2/|
    #    1

    # ----- building C2x1_RD ----------------------------------------------------
    C = env.C[(shitf_coord, (1, 1))]
    T1 = env.T[(shitf_coord, (0, 1))]

    #    1<-0        0
    # 2<-1--T1--2 1--C
    C2x1_RD = contract(C, T1, ([1], [2]))

    # reshape (01)2->(0)1
    C2x1_RD = view(contiguous(C2x1_RD), (C.size(0) * T1.size(0), T1.size(1)))

    #    0
    #    |
    # 1--C2x1
    if verbosity > 0:
        print("C2X1 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,1): " + str(
            C2x1_RD.size()))

    # ----- build right part C2x2_RU--C2x1_RD -----------------------------------
    # 1<-0--C2x2_RU
    #       |\2
    #       1
    #       0
    # 0<-1--C2x1_RD
    right_half = contract(C2x1_RD, C2x2_RU, ([0], [1]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2_LU--1 1----C2x2_RU
    # |\2->0          |\2->1
    # |               |
    # C2x1_LD--0 0----C2x1_RD
    rdm = contract(left_half, right_half, ([0, 1], [0, 1]))

    # reshape into single DoF indices
    dof1_pd= state.get_physical_dim()
    l00, l10= len(sites_to_keep_00), len(sites_to_keep_10)
    rdm= rdm.view( [dof1_pd]*(2*(l00+l10)) )
    # permute into order bra,ket order. Fused index 0 and index 1 is obtained
    # from bra,ket indices of uncontrated DoFs.
    perm_order= _expand_perm([l00,l10])
    rdm= rdm.permute(tuple(perm_order)).contiguous()

    # symmetrize and normalize
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who, **kwargs)

    return rdm

def rdm1x2_kagome(coord, state, env, sites_to_keep_00=('A', 'B', 'C'),\
    sites_to_keep_01=('A', 'B', 'C'), sym_pos_def=False, force_cpu=False,\
    verbosity=0, **kwargs):
    r"""
    :param coord: vertex (x,y) specifies position of 1x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param sites_to_keep_00: physical sites needed for the unit cell at coord + (0, 0)
    :param sites_to_keep_01: physical sites needed for the unit cell at coord + (0, 1)
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: torch.tensor

    Computes 2-site reduced density matrix :math:`\rho_{1x2}` of a vertical
    1x2 subsystem using following strategy:
    """
    who = "rdm1x2_kagome"
    # ----- building C2x2_LU ----------------------------------------------------
    C2x2_LU = enlarged_corner(coord, state, env, 'LU',open_sites=_abc_to_012_site(\
        sites_to_keep_00),force_cpu=force_cpu, verbosity=verbosity)
    # C2x2--1
    # |\2
    # 0

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
    #    |
    #    1
    C1x2_RU = view(contiguous(C1x2_RU), (C.size(0) * T1.size(1), T1.size(2)))
    if verbosity > 0:
        print("C1X2 RU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (1,-1): " + str(C1x2_RU.size()))

    # ----- build upper part C2x2_LU--C1x2_RU -----------------------------------
    # C2x2_LU--1 0--C1x2_RU
    # |\2           |
    # 0->1          1->0
    upper_half = contract(C1x2_RU, C2x2_LU, ([0], [1]))

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_LD = enlarged_corner(shitf_coord, state, env, 'LD',open_sites=_abc_to_012_site(\
        sites_to_keep_01),force_cpu=force_cpu, verbosity=verbosity)
    # 0
    # |/2
    # C2x2--1

    # ----- building C2x2_RD ----------------------------------------------------
    C = env.C[(shitf_coord, (1, 1))]
    T2 = env.T[(shitf_coord, (1, 0))]

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
        print("C1X2 RD " + str((coord[0] + vec[0], coord[1] + vec[1])) + "->" + str(shitf_coord) + " (1,1): " + str(
            C1x2_RD.size()))

    # ----- build lower part C2x2_LD--C1x2_RD -----------------------------------
    # 0->1          0
    # |/2           |
    # C2x2_LD--1 1--C1x2_RD
    lower_half = contract(C1x2_RD, C2x2_LD, ([1], [1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C1x2_RU
    # |\2->0       |
    # 1            0
    # 1            0
    # |/2->1       |
    # C2x2_LD------C1x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # reshape into single DoF indices
    dof1_pd= state.get_physical_dim()
    l00, l01= len(sites_to_keep_00), len(sites_to_keep_01)
    rdm= rdm.view( [dof1_pd]*(2*(l00+l01)) )
    # permute into order bra,ket order. Fused index 0 and index 1 is obtained
    # from bra,ket indices of uncontrated DoFs.
    perm_order= _expand_perm([l00,l01])
    rdm= rdm.permute(tuple(perm_order)).contiguous()

    # symmetrize and normalize
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who, **kwargs)

    return rdm


# ----- 2x2 subsystem -----
def rdm2x2_up_triangle_open(coord, state, env, sym_pos_def=False, force_cpu=False,\
    verbosity=0, **kwargs):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param force_cpu: perform on CPU
    :type force_cpu: bool
    :param sym_pos_def: make reduced density matrix positive-(semi)definite
    :type sym_pos_def: bool
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :return: reduced density matrix as rank-6 tensor 
    :rtype: torch.tensor

    Build reduced density matrix corresponding to the three sites s0, s1, and s2 
    of the "up" triangle of Kagome lattice::


        C    T             T          C => C2x2_LU(coord)--------C2x2(coord+(1,0))
             a             a               |                  s1/|
             |             |               |/s2               s0\|
        T b--\          b--\               C2x2_LD(coord+(0,1))--C2x2(coord+(1,1))
              \        /    \              
              XX--XX--d     XX--XX--d T
               | /           | /
               |/            |/
              XX            s1
               |             |
               c             c  
              /             /
             a             a
             |             |
        T b--\          b--\
              \        /    \
              XX--s2--d     s0--XX--d T
               | /           | /
               |/            |/
              XX            XX
               |             |
               c             c
        C      T             T        C

    """
    who = "rdm2x2"
    # ----- building C2x2_LU ----------------------------------------------------
    C2x2_LU = enlarged_corner(coord, state, env, 'LU', force_cpu=force_cpu,\
        verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_RU= enlarged_corner(shitf_coord, state, env, 'RU', open_sites=[1],\
        force_cpu=force_cpu, verbosity=verbosity)
    # 0--C2x2
    #  2/|
    #    1

    # ----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    #
    # C2x2_LU--1 0--C2x2_RU
    # |              |\2
    # 0              1
    #
    upper_half = contract(C2x2_LU, C2x2_RU, ([1], [0]))

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 1)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_RD = enlarged_corner(shitf_coord, state, env, 'RD', open_sites=[0],\
        force_cpu=force_cpu, verbosity=verbosity)
    #    0
    #    |/2
    # 1--C2x2

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_LD = enlarged_corner(shitf_coord, state, env, 'LD', open_sites=[2],\
        force_cpu=force_cpu, verbosity=verbosity)
    # 0
    # |/2
    # C2x2--1

    # ----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->2                0          2->1
    # |/2->1        |/2->3   & permute  |/1->2     |/3
    # C2x2_LD--1 1--C2x2_RD             C2x2_LD----C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))
    lower_half = permute(lower_half, (0, 2, 1, 3))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |             |\2->0
    # 0             1
    # 0             1
    # |/2->1        |/3->2
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # unfuse combined indices (s'0,s0)(s'1,s1)(s'2,s2)->s'0,s0,s'1,s1,s'2,s2
    dof1_pd= state.get_physical_dim()
    rdm = rdm.view([dof1_pd]*6)
    # permute into order of |ket><bra| = s0,s1,s2;s0',s1',s2' where primed indices
    # represent "bra"
    # 012345 -> 024135
    # C2x2_LU------C2x2_RU
    # |             |\0->0,1->0,3
    # 0             1
    # 0             1
    # |/1->2,3->1,4 |/2->4,5->2,5
    # C2x2_LD------C2x2_RD
    rdm = contiguous(permute(rdm, (0, 2, 4, 1, 3, 5)))

    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who, **kwargs)

    rdm = rdm.to(env.device)
    return rdm

def rdm2x2_dn_triangle_with_operator(coord, state, env, op, force_cpu=False,\
    verbosity=0,**kwargs):
    r"""
    :param coord: vertex (x,y) for which the reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :param op: operator to be contracted. It is expected that the op is either 
        rank-6 tensor of shape [physical_dim]*6 or rank-2 tensor 
        of shape [physical_dim**3]*2 (fused bra and ket spaces)
    :type op: torch.tensor
    :param force_cpu: perform on CPU
    :type force_cpu: bool
    :return: normalized expectation value of the operator `op` and the norm 
             of the reduced density matrix
    :rtype: torch.tensor, torch.tensor

    Returns a normalized expectation value of operator inserted into down triangle 
    of upper left corner of 2x2 subsystem::

        C    T             T          C
             a             a
             |             |
        T b--\          b--\
              \        /    \
              s0--s2--d     XX--XX--d T
               | /           | /
               |/            |/
              s1            XX
               |             |
               c             c  
              /             /
             a             a
             |             |
        T b--\          b--\
              \        /    \
              XX--XX--d     XX--XX--d T
               | /           | /
               |/            |/
              XX            XX
               |             |
               c             c
        C      T             T        C
    """
    who = 'rdm2x2_dn_triangle_with_operator'
    assert len(op.size())==2 or len(op.size())==6,"Invalid operator"
    if len(op.size())==6 and len(set(op.size()))==1:
        op= op.view([op.size(0)**3]*2)

    # ----- building C2x2_LU ----------------------------------------------------
    if force_cpu:
        C = env.C[(state.vertexToSite(coord), (-1, -1))].cpu()
        T1 = env.T[(state.vertexToSite(coord), (0, -1))].cpu()
        T2 = env.T[(state.vertexToSite(coord), (-1, 0))].cpu()
        a_1layer = state.site(coord).cpu()
        op = op.cpu()
    else:
        C = env.C[(state.vertexToSite(coord), (-1, -1))]
        T1 = env.T[(state.vertexToSite(coord), (0, -1))]
        T2 = env.T[(state.vertexToSite(coord), (-1, 0))]
        a_1layer = state.site(coord)
    dimsA = a_1layer.size()

    a = contiguous(einsum('mefgh,mabcd->eafbgchd', a_1layer, conj(a_1layer)))
    a = view(a, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))
    a_op = contiguous(
        einsum('mefgh,nm,nabcd->eafbgchd', a_1layer, op, conj(a_1layer)))
    a_op = view(a_op, (dimsA[1] ** 2, dimsA[2] ** 2, dimsA[3] ** 2, dimsA[4] ** 2))

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
    C2x2_LU_op = contract(C2x2_LU, a_op, ([0, 3], [0, 1]))
    C2x2_LU = contract(C2x2_LU, a, ([0, 3], [0, 1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    # C2x2--1
    # |\23
    # 0

    C2x2_LU_op = contiguous(permute(C2x2_LU_op, (1, 2, 0, 3)))
    C2x2_LU_op = view(C2x2_LU_op, (T2.size(1) * a.size(2), T1.size(2) * a.size(3)))
    C2x2_LU = contiguous(permute(C2x2_LU, (1, 2, 0, 3)))
    C2x2_LU = view(C2x2_LU, (T2.size(1) * a.size(2), T1.size(2) * a.size(3)))
    if verbosity > 0:
        print("C2X2 LU " + str(coord) + "->" + str(state.vertexToSite(coord)) + " (-1,-1): " + str(C2x2_LU.size()))

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shift_coord = _shift_coord(state,coord,vec)
    C2x2_RU = enlarged_corner(shift_coord, state, env, 'RU', force_cpu=force_cpu,\
        verbosity=verbosity)

    # ----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU
    # |              |
    # 0              1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?
    upper_half_op = contract(C2x2_LU_op, C2x2_RU, ([1], [0]))
    upper_half = contract(C2x2_LU, C2x2_RU, ([1], [0]))

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 1)
    shift_coord = _shift_coord(state,coord,vec)
    C2x2_RD = enlarged_corner(shift_coord, state, env, 'RD', force_cpu=force_cpu,\
        verbosity=verbosity)

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shift_coord = _shift_coord(state,coord,vec)
    C2x2_LD = enlarged_corner(shift_coord, state, env, 'LD', force_cpu=force_cpu,\
        verbosity=verbosity)

    # ----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->1
    # |             |
    # C2x2_LD--1 1--C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |            |
    # 0            1
    # 0            1
    # |            |
    # C2x2_LD------C2x2_RD
    rdm_op = contract(upper_half_op, lower_half, ([0, 1], [0, 1]))
    rdm_id = contract(upper_half, lower_half, ([0, 1], [0, 1]))
    rdm_id = _cast_to_real(rdm_id,who=who,**kwargs)

    rdm = rdm_op/rdm_id
    rdm_id = rdm_id.to(env.device)
    rdm = rdm.to(env.device)
    return rdm, rdm_id

def rdm2x2_kagome(coord, state, env, sites_to_keep_00=('A', 'B', 'C'),\
    sites_to_keep_10=('A', 'B', 'C'), sites_to_keep_01=('A', 'B', 'C'),\
    sites_to_keep_11=('A', 'B', 'C'), force_cpu=False, sym_pos_def=False,\
    verbosity=0,**kwargs):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :param sites_to_keep_00: physical sites needed for the unit cell at coord + (0, 0)
    :param sites_to_keep_10: physical sites needed for the unit cell at coord + (1, 0)
    :param sites_to_keep_01: physical sites needed for the unit cell at coord + (0, 1)
    :param sites_to_keep_11: physical sites needed for the unit cell at coord + (1, 1)
    :type coord: tuple(int,int)
    :type state: IPEPS_KAGOME
    :type env: ENV
    :type verbosity: int
    :param force_cpu: perform on CPU
    :type force_cpu: bool
    :param sym_pos_def: make reduced density matrix positive-(semi)definite
    :type sym_pos_def: bool
    :return: 4-site reduced density matrix with indices :math:`s_0s_1s_2s_3;s'_0s'_1s'_2s'_3`
    :rtype: torch.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem specified
    by the vertex ``coord`` of its upper left corner using strategy:
    
        1. compute four individual corners
        2. construct upper and lower half of the network
        3. contract upper and lower half to obtain final reduced density matrix
    
    ::

        C--T------------------T------------------C = C2x2_LU(coord)--------C2x2_RU(coord+(1,0))
        |  |                  |                  |   |                     |
        T--A^+A(coord)--------A^+A(coord+(1,0))--T   C2x2_LD(coord+(0,1))--C2x2_RD(coord+(1,1))
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
    who = "rdm2x2_kagome"
    # ----- building C2x2_LU ----------------------------------------------------
    C2x2_LU = enlarged_corner(coord, state, env, 'LU',open_sites=_abc_to_012_site(\
        sites_to_keep_00),force_cpu=force_cpu, verbosity=verbosity)
    # C2x2--1
    # |\2
    # 0

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, 0)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_RU= enlarged_corner(shitf_coord, state, env, 'RU',open_sites=_abc_to_012_site(\
        sites_to_keep_10),force_cpu=force_cpu, verbosity=verbosity)
    # 0--C2x2
    #  2/|
    #    1

    # ----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU              C2x2_LU------C2x2_RU
    # |\2->1        |\2->3   & permute   |\1->2        |\3
    # 0             1->2                 0             2->1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2x2_RU ?
    upper_half = contract(C2x2_LU, C2x2_RU, ([1], [0]))
    if len(sites_to_keep_00)>0 and len(sites_to_keep_10):
        upper_half = permute(upper_half, (0, 2, 1, 3))
    elif len(sites_to_keep_00)>0:
        upper_half = permute(upper_half, (0, 2, 1))

    # ----- building C2x2_RD ----------------------------------------------------
    vec = (1, 1)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_RD= enlarged_corner(shitf_coord, state, env, 'RD',open_sites=_abc_to_012_site(\
        sites_to_keep_11),force_cpu=force_cpu, verbosity=verbosity)

    #    0
    #    |/2
    # 1--C2x2

    # ----- building C2x2_LD ----------------------------------------------------
    vec = (0, 1)
    shitf_coord = _shift_coord(state,coord,vec)
    C2x2_LD = enlarged_corner(shitf_coord, state, env, 'LD',open_sites=_abc_to_012_site(\
        sites_to_keep_01),force_cpu=force_cpu, verbosity=verbosity)
    # 0
    # |/2
    # C2x2--1

    # ----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->2                 0             2->1
    # |/2->1        |/2->3   & permute   |/1->2        |/3
    # C2x2_LD--1 1--C2x2_RD              C2x2_LD------C2x2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LD,C2x2_RD ?
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))
    if len(sites_to_keep_01)>0 and len(sites_to_keep_11):
        lower_half = permute(lower_half, (0, 2, 1, 3))
    elif len(sites_to_keep_01)>0:
        lower_half = permute(lower_half, (0, 2, 1))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # |\2->0       |\3->1
    # 0            1
    # 0            1
    # |/2->2       |/3->4
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half, lower_half, ([0, 1], [0, 1]))

    # unfuse physical indices and permute them to bra,ket order
    dof1_pd= state.get_physical_dim()
    l00,l01,l10,l11= len(sites_to_keep_00), len(sites_to_keep_01),\
        len(sites_to_keep_10),len(sites_to_keep_11)
    rdm= rdm.view([dof1_pd]*(2*(l00+l01+l10+l11)))

    # permute into order of bra;ket order 
    perm_order= _expand_perm([l00,l10,l01,l11])
    rdm = contiguous(permute(rdm, tuple(perm_order)))

    # symmetrize and normalize
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who,\
        **kwargs)

    rdm = rdm.to(env.device)
    return rdm


# ----- next-to-next nearest neighbour interactions -----
# TODO? recomputing corners from scratch might be not neccessary if the only difference
#       is which DoF remains uncontracted
def rdm2x2_nnn_1(coord, state, env, operator, force_cpu=False, verbosity=0):
    r"""
    :param operator: two-site operator (rank-4 tensor), which acts on two DoFs of Kagome
                     lattice
    :type operator: torch.tensor
        
       C    T       T       C     and     C    T       T       C   
          / |     / |                        / |     / |  
         /  |    /  |                       /  |    /  |
       T --XX--XX--XX--XX-- T             T --XX--XX--s0--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX      s1                         XX      XX
          / |     / |                        / |     / | 
         /  |    /  |                       /  |    /  |
       T --s0--XX--XX--XX-- T             T --XX--s2--XX--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX       XX                        XX       XX
            |       |                          |       |
       C    T       T       C             C    T       T       C
    """
    C2x2_LU= enlarged_corner(coord, state, env, 'LU', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,1))
    C2x2_RD= enlarged_corner(shift_coord, state, env, 'RD', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)

    # bond 1--2
    # TODO? split operator by SVD and apply to individual corners
    shift_coord= _shift_coord(state,coord,(0,1))
    C2x2_LD = enlarged_corner(shift_coord, state, env, 'LD', csites=[0],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,0))
    C2x2_RU = enlarged_corner(shift_coord, state, env, 'RU', csites=[1],\
        force_cpu=force_cpu, verbosity=verbosity)
    upper_half = einsum('ij,jkab->ikab', C2x2_LU, C2x2_RU)
    lower_half = einsum('ijab,kj->ikab', C2x2_LD, C2x2_RD)
    bond_operator = operator.to(C2x2_LD.device)
    bond12 = einsum('ijab,badc,ijcd->', upper_half, bond_operator, lower_half)

    # bond 3--1
    shift_coord= _shift_coord(state,coord,(0,1))
    C2x2_LD = enlarged_corner(shift_coord, state, env, 'LD', csites=[2],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,0))
    C2x2_RU = enlarged_corner(shift_coord, state, env, 'RU', csites=[0],\
        force_cpu=force_cpu, verbosity=verbosity)
    upper_half = einsum('ij,jkab->ikab', C2x2_LU, C2x2_RU)
    lower_half = einsum('ijab,kj->ikab', C2x2_LD, C2x2_RD)
    bond31 = einsum('ijab,badc,ijcd->', upper_half, bond_operator, lower_half)

    bond12 = bond12.to(env.device)
    bond31 = bond31.to(env.device)
    return (bond12, bond31)

def rdm2x2_nnn_2(coord, state, env, operator, force_cpu=False, verbosity=0):
    r"""
    :param operator: two-site operator (rank-4 tensor), which acts on two DoFs of Kagome
                     lattice
    :type operator: torch.tensor
        
       C    T       T       C     and     C    T       T       C   
          / |     / |                        / |     / |  
         /  |    /  |                       /  |    /  |
       T --XX--s2--XX--XX-- T             T --XX--XX--s0--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX      s1                         s1      XX
          / |     / |                        / |     / | 
         /  |    /  |                       /  |    /  |
       T --XX--XX--XX--XX-- T             T --XX--XX--XX--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX       XX                        XX       XX
            |       |                          |       |
       C    T       T       C             C    T       T       C
    """
    # --------------upper half -------------------------------------------------

    # build upper part C2x2_LU--C2x2_RU and contract with the 2-cell operator
    # C2x2_LU-----1     0-----C2x2_RU
    # |\23________op_______23/|
    # 0                       1

    # NNN bond 3--2
    C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[2],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,0))
    C2x2_RU = enlarged_corner(shift_coord, state, env, corner='RU', csites=[1],\
        force_cpu=force_cpu, verbosity=verbosity)
    bond_operator = operator.to(C2x2_LU.device)
    upper_half_32 = einsum('ijab,badc,jkcd->ik', C2x2_LU, bond_operator, C2x2_RU)

    # NNN bond 2--1
    C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[1],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,0))
    C2x2_RU = enlarged_corner(shift_coord, state, env, corner='RU', csites=[0],\
        force_cpu=force_cpu, verbosity=verbosity)
    upper_half_21 = einsum('ijab,badc,jkcd->ik', C2x2_LU, bond_operator, C2x2_RU)

    # --------------bottom half-------------------------------------------------

    # 0             0->1
    # |             |
    # C2x2_LD--1 1--C2x2_RD
    shift_coord= _shift_coord(state,coord,(1,1))
    C2x2_RD = enlarged_corner(shift_coord, state, env, corner='RD', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(0,1))
    C2x2_LD = enlarged_corner(shift_coord, state, env, corner='LD', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)
    lower_half = contract(C2x2_LD, C2x2_RD, ([1], [1]))

    # contracting lower and upper halfs
    # C2x2_LU--op--C2x2_RU
    # |            |
    # 0            1
    # 0            1
    # |            |
    # C2x2_LD------C2x2_RD

    bond32 = contract(upper_half_32, lower_half, ([0, 1], [0, 1])).to(env.device)
    bond21 = contract(upper_half_21, lower_half, ([0, 1], [0, 1])).to(env.device)
    return (bond32, bond21)

def rdm2x2_nnn_3(coord, state, env, operator, force_cpu=False, verbosity=0):
    r"""
    :param operator: two-site operator (rank-4 tensor), which acts on two DoFs of Kagome
                     lattice
    :type operator: torch.tensor
        
       C    T       T       C     and     C    T       T       C   
          / |     / |                        / |     / |  
         /  |    /  |                       /  |    /  |
       T --XX--s2--XX--XX-- T             T --XX--XX--XX--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX      XX                         s1      XX
          / |     / |                        / |     / | 
         /  |    /  |                       /  |    /  |
       T --s0--XX--XX--XX-- T             T --XX--s2--XX--XX-- T
            | /     | /                        | /     | /
            |/      |/                         |/      |/
           XX       XX                        XX       XX
            |       |                          |       |
       C    T       T       C             C    T       T       C
    """
    # ---------------- left half -----------------------------------
    # build left half and contract with the 2-cell operator
    # C2x2_LU--1->0
    # |\23
    # |   \
    # 0    op
    # 0    /
    # |   /
    # |/23
    # C2x2_LD--1

    # NN bond 3--1
    C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[2],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(0,1))
    C2x2_LD = enlarged_corner(shift_coord, state, env, corner='LD', csites=[0],\
        force_cpu=force_cpu, verbosity=verbosity)
    bond_operator = operator.to(C2x2_LU.device)
    left_half_31 = einsum('ijab,badc,ikcd->jk', C2x2_LU, bond_operator, C2x2_LD)

    # NN bond 2--3
    C2x2_LU = enlarged_corner(coord, state, env, corner='LU', csites=[1],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(0,1))
    C2x2_LD = enlarged_corner(shift_coord, state, env, corner='LD', csites=[2],\
        force_cpu=force_cpu, verbosity=verbosity)
    left_half_23 = einsum('ijab,badc,ikcd->jk', C2x2_LU, bond_operator, C2x2_LD)

    # ---------------- right half -----------------------------------

    # 0--C2x2_RU
    #    |
    #    1
    #    0
    #    |
    # 1--C2x2_RD
    shift_coord= _shift_coord(state,coord,(1,0))
    C2x2_RU = enlarged_corner(shift_coord, state, env, corner='RU', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)
    shift_coord= _shift_coord(state,coord,(1,1))
    C2x2_RD = enlarged_corner(shift_coord, state, env, corner='RD', csites=[],\
        force_cpu=force_cpu, verbosity=verbosity)
    right_half = contract(C2x2_RU, C2x2_RD, ([1], [0]))

    # construct reduced density matrix by contracting left and right halves
    # C2x2_LU-0--0-C2x2_RU
    # |            |
    # op           |
    # |            |
    # |            |
    # C2x2_LD-1--1-C2x2_RD
    bond31 = contract(left_half_31, right_half, ([0, 1], [0, 1])).to(env.device)
    bond23 = contract(left_half_23, right_half, ([0, 1], [0, 1])).to(env.device)
    return (bond31, bond23)