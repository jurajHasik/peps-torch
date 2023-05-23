import logging
import warnings
from tn_interface_abelian import contract, permute, conj
from ctm.generic_abelian.ctm_components import enlarged_corner

log= logging.getLogger('peps.ctm.generic_abelian.rdm')

def _cast_to_real(t, fail_on_check=False, warn_on_check=True, imag_eps=1.0e-8,\
    who="unknown", **kwargs):
    if t.is_complex():
        _t= t.item()
        if abs(_t.imag)/(abs(_t.real)+1.0e-8) > imag_eps:
            if warn_on_check:
                log.warning(f"Unexpected imaginary part "+who+" "+str(t))
            if fail_on_check: 
                raise RuntimeError("Unexpected imaginary part "+who+" "+str(t))
        return t.real()
    return t

def _sym_pos_def_matrix(rdm, sym_pos_def=False, verbosity=0, who="unknown", **kwargs):
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
    norm= _cast_to_real(rdm.trace(),who=who,**kwargs).to_number()
    rdm = rdm / norm
    return rdm

def _sym_pos_def_rdm(rdm, sym_pos_def=False, verbosity=0, who=None, **kwargs):
    assert rdm.ndim%2==0, "invalid rank of RDM"
    nsites= rdm.ndim//2
    # print(f"{tuple(nsites+i for i in range(nsites))} {tuple(i for i in range(nsites))}")
    rdm= rdm.fuse_legs(axes=(tuple(i for i in range(nsites)),\
        tuple(nsites+i for i in range(nsites))))
    rdm= _sym_pos_def_matrix(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who,**kwargs)
    rdm= rdm.unfuse_legs(axes=(0,1))
    return rdm

def _validate_precomputed(state,env):
    if state.build_dl:
        assert state.sites_dl!=None,"state's member sites_dl is not initialized"
        assert len(state.sites_dl)==len(state.sites),\
            "Inconsistent state.sites and state.sites_dl"

        requires_grad_state= any([ t.requires_grad for t in state.sites.values() ])
        requires_grad_env= any([ t.requires_grad for t in env.C.values()]) \
            or any([ t.requires_grad for t in env.T.values()])
        requires_grad_dl= any( [t.requires_grad for t in state.sites_dl.values()] )
        
        if requires_grad_state and not requires_grad_dl:
            warnings.warn("state members sites and sites_dl have different requires_grad", Warning)

    if state.build_dl_open:
        assert state.sites_dl_open!=None,"state's member sites_dl_open is not initialized"
        assert len(state.sites_dl_open)==len(state.sites),\
            "Inconsistent state.sites and state.sites_dl_open"

        requires_grad_state= any([ t.requires_grad for t in state.sites.values() ])
        requires_grad_env= any([ t.requires_grad for t in env.C.values()]) \
            or any([ t.requires_grad for t in env.T.values()])
        requires_grad_dl= any( [t.requires_grad for t in state.sites_dl_open.values()] )
        
        if requires_grad_state and not requires_grad_dl:
            warnings.warn("state members sites and sites_dl_open have different requires_grad", Warning)

    return True


# CONVENTION:
#
# when grouping indices, environment index always preceeds aux-indices of 
# double-layer on-site tensor

# ----- COMPONENTS ------------------------------------------------------------
def open_C2x2_LU(coord, state, env, fusion_level="full", verbosity=0):
    assert fusion_level in ["full","basic"],"Unsupported fusion_level option "+fusion_level
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
    
    if state.build_dl_open and not state.sites_dl_open is None:
        # C----------T1--3->1
        # |          2
        # |          1
        # T2----1 2--A--4
        # |          |\0->2
        # |          3
        # 0 
        #                
        A= state.site_dl_open(r)
        c2x2= contract(c2x2, A, ([1,2],[2,1]))
        # C----T--2
        # |    |
        # T---a*a--3
        # |    |\4
        # 0    1
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,3),(1,4),2) )
        elif fusion_level=="basic":
            c2x2= c2x2.transpose( axes=(0,3,1,4,2) )
    else:
        a= state.site(r)
        # C------T1--3->5
        # |      2->3,4
        # |
        # T2--1->1,2
        # 0
        c2x2= c2x2.unfuse_legs(axes=(1,2))
        # C--------T1--5->3
        # |        |\
        # |        3 4->2
        # |        1
        # T2--1 2--a--4->6
        # |\--2    3\0->4
        # 0   ->1  ->5
        c2x2= contract(c2x2, a, ([1,3],[2,1]))
        # C--------T1--3->1
        # |        |\
        # |   2<-4 | 2
        # |       \| 1
        # T2-------a-----6->4
        # |\--1 2----a*--4->7
        # 0        5 |\0->5
        #        ->3 3->6
        c2x2= contract(c2x2, a, ([1,2],[2,1]), conj=(0,1))
        c2x2= c2x2.fuse_legs(axes=(0,(3,6),1,(4,7),(2,5)))
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs(axes=((0,1),(2,3),4))
        elif fusion_level=="basic":
            pass

    return c2x2

def open_C2x2_LD(coord, state, env, fusion_level="full", verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param fusion_level: controls fusion of indices of open enlarged corner
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type fusion_level: str
    :type verbosity: int
    :return: left-down enlarged corner with open physical indices
    :rtype: yastn.tensor

    Computes lower-down enlarged corner centered on vertex ``coord`` by contracting 
    the following tensor network::

              s,s'
        |  | /
        T--a^+a--
        |  |
        C--T-----

    The physical indices `s` and `s'` of on-site tensor :math:`a` at vertex ``coord`` 
    and its hermitian conjugate :math:`a^\dagger` are left uncontracted

    Depending on `fusion_level`, the resulting tensor is::

        rank-3 : fusion_level= 'full'             rank-5 : fusion_level= 'basic'

          0
         /\   2 (s,s')                            0  1  4 (s,s')
        |  | /                                    |  | /
        T--a^+a--\                                T--a^+a--3
        |  |      >--1                            |  |     
        C--T-----/                                C--T-----2
 
    """
    assert fusion_level in ["full","basic"],"Unsupported fusion_level option "+fusion_level
    r= state.vertexToSite(coord)
    # 0
    # |
    # T(-1,0)--2->1
    # 1
    # 0
    # C(-1,1)--1->2
    c2x2 = contract(env.T[(r,(-1,0))],env.C[(r,(-1,1))],([1],[0]))
    if verbosity>0: print("c2x2=TC "+str(c2x2))
    
    # 0
    # |
    # T(-1,0)--1
    # |             0->2
    # C(-1,1)--2 1--T(0,1)--2->3
    c2x2 = contract(c2x2,env.T[(r,(0,1))],([2],[1]))
    if verbosity>0: print("c2x2=TCT "+str(c2x2))
    
    if state.build_dl_open and not state.sites_dl_open is None:
        # 0            0->2
        # |       3<-1/
        # T-----1 2--A--4
        # |          3
        # |          2
        # C----------T--3->1
        # 
        A= state.site_dl_open(r)
        c2x2= contract(c2x2, A, ([1,2],[2,3]))
        # 0    1  4
        # |    | /
        # T----A---3
        # |    |
        # C----T---2
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,3),(1,4),2) )
        elif fusion_level=="basic":
            c2x2= c2x2.transpose( axes=(0,3,1,4,2) )
    else:
        a= state.site(r)
        # 0
        # |
        # T(-1,0)--1->1,2
        # |             2->3,4
        # C(-1,1)--2 1--T(0,1)--3->5
        c2x2= c2x2.unfuse_legs(axes=(1,2))
        # 0   ->1  ->5 0->4
        # |/----2    1/
        # T-----1 2--a--4->6
        # |          3 4->2
        # |          3/
        # C----------T--5->3
        c2x2= contract(c2x2, a, ([1,3],[2,3]))
        # 0            1->6
        # |       3<-5 | 0->5
        # |          | |/ 
        # |/----1 2----a*----4->7
        # T----------a----6->4
        # |     2<-4/| 3
        # |          | 2 
        # |          |/
        # C----------T--3->1
        c2x2= contract(c2x2, a, ([1,2],[2,3]), conj=(0,1))
        c2x2= c2x2.fuse_legs(axes=(0,(3,6),1,(4,7),(2,5)))
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,1),(2,3),4) )
        elif fusion_level=="basic":
            pass

    if verbosity>0: print("c2x2=TCTa*a "+str(c2x2))

    return c2x2

def open_C2x2_RU(coord, state, env, fusion_level="full", verbosity=0):
    assert fusion_level in ["full","basic"],"Unsupported fusion_level option "+fusion_level
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
    #          3<-2
    c2x2 =contract(T2, c2x2, ([2],[0]))

    if state.build_dl_open and not state.sites_dl_open is None:
        #     0--T2------C
        #        1       |
        # 2<-0--\1       | 
        #  3<-2--a--4 2--T1
        #     4<-3    1<-3
        # 
        A= state.site_dl_open(r)
        c2x2= contract(c2x2,A,([1,2],[1,4]))
        #  0--T2----C
        #     |     |
        # 1--a*a----T1
        #    /|     |
        #   4 3     2
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,3),(1,4),2) )
        elif fusion_level=="basic":
            c2x2= c2x2.transpose( axes=(0,3,1,4,2) )
    else:
        a= state.site(r)
        #    0--T2--2 0--C
        #  1,2<-1        |
        #        3,4<-2--T1
        #             5<-3
        c2x2= c2x2.unfuse_legs(axes=(1,2))
        #    0--T2-------C
        #      /|        |
        #  1<-2 1 0->4   |
        #       1/       |
        # 5<-2--a--4 3---T1
        #    6<-3    4--/|
        #          ->2   5->3
        c2x2= contract(c2x2,a,([1,3],[1,4]))
        #      0--T2-------C
        #        /|        |
        #       1 | 4->2   |
        #       1 |/       |
        # 3<-5----a--------T1
        # 6<-2--a*|--4 2--/|
        #  5<-0/| |        |
        #    7<-3 |        |
        #      4<-6     1<-3
        c2x2= contract(c2x2,a,([1,2],[1,4]), conj=(0,1))
        c2x2= c2x2.fuse_legs(axes=(0,(3,6),1,(4,7),(2,5)))
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,1),(2,3),4) )
        elif fusion_level=="basic":
            pass
    
    return c2x2

def open_C2x2_RD(coord, state, env, fusion_level="full", verbosity=0):
    assert fusion_level in ["full","basic"],"Unsupported fusion_level option "+fusion_level
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

    if state.build_dl_open and not state.sites_dl_open is None:
        #    3<-1          0 
        # 4<-2--A--4 1-----T2
        #  2<-0/3          | 
        #       2          |
        # 1<-3--T1---------C
        #                   
        A= state.site_dl_open(r)
        c2x2= contract(c2x2,A,([1,2],[4,3]))
        #       4 1    0
        #        \|    |
        #     3---A----T2
        #         |    |
        #      2--T1---C
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,3),(1,4),2) )
        elif fusion_level=="basic":
            c2x2= c2x2.transpose( axes=(0,3,1,4,2) )
    else:
        a= state.site(r)
        #            0
        #    1,2<-1--T2
        #            |
        #  3,4<-2    |
        # 5<-3--T1---C
        c2x2= c2x2.unfuse_legs(axes=(1,2))
        #     5<-  1<-   0
        #       1    2--\|
        # 6<-2--a--4 1---T2
        #       3\0->4   |
        #  2<-4 3        |
        #      \|        |
        # 3<-5--T1-------C
        c2x2= contract(c2x2,a,([1,3],[4,3]))
        #     6<-1 5->3
        #   5<-0\| |        0
        #  7<-2--a*|--4 1--\|
        #  4<-6----a--------T2
        #        3 |\4->2   |
        #        2 |        |
        #         \|        |
        #    1<-3--T1-------C
        c2x2= contract(c2x2,a,([1,2],[4,3]),conj=(0,1))
        c2x2= c2x2.fuse_legs(axes=(0,(3,6),1,(4,7),(2,5)))
        if fusion_level=="full":
            c2x2= c2x2.fuse_legs( axes=((0,1),(2,3),4) )
        elif fusion_level=="basic":
            pass
    
    return c2x2

# ----- 1-site RDM ------------------------------------------------------------
def rdm1x1(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) for which reduced density matrix is constructed
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int) 
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type verbosity: int
    :return: 1-site reduced density matrix with indices :math:`s;s'`
    :rtype: torch.tensor

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
    assert _validate_precomputed(state,env),"Inconsistent requires_grad for state and/or env tensors"
    r= state.vertexToSite(coord)
    rdm= open_C2x2_LD(r, state, env, verbosity=verbosity)

    # C(-1,-1)--1 0--T(0,-1)--2  => C---T--2->1(-1)
    # 0              1               \ /
    #                                 0(-1)
    C2x1_LU= contract(env.C[(r,(-1,-1))], env.T[(r,(0,-1))],([1],[0]))
    C2x1_LU= C2x1_LU.fuse_legs( axes=((0,1),2) ) 

    # C2x1_LU--1->0
    # |
    # 0     2
    # 0__ _/
    # |    |
    # |rdm_|--1 <-NOTE: contains both env index and double layer aux-indices)
    rdm= contract(C2x1_LU, rdm, ([0],[0]))

    if verbosity>0:
        print("rdm=CTCTaT "+str(rdm))

    #    1<-0       =>      
    # 2<-1--T(1,0)            1
    #       2              2--T(1,0)
    #       0          0--<   |
    # 0<-1--C(1,1)         0--C(1,1)
    E= contract(env.C[(r,(1,1))], env.T[(r,(1,0))], ([0],[2]))
    E= E.fuse_legs( axes=((0,2),1) )

    #    0--C(1,-1) =>          0--C
    #       1                      |
    #       1                   1--E
    # 1<-0--E
    E= contract(env.C[(r,(1,-1))], E, ([1],[1]))

    if verbosity>0:
        print("rdm=CTC "+str(E))

    # C(-1,-1)--T(0,-1)---------0 0-----C(1,-1)
    # |         |                       |
    # |         |/2->0                  |
    # T(-1,0)---a------------\       /--T(1,0) 
    # |         |            |--1 1--|  | 
    # |         |            |       |  |
    # C(-1,1)---T(0,1)-------/       \--C(1,1)
    rdm = contract(rdm,E,([0,1],[0,1]))

    # unfuse physical indices into ket,bra: 0 -> s0,s0'
    rdm= rdm.unfuse_legs(axes=0)
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
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type verbosity: int
    :return: 2-site reduced density matrix with indices :math:`s_0s_1;s'_0s'_1`
    :rtype: yastn.tensor

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
    assert _validate_precomputed(state,env),"Inconsistent requires_grad for state and/or env tensors"
    #----- building C2x2_LU ----------------------------------------------------
    C2x2_LU= open_C2x2_LU(coord, state, env, verbosity=verbosity)

    if verbosity>0:
        print(f"C2X2 LU {coord} -> f{state.vertexToSite(coord)} (-1,-1): {C2x2_LU}")

    #----- building C2x1_LD ----------------------------------------------------
    C = env.C[(state.vertexToSite(coord),(-1,1))]
    T2 = env.T[(state.vertexToSite(coord),(0,1))]

    #                    0(+1)
    # 0       0->1      / \
    # C--1 1--T2--2 => C---T--2->1
    C2x1_LD= contract(C, T2, ([1],[1]))
    C2x1_LD= C2x1_LD.fuse_legs(axes=((0,1),2))

    if verbosity>0:
        print(f"C2X1 LD {coord} -> {state.vertexToSite(coord)} (-1,1): {C2x1_LD}")

    #----- build left part C2x2_LU--C2x1_LD ------------------------------------
    # C2x2_LU--1 
    # |\2
    # 0
    # 0
    # C2x1_LD--1->0
    left_half= contract(C2x1_LD, C2x2_LU, ([0],[0]))

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RU= open_C2x2_RU(shift_r, state, env, verbosity=verbosity)
    
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
    C2x1_RD= C2x1_RD.fuse_legs(axes=((0,1),2))

    if verbosity>0:
        print(f"C2X1 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shitf_coord} "\
            +f"(1,1): {C2x1_RD}")

    #----- build right part C2x2_RU--C2x1_RD -----------------------------------
    # 1<-0--C2x2_RU
    #       |\2
    #       1
    #       0
    # 0<-1--C2x1_RD
    right_half =contract(C2x1_RD, C2x2_RU, ([0],[1]))

    # construct reduced density matrix by contracting left and right halfs
    # C2x2_LU--1 1----C2x2_RU
    # |\2->0          |\2->1
    # |               |    
    # C2x1_LD--0 0----C2x1_RD
    rdm =contract(left_half,right_half,([0,1],[0,1]))

    # unfuse physical indices into ket,bra: 01 -> s0,s0',s1,s1'
    # permute into order of s0,s1;s0',s1'
    # 0123->0213
    rdm= rdm.unfuse_legs(axes=(0,1))
    rdm= permute(rdm, (0,2,1,3))
    # symmetrize and normalize
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
    :rtype: yastn.tensor

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
    assert _validate_precomputed(state,env),"Inconsistent requires_grad for state and/or env tensors"
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
    C1x2_RU= C1x2_RU.fuse_legs(axes=((0,1),2))

    if verbosity>0:
        print(f"C1X2 RU {coord} -> {state.vertexToSite(coord)} (1,-1): {C1x2_RU}")

    #----- build upper part C2x2_LU--C1x2_RU -----------------------------------
    # C2x2_LU--1 0--C1x2_RU
    # |\2           |
    # 0->1          1->0
    upper_half =contract(C1x2_RU, C2x2_LU, ([0],[1]))

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_LD= open_C2x2_LD(shift_r, state, env, verbosity=verbosity)

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
    C1x2_RD= C1x2_RD.fuse_legs(axes=(0,(2,1)))

    if verbosity>0:
        print(f"C1X2 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(1,1): {C1x2_RD}")

    #----- build lower part C2x2_LD--C1x2_RD -----------------------------------
    # 0->1(+1)      0
    # |/2           |
    # C2x2_LD--1 1--C1x2_RD
    lower_half =contract(C1x2_RD, C2x2_LD, ([1],[1]))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C1x2_RU
    # |\2->0       |
    # 1            0
    # 1            0
    # |/2->1       |
    # C2x2_LD------C1x2_RD
    rdm =contract(upper_half,lower_half,([0,1],[0,1]))

    # unfuse physical indices into ket,bra: 01 -> s0,s0',s1,s1'
    # permute into order of s0,s1;s0',s1'
    # 0123->0213
    rdm= rdm.unfuse_legs(axes=(0,1))
    rdm= permute(rdm, (0,2,1,3))
    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity,\
        who=who)

    return rdm

def rdm2x2_NNN_1n1(coord, state, env, sym_pos_def=False, verbosity=0):
    r"""
    :param coord: vertex (x,y) specifies upper left site of 2x2 subsystem
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
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
    who = "rdm2x2_NNN_1n1"
    assert _validate_precomputed(state,env),"Inconsistent requires_grad for state and/or env tensors"
    # ----- building C2X2_LU ----------------------------------------------------
    vec = (0, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_LU= enlarged_corner(shift_coord,state,env,'LU',verbosity=verbosity)

    # ----- building C2x2_RU ----------------------------------------------------
    vec = (1, -1)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RU= open_C2x2_RU(shift_coord, state, env, verbosity=verbosity)

    # ----- build upper part C2x2_LU--C2X2_RU -----------------------------------
    # C2x2_LU--1 0--C2X2_RU
    # |             |\2
    # 0             1
    # TODO is it worthy(performance-wise) to instead overwrite one of C2x2_LU,C2X2_RU ?
    upper_half = contract(C2X2_LU, C2X2_RU, ([1], [0]))

    # ----- building C2X2_RD ----------------------------------------------------
    vec = (1, 0)
    shift_coord = state.vertexToSite((coord[0] + vec[0], coord[1] + vec[1]))
    C2X2_RD= enlarged_corner(shift_coord,state,env,'RD',verbosity=verbosity)    

    # ----- building C2X2_LD ----------------------------------------------------
    C2X2_LD= open_C2x2_LD(coord,state,env,verbosity=verbosity)

    # ----- build lower part C2X2_LD--C2X2_RD -----------------------------------
    # 0             0->2                 0            2->1
    # |/2->1        |          & permute |/1->2       |
    # C2X2_LD--1 1--C2X2_RD              C2X2_LD------C2X2_RD
    # TODO is it worthy(performance-wise) to instead overwrite one of C2X2_LD,C2X2_RD ?
    lower_half = contract(C2X2_LD, C2X2_RD, ([1], [1]))
    lower_half = permute(lower_half, (0, 2, 1))

    # construct reduced density matrix by contracting lower and upper halfs
    # C2X2_LU------C2X2_RU
    # |            |\2->1
    # 0            1
    # 0            1
    # |/2->0       |
    # C2X2_LD------C2X2_RD
    rdm = contract(lower_half, upper_half, ([0, 1], [0, 1]))

    # permute into order of s0,s1;s0',s1' where primed indices
    # represent "ket"
    # 0123->0213
    # symmetrize and normalize
    rdm= rdm.unfuse_legs(axes=(0,1))
    rdm= permute(rdm, (0,2,1,3))
    
    rdm = _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)
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
    :rtype: yastn.tensor

    Computes 4-site reduced density matrix :math:`\rho_{2x2}` of 2x2 subsystem specified
    by the vertex ``coord`` of its upper left corner using strategy:

        1. compute four individual corners
        2. construct upper and lower half of the network
        3. contract upper and lower half to obtain final reduced density matrix

    ::

        C--T------------------T------------------C = C2x2_LU(coord)--------C2x2(coord+(1,0))
        |  |                  |                  |   |                     |
        T--a^+a(coord)--------a^+a(coord+(1,0))--T   C2x2_LD(coord+(0,1))--C2x2(coord+(1,1))
        |  |                  |                  |
        T--a^+a(coord+(0,1))--a^+a(coord+(1,1))--T
        |  |                  |                  |
        C--T------------------T------------------C
        
    The physical indices `s` and `s'` of on-sites tensors :math:`A` (and :math:`A^\dagger`) 
    at vertices ``coord``, ``coord+(1,0)``, ``coord+(0,1)``, and ``coord+(1,1)`` are 
    left uncontracted and given in the same order::
        
        s0 s1
        s2 s3

    """
    who= "rdm2x2"
    assert _validate_precomputed(state,env),"Inconsistent requires_grad for state and/or env tensors"
    #----- building C2x2_LU ----------------------------------------------------
    C2x2_LU= open_C2x2_LU(coord, state, env, verbosity=verbosity)

    if verbosity>0:
        print(f"C2X2 LU {coord} -> {state.vertexToSite(coord)} (-1,-1):\n{C2x2_LU}")

    #----- building C2x2_RU ----------------------------------------------------
    vec = (1,0)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RU= open_C2x2_RU(shift_r, state, env, verbosity=verbosity)
    
    if verbosity>0:
        print(f"C2X2 RU {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            f"(1,-1):\n{C2x2_RU}")

    #----- build upper part C2x2_LU--C2x2_RU -----------------------------------
    # C2x2_LU--1 0--C2x2_RU
    # |\2->1        |\2->3
    # 0             1->2 
    upper_half = contract(C2x2_LU, C2x2_RU, ([1],[0]))

    #----- building C2x2_RD ----------------------------------------------------
    vec = (1,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_RD= open_C2x2_RD(shift_r, state, env, verbosity=verbosity)

    #    0
    #    |/2
    # 1--C2x2
    if verbosity>0:
        print(f"C2X2 RD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(1,1):\n{C2x2_RD}")

    #----- building C2x2_LD ----------------------------------------------------
    vec = (0,1)
    shift_r = state.vertexToSite((coord[0]+vec[0],coord[1]+vec[1]))
    C2x2_LD= open_C2x2_LD(shift_r, state, env, verbosity=verbosity)

    if verbosity>0:
        print(f"C2X2 LD {(coord[0]+vec[0],coord[1]+vec[1])} -> {shift_r} "\
            +f"(-1,1): {C2x2_LD}")

    #----- build lower part C2x2_LD--C2x2_RD -----------------------------------
    # 0             0->2
    # |/2->1        |/2->3
    # C2x2_LD--1 1--C2x2_RD
    lower_half = contract(C2x2_LD, C2x2_RD, ([1],[1]))
    
    # construct reduced density matrix by contracting lower and upper halfs
    # C2x2_LU------C2x2_RU
    # | \1->0      | \3->1
    # 0            2
    # 0            2  
    # | /1->2      | /3
    # C2x2_LD------C2x2_RD
    rdm = contract(upper_half,lower_half,([0,2],[0,2]))

    # unfuse physical indices into ket,bra: 0123 -> s0,s0',s1,s1',s2,s2',s3,s3'
    # permute into order of s0,s1,s2,s3;s0',s1',s2',s3'
    # 01234567->02461357
    rdm= rdm.unfuse_legs(axes=(0,1,2,3))
    rdm= permute(rdm, (0,2,4,6,1,3,5,7))
    # symmetrize and normalize
    rdm= _sym_pos_def_rdm(rdm, sym_pos_def=sym_pos_def, verbosity=verbosity, who=who)

    return rdm