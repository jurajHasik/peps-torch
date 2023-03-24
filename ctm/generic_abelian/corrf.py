import numpy as np
from yast.yast import einsum as einsum
from ipeps.ipeps_abelian import _fused_dl_site
from tn_interface_abelian import contract, permute

def get_edge(coord, direction, state, env, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction of the edge
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type verbosity: int
    :return: tensor with indices :math:`\chi \times D^2 \times \chi`
    :rtype: yast.Tensor

    Build an edge of site ``coord`` by contracting one of the following networks
    depending on the chosen ``direction``::

            up=(0,-1)   left=(-1,0)  down=(0,1)   right=(1,0)

                                     (-)(-)(-)
                         C--0(+)      0  1  2      (-)0--C
                         |            |  |  |            |
        E =  C--T--C     T--1(+)      C--T--C      (-)1--T  
             |  |  |     |                               |
             0  1  2     C--2(+)                   (-)2--C
            (+)(+)(+)

    The indices of the resulting tensor are always ordered from 
    left-to-right and from up-to-down.
    """
    c = state.vertexToSite(coord)
    if direction==(0,-1): #up
        C1= env.C[(c,(1,-1))]
        T= env.T[(c,direction)]
        # 0--T--2 0--C1
        #    1       1->2
        E = contract(T,C1,([2],[0]))
        # C2--1 0--T--C1
        # 0        1  2 
        C2= env.C[(c,(-1,-1))]
        E= contract(C2,E,([1],[0]))
    elif direction==(-1,0): #left
        C1= env.C[(c,(-1,-1))]
        T= env.T[(c,direction)]
        # C1--1->0
        # 0
        # 0
        # T--2
        # 1
        E = contract(C1,T,([0],[0]))
        # C1--0
        # |
        # T--2->1
        # 1
        # 0
        # C2--1->2
        C2= env.C[(c,(-1,1))]
        E= contract(E,C2,([1],[0]))
    elif direction==(0,1): #down
        C1= env.C[(c,(-1,1))]
        T= env.T[(c,direction)]
        # 0        0->1
        # C1--1 1--T--2
        E = contract(C1,T,([1],[1]))
        # 0   1       0->2
        # C1--T--2 1--C2 
        C2= env.C[(c,(1,1))]
        E= contract(E,C2,([2],[1]))
    elif direction==(1,0): #right
        C1= env.C[(c,(1,1))]
        T= env.T[(c,direction)]
        #       0 
        #    1--T
        #       2
        #       0
        # 2<-1--C1
        E = contract(T,C1,([2],[0]))
        #    0--C2
        #       1
        #       0
        #    1--T
        #       |
        #    2--C1
        C2= env.C[(c,(1,-1))]
        E= contract(C2,E,([1],[0]))
    else:
        raise ValueError("Invalid direction: "+str(direction))
    
    return E

def apply_edge(coord, direction, state, env, vec, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction of the edge
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times (D^2)^l \times \chi`
                representing an edge of length l
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type vec: yast.Tensor
    :type verbosity: int
    :return: scalar resulting from the contraction of ``vec`` with an edge
             built from the environment
    :rtype: yast.Tensor
    
    Contracts ``vec`` tensor with the edge of length l defined by ``coord`` site 
    and a chosen ``direction``. Afterwards, their dot product is computed::

                            get_edge(coord,direction,...)
                   ------0 0------C
                  |               |     
        scalar = vec-----1 1------T
                  |               |
                 ...--         --... 
                  |               |
                   --(V+1) (V+1)--C
    """
    if vec.ndim==3:
        E= get_edge(coord, direction, state, env, verbosity=verbosity)
    elif vec.ndim==4:
        E= get_edge_2(coord, direction, state, env, verbosity=verbosity)
    else:
        ValueError("Unsupported edge: "+vec.ndim)

    inds=list(range(vec.ndim))
    S = contract(vec,E,(inds,inds))

    return S

def apply_TM_1sO(coord, direction, state, env, edge, op=None, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction in which the transfer operator is applied
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi (\times d_{MPO})`,
                 potentially with 4th index beloning to some MPO
    :param op: operator to be inserted into transfer matrix
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied 
             The resulting tensor has an identical index structure as the 
             original ``edge`` 
    :rtype: torch.tensor
    
    Applies a single instance of the transfer matrix of site r=(x,y) to 
    the ``edge`` tensor by contracting the following network, or its corresponding 
    rotation depending on the ``direction``::

        direction:  right=(1,0)                down=(0,1)

                 -----T----------------      --------edge--------   
                |     |                     |         |          |\
               edge--(a(r)^+ op a(r))--     T--(a(r)^+ op a(r))--T \
                |     |                     |         |          | (MPO)
                 -----T----------------
                (\-----------------MPO)

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``op`` (if supplied). Potentially,
    the ``edge`` can carry an extra MPO index. 
    """
    # Check if edge has MPO index
    contains_mpo= edge.ndim==4

    # Four basic cases of passed op
    def get_aXa(c, op):
        # c - coord
        # op - operator
        a= state.site(c)
        dims_op= None if op is None else op.ndim
        if op is None:
            # identity
            if not (state.sites_dl is None):
                A= state.site_dl(c)
            else:
                A= _fused_dl_site(a)
        elif dims_op==2:
            # one-site operator
            A= _fused_dl_site(einsum('mefgh,nm->nefgh',a,op),a.conj())
        elif len(dims_op)==3:
            # edge operators of some MPO
            #
            # 0                   0
            # |                   |
            # op--2 ... or ... 2--op
            # |                   |
            # 1                   1
            #
            # assume the last index of the op is the MPO dimension.
            # It will become the last index of the resulting tensor 
            A= einsum('nefghl,nabcd->eafbgchdl', einsum('mefgh,nml->nefghl',a,op), a.conj())
            A= A.fuse_legs( axes=((0,4),(1,5),(2,6),(3,7), 8) )
        elif len(dims_op)==4:
            # intermediate op of some MPO
            #
            #        0
            #        |
            # ... 2--op--3 ...
            #        |
            #        1
            #
            # assume the index 2 is to be contracted with extra (MPO) index
            # of edge. The remaining index 3 becomes last index resulting edge
            A= einsum('nefghlk,nabcd->eafbgchdlk', einsum('mefgh,nmlk->nefghlk',a,op), a.conj())
            A= A.fuse_legs( axes=((0,4),(1,5),(2,6),(3,7), 8,9) )
        return A

    c = state.vertexToSite(coord)
    if direction == (0,-1): #up
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        # 0
        # T1--2->1
        # 1
        # 0  1->2 2->3 (3->4)
        # --edge-- -----
        E = contract(T1,edge,([1],[0]))

        # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
        #        Possibly reshape indices 1,2 of E, which are to be contracted with 
        #        on-site tensor and contract bra,ket in two steps instead of creating 
        #        double layer tensor
        #    /
        # --A--
        #  /|s
        #   X   
        # s'|/
        # --A--
        #  /
        #
        # where X is Id or op
        A= get_aXa(c, op)

        # edge with no MPO              edge with MPO index
        #
        # 0     2<-0 (4)                0     2<-0 ---------(4)(5->4)
        # |        |/                   |        |/          |
        # T1--1 1--A--3                 T1--1 1--A--3        |
        # |        2                    |        2           | 
        # |        2    3->1            |        2    3->1  (4)
        # --------edge--                --------edge-- ------
        E = contract(E,A,([1,2,4],[1,2,4])) if contains_mpo and A.ndim>4 else \
            contract(E,A,([1,2],[1,2]))

        # case of extra index only on mpo
        #
        # 0     3<-0
        # |        |   
        # T1--1 1--A--3->4 
        # |        2     
        # |        2    3->1,4->2
        # --------edge--  
        if contains_mpo and A.ndim==4: E= permute(E,(0,1,3,4,2))

        # 0  1<-2 (4->2)    0->2 (0->3)
        # |     |/          |
        # T1----A--3 1------T2
        # |     |           2
        # |     |           1
        # -----edge---------
        T2 = env.T[(c,(1,0))]
        E = contract(E,T2,([1,3],[2,1]))
        # if we have extra MPO dimension, permute it to the last index
        if E.ndim==4: E= permute(E,(0,1,3,2))
    elif direction == (-1,0): #left
        T1 = env.T[(c,(0,-1))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #       0 -- 
        #       1 --| edge 
        #       2 -- 
        #
        #   0--T1--2 0---- 
        #      1          |
        #         2<-1--edge 
        #                 |
        #         3<-2----|
        #        (4<-3)--- 
        E = contract(T1,edge,([2],[0]))

        # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
        #        Possibly reshape indices 1,2 of E, which are to be contracted with 
        #        on-site tensor and contract bra,ket in two steps instead of creating 
        #        double layer tensor
        #    /
        # --A--
        #  /|s
        #   X   
        # s'|/
        # --A--
        #  /
        #
        # where X is Id or op
        A= get_aXa(c, op)

        #        0--T1---------              0--T1---------
        #           1          |                1          | 
        #           0          |                0          |
        #     2<-1--A----3 2--edge        2<-1--A----3 2--edge
        #         / 2->3       |              / 2->3       |
        #        |       1<-3--|             (4)     1<-3--
        # (4<-5)(4)-------(4)--
        E = contract(E,A,([1,2,4],[0,3,4])) if contains_mpo and A.ndim>4 else \
            contract(E,A,([1,2],[0,3]))
        if contains_mpo and A.ndim==4: E= permute(E,(0,1,3,4,2))

        #           0--T1-------
        #              |        |
        #              |        |
        #        1<-2--A-------edge
        #            / 3        |
        #       (2<-4) 0        |
        # (3<-1) 2<-1--T2--2 1--
        T2 = env.T[(c,(0,1))]
        E = contract(E,T2,([1,3],[2,0]))
        if E.ndim==4: E= permute(E,(0,1,3,2))
    elif direction == (0,1): #down
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #  --edge-- ------
        # 0   1->2 2->3  (3->4)
        # 0
        # T1--2->1
        # 1->0
        E = contract(T1,edge,([0],[0]))

        # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
        #        Possibly reshape indices 1,2 of E, which are to be contracted with 
        #        on-site tensor and contract bra,ket in two steps instead of creating 
        #        double layer tensor
        #    /
        # --A--
        #  /|s
        #   X   
        # s'|/
        # --A--
        #  /
        #
        # where X is Id or op
        A= get_aXa(c, op)

        #  -------edge----         -------edge---- ------
        # |        2      3->1    |        2      3->1  (4)
        # |        0              |        0             |
        # T1--1 1--A--3           T1--1 1--A--3          |
        # |        |\             |        |\            |
        # 0        2 (4)          0        2 -----------(4)(5->4) 
        E = contract(E,A,([1,2,4],[1,0,4])) if contains_mpo and A.ndim>4 else \
            contract(E,A,([1,2],[1,0]))
        if contains_mpo and A.ndim==4: E= permute(E,(0,1,3,4,2))

        #  ----edge-------
        # |     |         1
        # |     |         0
        # T1----A--3 1----T2
        # |     |\        |
        # 0  1<-2 (4->2)  2 (2->3)
        T2 = env.T[(c,(1,0))]
        E = contract(E,T2,([1,3],[0,1]))
        if E.ndim==4: E= permute(E,(0,1,3,2))
    elif direction == (1,0): #right
        T1 = env.T[(c,(0,-1))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #       -- 0
        # edge |-- 1
        #       -- 2
        #
        #   ----0 0--T1--2->1
        #  |         1->0
        # edge--1->2
        #  |
        #   ----2->3
        #  |
        #   ----(3->4)
        E = contract(T1,edge,([0],[0]))

        # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
        #        Possibly reshape indices 1,2 of E, which are to be contracted with 
        #        on-site tensor and contract bra,ket in two steps instead of creating 
        #        double layer tensor
        #    /
        # --A--
        #  /|s
        #   X   
        # s'|/
        # --A--
        #  /
        #
        # where X is Id or op
        A= get_aXa(c, op)

        #   ---------T1--1->0       ---------T1--1->0
        #  |         0             |         0
        #  |         0             |         0
        # edge--2 1--A--3         edge--2 1--A--3
        #  |         2 \           |         2
        #   ----3->1    |           ----3->1
        #  |            |
        #   ----4 ---- (4)(5->4)
        E = contract(E,A,([0,2,4],[0,1,4])) if contains_mpo and A.ndim>4 else \
            contract(E,A,([0,2],[0,1]))
        if contains_mpo and A.ndim==4: E= permute(E,(0,1,3,4,2))

        #   -------T1--0
        #  |       |
        #  |       |
        # edge-----A--3->1
        #  |       2 \(4->2)
        #  |       0
        #   --1 1--T2--2 (2->3)
        T2 = env.T[(c,(0,1))]
        E = contract(E,T2,([1,2],[1,0]))
        if E.ndim==4: E= permute(E,(0,1,3,2))     
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return E

def corrf_1sO1sO(coord, direction, state, env, op1, get_op2, dist, rl_0=None, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: orientation of correlation function
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op1: first one-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    one-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS_ABELIAN
    :type env: ENV_ABELIAN
    :type op1: yast.tensor
    :type get_op2: function(int)->yast.tensor
    :type dist: int
    :type rl_0: tuple(function(tuple(int,int))->yast.Tensor, function(tuple(int,int))->yast.Tensor)
    :type verbosity: int
    :return: vector ``corrf`` of length ``dist`` holding the values of 
             correlation function :math:`\langle O_1(0) O_2(r) \rangle` 
             for :math:`r \in [1,dist]`
    :rtype: yast.tensor
    
    Computes the two-point correlation function :math:`\langle O_1(0) O_2(r) \rangle`
    by contracting the following network::

        C-----T------- ... -----T------- ... ------T----------C
        |     |                 |                  |          |
        |    a(0)^+            a(i)^+             a(r)^+      |
        T--( op_1  )-- ... --(  |    )-- ... --( gen_op2(r))--T
        |    a(0))             a(i)               a(r)        |
        |     |                 |                  |          | 
        C-----T------- ... -----T------- ... ------T----------C

    for increasingly large distance ``r`` up to ``dist``. The ``op1`` is 
    applied at site 0=(x,y), the transfer matrices are applied in the 
    ``direction`` up to site :math:`r=(x,y) + \text{dist} \times \text{direction}`. 
    """
    def shift_c(c,d):
        return (c[0]+d[0],c[1]+d[1])

    c0= coord
    rev_d = (-direction[0],-direction[1]) # opposite direction
    E0 = get_edge(c0, rev_d, state, env, verbosity=verbosity) if rl_0 is None else rl_0[0](c0)
    # Apply transfer matrix with operator op1
    #
    #   -- 0     -- -----T--------- 0
    # E1-- 1 = E0-- --(A^+ op1 A)-- 1
    #   -- 2     -- -----T--------- 2
    E1 = apply_TM_1sO(c0, direction, state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_1sO(c0, direction, state, env, E0, verbosity=verbosity)

    corrf=np.empty(dist+1,dtype=state.dtype)
    for r in range(dist+1):
        # close the end of the network by appending final transfer matrix 
        # with op2
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        c0= shift_c(c0,direction)
        E12= apply_TM_1sO(c0, direction, state, env, E1, op=get_op2(r), verbosity=verbosity)
        # and corresponding normalization network
        E0 = apply_TM_1sO(c0, direction, state, env, E0, verbosity=verbosity)
        # and to network with only a op1 operator
        E1 = apply_TM_1sO(c0, direction, state, env, E1, verbosity=verbosity)

        # and finally apply edge in the direction of growth
        if rl_0 is None:
            E12= apply_edge(c0, direction, state, env, E12, verbosity=verbosity)
            E00= apply_edge(c0, direction, state, env, E0, verbosity=verbosity)
        else:
            E12= torch.tensordot(E12,rl_0[1](c0),([0,1,2],[0,1,2]))
            E00= torch.tensordot(E0,rl_0[1](c0),([0,1,2],[0,1,2]))
        corrf[r]= E12.to_number()/E00.to_number()

        # normalize by the largest element of E0
        max_elem_E0 = E0.norm(p='inf')
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf