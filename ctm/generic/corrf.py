import torch
from ipeps import IPEPS
from ctm.generic.env import ENV

def get_edge(coord, direction, state, env, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction of the edge
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type verbosity: int
    :return: tensor with indices :math:`\chi \times D^2 \times \chi`
    :rtype: torch.tensor

    Build an edge of site ``coord`` by contracting one of the following networks
    depending on the chosen ``direction``::

            up=(0,-1)   left=(-1,0)  down=(0,1)   right=(1,0)

                         C--0         0  1  2       0--C
                         |            |  |  |          |
        E =  C--T--C     T--1         C--T--C       1--T  
             |  |  |     |                             |
             0  1  2     C--2                       2--C

    The indices of the resulting tensor are always ordered from 
    left-to-right and from up-to-down.
    """
    c = state.vertexToSite(coord)
    if direction==(0,-1): #up
        C1= env.C[(c,(1,-1))]
        T= env.T[(c,direction)]
        # 0--T--2 0--C1
        #    1       1->2
        E = torch.tensordot(T,C1,([2],[0]))
        if verbosity>0: print("E=CT "+str(E.size()))
        # C2--1 0--T--C1
        # 0        1  2 
        C2= env.C[(c,(-1,-1))]
        E= torch.tensordot(C2,E,([1],[0]))
    elif direction==(-1,0): #left
        C1= env.C[(c,(-1,-1))]
        T= env.T[(c,direction)]
        # C1--1->0
        # 0
        # 0
        # T--2
        # 1
        E = torch.tensordot(C1,T,([0],[0]))
        if verbosity>0: print("E=CT "+str(E.size()))
        # C1--0
        # |
        # T--2->1
        # 1
        # 0
        # C2--1->2
        C2= env.C[(c,(-1,1))]
        E= torch.tensordot(E,C2,([1],[0]))
    elif direction==(0,1): #down
        C1= env.C[(c,(-1,1))]
        T= env.T[(c,direction)]
        # 0        0->1
        # C1--1 1--T--2
        E = torch.tensordot(C1,T,([1],[1]))
        if verbosity>0: print("E=CT "+str(E.size()))
        # 0   1       0->2
        # C1--T--2 1--C2 
        C2= env.C[(c,(1,1))]
        E= torch.tensordot(E,C2,([2],[1]))
    elif direction==(1,0): #right
        C1= env.C[(c,(1,1))]
        T= env.T[(c,direction)]
        #       0 
        #    1--T
        #       2
        #       0
        # 2<-1--C1
        E = torch.tensordot(T,C1,([2],[0]))
        if verbosity>0: print("E=CT "+str(E.size()))
        #    0--C2
        #       1
        #       0
        #    1--T
        #       |
        #    2--C1
        C2= env.C[(c,(1,-1))]
        E= torch.tensordot(C2,E,([1],[0]))
    else:
        raise ValueError("Invalid direction: "+str(direction))

    if verbosity>0: print("E=CTC "+str(E.size()))
    
    return E

def apply_edge(coord, direction, state, env, vec, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction of the edge
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type vec: torch.tensor
    :type verbosity: int
    :return: scalar resulting from the contraction of ``vec`` with an edge
             built from the environment
    :rtype: torch.tensor
    
    Contracts ``vec`` tensor with the edge of site ``coord`` and chosen 
    ``direction``. Afterwards, their dot product is computed::

                         get_edge(coord,direction,...)
                   ---0  0--C
                  |         |     
        scalar = vec--1  1--T
                  |         |
                   ---2  2--C
    """
    E= get_edge(coord, direction, state, env, verbosity=verbosity)
    S = torch.tensordot(vec,E,([0,1,2],[0,1,2]))
    if verbosity>0: print("S=SC "+str(S.size()))

    return S

def apply_TM_1sO(coord, direction, state, env, edge, op=None, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction in which the transfer operator is applied
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: operator to be inserted into transfer matrix
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied 
             The resulting tensor has an identical index structure as the 
             original ``edge`` 
    :rtype: torch.tensor
    
    Applies a single instance of the "transfer matrix" of site r=(x,y) to 
    the ``edge`` tensor by contracting the following network, or its corresponding 
    rotation depending on the ``direction``::

        direction:  right=(1,0)                down=(0,1)

                 -----T----------------      --------edge--------   
                |     |                     |         |          |
               edge--(a(r)^+ op a(r))--     T--(a(r)^+ op a(r))--T 
                |     |                     |         |          |
                 -----T----------------

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``op`` (if supplied).
    """
    # TODO stronger verification
    if op is not None:
        assert(len(op.size())==2)

    c = state.vertexToSite(coord)
    if direction == (0,-1): #up
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        # 0
        # T1--2->3
        # 1
        # 0   1   2
        # --edge--
        E = torch.tensordot(edge,T1,([0],[1]))
        if verbosity>0: print("E=edgeT "+str(E.size()))

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
        a= state.site(c)
        dims_a = a.size()
        X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device) if op is None else op
        A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a).contiguous()\
            .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

        # 0        0->2
        # T1--3 1--A--3
        # |        2
        # |        1    2->1
        # --------edge--    
        E = torch.tensordot(E,A,([1,3],[2,1]))
        if verbosity>0: print("E=EA "+str(E.size()))

        # 0   2->1    0->2
        # T1--A--3 1--T2
        # |   |       2
        # |   |       1
        # ---edge----- 
        T2 = env.T[(c,(1,0))]
        E = torch.tensordot(E,T2,([1,3],[2,1]))
        if verbosity>0: print("E=ET "+str(E.size()))
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
        #         3<-2----
        E = torch.tensordot(T1,edge,([2],[0]))
        if verbosity>0: print("E=edgeT "+str(E.size()))

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
        a= state.site(c)
        dims_a = a.size()
        X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device) if op is None else op
        A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a).contiguous()\
            .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

        #        0--T1-------
        #           1        | 
        #           0        |
        #     2<-1--A--3 2--edge  
        #        3<-2        |
        #             1<-3---
        E = torch.tensordot(E,A,([1,2],[0,3]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #       0--T1-------
        #          |        |
        #          |        |
        #    1<-2--A-------edge
        #          3        |
        #          0        |
        #    2<-1--T2--2 1--
        T2 = env.T[(c,(0,1))]
        E = torch.tensordot(E,T2,([1,3],[2,0]))
        if verbosity>0: print("E=ET "+str(E.size()))
    elif direction == (0,1): #down
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #  --edge--
        # 0   1->2 2->3
        # 0
        # T1--2->1
        # 1->0
        E = torch.tensordot(T1,edge,([0],[0]))
        if verbosity>0: print("E=edgeT "+str(E.size()))

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
        a= state.site(c)
        dims_a = a.size()
        X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device) if op is None else op
        A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a).contiguous()\
            .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

        #  -------edge--
        # |        2    3->1
        # |        0    
        # T1--1 1--A--3
        # 0        2
        E = torch.tensordot(E,A,([1,2],[1,0]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #  --edge-----
        # |   |       1
        # |   |       0
        # T1--A--3 1--T2
        # 0   2->1    2
        T2 = env.T[(c,(1,0))]
        E = torch.tensordot(E,T2,([1,3],[0,1]))
        if verbosity>0: print("E=ET "+str(E.size()))
    elif direction == (1,0): #right
        T1 = env.T[(c,(0,-1))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #       -- 0
        # edge |-- 1
        #       -- 2
        #
        #   ----0 0--T1--2->3
        #  |         1->2
        # edge--1->0
        #  |
        #   ----2->1
        E = torch.tensordot(edge,T1,([0],[0]))
        if verbosity>0: print("E=edgeT "+str(E.size()))

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
        a= state.site(c)
        dims_a = a.size()
        X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device) if op is None else op
        A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a).contiguous()\
            .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

        #   ---------T1--3->1 
        #  |         2
        #  |         0
        # edge--0 1--A--3   
        #  |         2
        #   ----1->0
        E = torch.tensordot(E,A,([0,2],[1,0]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #   -------T1--1->0
        #  |       |
        #  |       |
        # edge-----A--3->1
        #  |       2
        #  |       0
        #   --0 1--T2--2->2
        T2 = env.T[(c,(0,1))]
        E = torch.tensordot(E,T2,([0,2],[1,0]))
        if verbosity>0: print("E=ET "+str(E.size()))
        
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return E

# def apply_TM_2sO(state, env, edge, op=None, verbosity=0):
#     r"""
#     :param state: underlying 1-site C4v symmetric wavefunction
#     :param env: C4v symmetric environment corresponding to ``state``
#     :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
#     :param op: two-site operator to be inserted into the two consecutive
#                transfer matrices
#     :param verbosity: logging verbosity
#     :type state: IPEPS
#     :type env: ENV_C4V
#     :type edge: torch.tensor
#     :type op: torch.tensor
#     :type verbosity: int
#     :return: ``edge`` with two transfer matrices (and operator ``op``, if any) applied.
#              The resulting tensor has an identical index structure as the 
#              original ``edge``
#     :rtype: torch.tensor
    
#     Applies two transfer matrices to the ``edge`` tensor, including the two-site operator
#     ``op`` by contracting the following network::

#          -----T-------------T------------
#         |     |             |
#        edge--(A^+ op_l A)==(A^+ op_r A)--
#         |     |             |
#          -----T-------------T------------

#     where the physical indices `s` and `s'` of the on-site tensor :math:`A` 
#     and it's hermitian conjugate :math:`A^\dagger` are contracted with 
#     identity :math:`\delta_{s,s'}` or ``op_l`` and ``op_r`` if ``op`` is supplied.
#     The ``op_l`` and ``op_r`` are given by the SVD decomposition of two-site operator
#     ``op``::

#          0  1        0           1          0            1->0
#          |  |  SVD   |           |          |            |
#         | op |  =  |op_l|--(S--|op^~_r|) = |op_l|--2 2--|op_r| 
#          |  |        |           |          |            |
#          2  3        2           3          2->1         3->1
#     """
#     # TODO stronger verification
#     if op is not None:
#         assert(len(op.size())==4)

#         # pre-process ``op``
#         # TODO possibly truncate/compress according to the vanishingly small singular values
#         dims_op= op.size()
#         op_mat= op.permute(0,2,1,3).contiguous().reshape(dims_op[0]**2,dims_op[0]**2)
#         op_l, s, op_r= torch.svd(op_mat)
#         op_l= op_l.reshape(dims_op[0],dims_op[0],s.size()[0])
#         op_r= torch.einsum('i,ij->ij',s,op_r.t()).reshape(s.size()[0],dims_op[0],dims_op[0])
#         op_r= op_r.permute(1,2,0).contiguous()

#     T = env.T[env.keyT]
#     # Assume index structure of ``edge`` tensor to be as follows
#     # 
#     #       -- 0
#     # edge |-- 1
#     #       -- 2
#     #
#     #   ----0 0--T--1->2
#     #  |         2->3
#     # edge--1->0
#     #  |
#     #   ----2->1
#     E = torch.tensordot(edge,T,([0],[0]))
#     if verbosity>0: print("E=edgeT "+str(E.size()))

#     # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
#     #        Possibly reshape indices 1,2 of E, which are to be contracted with 
#     #        on-site tensor and contract bra,ket in two steps instead of creating 
#     #        double layer tensor
#     #    /
#     # --A--
#     #  /|s
#     #   X
#     # s'|/
#     # --A--
#     #  /
#     #
#     # where X is Id or op
#     a= next(iter(state.sites.values()))
#     dims_a = a.size()
#     X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device)[:,:,None] if op is None else op_l
#     A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a).contiguous()\
#         .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

#     #   ---------T--2->1 
#     #  |         3 4
#     #  |         0/
#     # edge--0 1--A--3   
#     #  |         2
#     #   ----1->0
#     E = torch.tensordot(E,A,([0,3],[1,0]))
#     if verbosity>0: print("E=EA "+str(E.size()))

#     #   -------T--1->0
#     #  |       | 4->2
#     #  |       |/
#     # edge-----A--3->1
#     #  |       2
#     #  |       2
#     #   --0 0--T--1->3
#     E = torch.tensordot(E,T,([0,2],[0,2]))
#     if verbosity>0: print("E=ET "+str(E.size()))

#     #   ----0 0----T--1->3
#     #  |----2->1   2->4
#     # edge--1->0
#     #  |
#     #   ----3->2
#     E = torch.tensordot(E,T,([0],[0]))
#     if verbosity>0: print("E=ET "+str(E.size()))

#     # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
#     #        Possibly reshape indices 1,2 of E, which are to be contracted with 
#     #        on-site tensor and contract bra,ket in two steps instead of creating 
#     #        double layer tensor
#     #    /
#     # --A--
#     #  /|s
#     #   X
#     # s'|/
#     # --A--
#     #  /
#     #
#     # where X is Id or op
#     X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device)[:,:,None] if op is None else op_r
#     A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a).contiguous()\
#         .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

#     #   ---------T--3->1
#     #  |         4
#     #  |----1 4-\0
#     # edge--0 1--A--3   
#     #  |         2
#     #   ----2->0
#     E = torch.tensordot(E,A,([0,1,4],[1,4,0]))
#     if verbosity>0: print("E=EA "+str(E.size()))

#     #   -------T--1->0
#     #  |       |
#     #  |       |
#     # edge-----A--3->1
#     #  |       2
#     #  |       2
#     #   --0 0--T--1->2
#     E = torch.tensordot(E,T,([0,2],[0,2]))
#     if verbosity>0: print("E=ET "+str(E.size()))

#     return E

def corrf_1sO1sO(coord, direction, state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: orientation of correlation function
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op1: first one-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    one-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
    :type dist: int
    :type verbosity: int
    :return: vector ``corrf`` of length ``dist`` holding the values of 
             correlation function :math:`\langle O_1(0) O_2(r) \rangle` 
             for :math:`r \in [1,dist]`
    :rtype: torch.tensor
    
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
    E0 = get_edge(c0, rev_d, state, env, verbosity=verbosity)
    # Apply transfer matrix with operator op1
    #
    #   -- 0     -- -----T--------- 0
    # E1-- 1 = E0-- --(A^+ op1 A)-- 1
    #   -- 2     -- -----T--------- 2
    E1 = apply_TM_1sO(c0, direction, state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_1sO(c0, direction, state, env, E0, verbosity=verbosity)

    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
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

        # and finally apply edge of opposite direction
        E12= apply_edge(c0, direction, state, env, E12, verbosity=verbosity)
        E00= apply_edge(c0, direction, state, env, E0, verbosity=verbosity)
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = torch.max(torch.abs(E0))
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf

# def corrf_2sO2sO_H(state, env, op1, get_op2, dist, verbosity=0):
#     E0 = get_edge(state, env, verbosity=verbosity)
#     # Apply double transfer matrix with operator op2
#     #
#     #   -- 0     ----T-------T---- 0
#     #  |        |  --A^+     A^+
#     #  |        | /  |--op1--|   \  
#     # E1-- 1 = E0----A       A---- 1
#     #   -- 2     ----T-------T---- 2
#     E1 = apply_TM_2sO(state, env, E0, op=op1, verbosity=verbosity)
#     # used for normalization
#     E0 = apply_TM_2sO(state, env, E0, verbosity=verbosity)

#     corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
#     for r in range(dist+1):
#         # close the end of the network by appending final double transfer matrix 
#         # with op2
#         #
#         #       C--T--T-- [ --T-- ]^r --T--T--
#         # E12 = T--|O1|-- [ --A-- ]   --|O2|--
#         #       C--T--T-- [ --T-- ]   --T--T--
#         E12= apply_TM_2sO(state, env, E1, op=get_op2(r), verbosity=verbosity)
#         # grow normalization network by single TM
#         E0 = apply_TM_1sO(state, env, E0, verbosity=verbosity)
#         # grow network with only a op1 operator by single TM
#         E1 = apply_TM_1sO(state, env, E1, verbosity=verbosity)

#         E12= apply_edge(state, env, E12, verbosity=verbosity)
#         # grow normalization network by additional TM to match the length
#         # of the network with op2 applied
#         E00= apply_TM_1sO(state, env, E0, verbosity=verbosity)
#         E00= apply_edge(state, env, E00, verbosity=verbosity)
#         corrf[r]= E12/E00

#         # normalize by largest element of E0
#         max_elem_E0 = torch.max(torch.abs(E0))
#         E0=E0/max_elem_E0
#         E1=E1/max_elem_E0

#     return corrf