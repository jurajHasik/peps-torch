import torch
from ipeps.ipeps import IPEPS
from ctm.generic.env import ENV
from linalg.custom_svd import truncated_svd_gesdd
from tn_interface import mm, contract, einsum
from tn_interface import view, permute, contiguous
from tn_interface import conj, transpose


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
        E = contract(T,C1,([2],[0]))
        if verbosity>0: print("E=CT "+str(E.size()))
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
        if verbosity>0: print("E=CT "+str(E.size()))
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
        if verbosity>0: print("E=CT "+str(E.size()))
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
        if verbosity>0: print("E=CT "+str(E.size()))
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

    if verbosity>0: print("E=CTC "+str(E.size()))
    
    return E

def get_edge_2(coord, direction, state, env, verbosity=0):
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

    Build an edge of two sites, starting at position ``coord``, by contracting 
    one of the following networks depending on the chosen ``direction``::

            up=(0,-1)   left=(-1,0)  down=(0,1)   right=(1,0)

                        C--0         0  1  2  3    0--C
                        |            |  |  |  |       |
        E = C--T--T--C  T--1         C--T--T--C    1--T  
            |  |  |  |  |                             |
            0  1  2  3  T--2                       2--T
                        |                             |
                        C--3                       3--C
    
    The edge itself is grown from top to bottom or from left to right.
    The indices of the resulting tensor are always ordered from 
    left-to-right and from up-to-down.
    """
    dir_r=(1,0)
    dir_b=(0,1)
    def shift_coord(c,d=(0,0)):
        c0= (c[0]+d[0],c[1]+d[1])
        return c0, state.vertexToSite(c0)

    c0, s= shift_coord(coord)
    if direction==(0,-1): #up
        C2= env.C[(s,(-1,-1))]
        T= env.T[(s,direction)]
        # C2--1 0--T--2
        # 0        1
        E= contract(C2,T,([1],[0]))
        c0, s= shift_coord(c0,dir_r)
        T= env.T[(s,direction)]
        # C2--T--2 0--T--2->3
        # 0   1       1->2
        E= contract(E,T,([2],[0]))
        C1= env.C[(s,(1,-1))]
        # C2--T--T--3 0--C1
        # 0   1  2        1->3
        E = contract(E,C1,([3],[0]))
    elif direction==(-1,0): #left
        C1= env.C[(s,(-1,-1))]
        T= env.T[(s,direction)]
        # C1--1->0
        # 0
        # 0
        # T--2
        # 1
        E = contract(C1,T,([0],[0]))
        c0, s= shift_coord(coord,dir_b)
        T= env.T[(s,direction)]
        # C1--0
        # T--2->1
        # 1
        # 0
        # T--2->3
        # 1->2
        E = contract(E,T,([1],[0]))
        C2= env.C[(s,(-1,1))]
        # C1--0
        # |
        # T--1
        # T--3->2
        # 2
        # 0
        # C2--1->3
        E= contract(E,C2,([2],[0]))
    elif direction==(0,1): #down
        C1= env.C[(s,(-1,1))]
        T= env.T[(s,direction)]
        # 0        0->1
        # C1--1 1--T--2
        E = contract(C1,T,([1],[1]))
        c0, s= shift_coord(c0,dir_r)
        T= env.T[(s,direction)]
        # 0   1       0->2
        # C1--T--2 1--T--2->3
        E = contract(E,T,([2],[1]))
        C2= env.C[(s,(1,1))]
        # 0   1  2       0->3
        # C1--T--T--3 1--C2 
        E= contract(E,C2,([3],[1]))
    elif direction==(1,0): #right
        C2= env.C[(s,(1,-1))]
        T= env.T[(s,direction)]
        #    0--C2
        #       1
        #       0
        #    1--T
        #       2
        E = contract(C2,T,([1],[0]))
        c0, s= shift_coord(coord, dir_b)
        T= env.T[(s,direction)]        
        #    0--C2
        #    1--T
        #       2
        #       0
        # 2<-1--T
        #    3<-2
        E = contract(E,T,([2],[0]))
        C1= env.C[(s,(1,-1))]
        #    0--C2
        #    1--T
        #    2--T
        #       3
        #       0 
        # 3<-2--C1
        E= contract(E,C1,([3],[0]))
    else:
        raise ValueError("Invalid direction: "+str(direction))

    if verbosity>0: print("E=CTTC "+str(E.size()))
    
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
    :type state: IPEPS
    :type env: ENV
    :type vec: torch.tensor
    :type verbosity: int
    :return: scalar resulting from the contraction of ``vec`` with an edge
             built from the environment
    :rtype: torch.tensor
    
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
    if len(vec.size())==3:
        E= get_edge(coord, direction, state, env, verbosity=verbosity)
    elif len(vec.size())==4:
        E= get_edge_2(coord, direction, state, env, verbosity=verbosity)
    else:
        ValueError("Unsupported edge: "+vec.size())

    inds=[i for i in range(0,len(vec.size()))]
    S = contract(vec,E,(inds,inds))
    if verbosity>0: print("S "+str(S.size()))

    return S

def apply_TM_0sO(coord, direction, state, env, edge, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction in which the transfer operator is applied
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi^2`
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type edge: torch.tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied 
             The resulting tensor has an identical index structure as the 
             original ``edge`` 
    :rtype: torch.tensor
    
    Applies a single instance of the "0-width channel" transfer matrix of site r=(x,y) to 
    the ``edge`` tensor by contracting the following network, or its corresponding 
    rotation depending on the ``direction``::

        direction:  right=(1,0)                down=(0,1)

                 -----T--                    edge
                |     |                     |    |
                 -----T--                   T----T
                                            |    |

    """
    assert direction in [(0,-1),(-1,0),(0,1),(1,0)],"Invalid direction: "+str(direction)
    # right is identical do down and left is identical to right
    if direction==(1,0): direction=(-1,0) 
    if direction==(0,1): direction=(0,-1)

    c = state.vertexToSite(coord)
    if direction == (0,-1): #up
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        # 0
        # T1(x,y)--2->1
        # 1
        # 0       1->2 
        # --edge--- 
        E = contract(T1,edge,([1],[0]))
        if verbosity>0: print("E=edgeT "+str(E.size()))


        # 0             0->1
        # |             |
        # T1(x,y)--1 1--T2(x-1,y)
        # |             2
        # |             2
        # ---edge--------
        T2 = env.T[(state.vertexToSite( (c[0]-1,c[1]) ),(1,0))]
        E = contract(E,T2,([1,2],[1,2]))
        if verbosity>0: print("E=ETT "+str(E.size()))
    elif direction == (-1,0): #left
        T1 = env.T[(c,(0,-1))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #       0 -- 
        #           | edge
        #       2 -- 
        #
        #   0--T1(x,y)--2 0---- 
        #      1               edge
        #              2<-1---- 
        #                
        E = contract(T1,edge,([2],[0]))
        if verbosity>0: print("E=edgeT "+str(E.size()))

        #           0--T1(x,y)---------
        #              1               |
        #              0              edge
        #              |               |
        #           1--T2(x,y-1)--2 2--
        #              (c+(0,-1))
        T2 = env.T[( state.vertexToSite( (c[0],c[1]-1) ),(0,1))]
        E = contract(E,T2,([1,2],[0,2]))
        if verbosity>0: print("E=ETT "+str(E.size()))

    return E

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
    :type state: IPEPS
    :type env: ENV
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
    contains_mpo= len(edge.size())==4

    # Four basic cases of passed op
    def get_aXa(a, op):
        # a - on-site tensor
        # op - operator
        dims_a= a.size()
        dims_op= None if op is None else op.size()
        if op is None:
            # identity
            A= einsum('nefgh,nabcd->eafbgchd',a,conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2))
        elif len(dims_op)==2:
            # one-site operator
            A= einsum('nefgh,nabcd->eafbgchd', einsum('mefgh,mn->nefgh',a,op), conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2))
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
            # It will become the last index of the resulting edge 
            A= einsum('nefghl,nabcd->eafbgchdl', einsum('mefgh,mnl->nefghl',a,op), conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1))
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
            A= einsum('nefghlk,nabcd->eafbgchdlk', einsum('mefgh,mnlk->nefghlk',a,op), conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, dims_op[2], dims_op[3]))
        if verbosity>0: print(f"aXa {A.size()}")
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
        A= get_aXa(a, op)

        # edge with no MPO              edge with MPO index
        #
        # 0     2<-0 (4)                0     2<-0 ---------(4)(5->4)
        # |        |/                   |        |/          |
        # T1--1 1--A--3                 T1--1 1--A--3        |
        # |        2                    |        2           | 
        # |        2    3->1            |        2    3->1  (4)
        # --------edge--                --------edge-- ------
        E = contract(E,A,([1,2,4],[1,2,4])) if contains_mpo else \
            contract(E,A,([1,2],[1,2]))
        if verbosity>0: print("E=EA "+str(E.size()))

        # 0  1<-2 (4->2)    0->2 (0->3)
        # |     |/          |
        # T1----A--3 1------T2
        # |     |           2
        # |     |           1
        # -----edge---------
        T2 = env.T[(c,(1,0))]
        E = contract(E,T2,([1,3],[2,1]))
        # if we have extra MPO dimension, permute it to the last index
        if len(E.size())==4: E= contiguous(permute(E,(0,1,3,2)))
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
        #         3<-2----|
        #        (4<-3)--- 
        E = contract(T1,edge,([2],[0]))
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
        A= get_aXa(a, op)

        #        0--T1---------              0--T1---------
        #           1          |                1          | 
        #           0          |                0          |
        #     2<-1--A----3 2--edge        2<-1--A----3 2--edge
        #         / 2->3       |              / 2->3       |
        #        |       1<-3--|             (4)     1<-3--
        # (4<-5)(4)-------(4)--
        E = contract(E,A,([1,2,4],[0,3,4])) if contains_mpo else \
            contract(E,A,([1,2],[0,3]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #           0--T1-------
        #              |        |
        #              |        |
        #        1<-2--A-------edge
        #            / 3        |
        #       (2<-4) 0        |
        # (3<-1) 2<-1--T2--2 1--
        T2 = env.T[(c,(0,1))]
        E = contract(E,T2,([1,3],[2,0]))
        if len(E.size())==4: E= contiguous(permute(E,(0,1,3,2)))
        if verbosity>0: print("E=ET "+str(E.size()))
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
        A= get_aXa(a, op)

        #  -------edge----         -------edge---- ------
        # |        2      3->1    |        2      3->1  (4)
        # |        0              |        0             |
        # T1--1 1--A--3           T1--1 1--A--3          |
        # |        |\             |        |\            |
        # 0        2 (4)          0        2 -----------(4)(5->4) 
        E = contract(E,A,([1,2,4],[1,0,4])) if contains_mpo else \
            contract(E,A,([1,2],[1,0]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #  ----edge-------
        # |     |         1
        # |     |         0
        # T1----A--3 1----T2
        # |     |\        |
        # 0  1<-2 (4->2)  2 (2->3)
        T2 = env.T[(c,(1,0))]
        E = contract(E,T2,([1,3],[0,1]))
        if len(E.size())==4: E= contiguous(permute(E,(0,1,3,2)))
        if verbosity>0: print("E=ET "+str(E.size()))
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
        A= get_aXa(a, op)

        #   ---------T1--1->0       ---------T1--1->0
        #  |         0             |         0
        #  |         0             |         0
        # edge--2 1--A--3         edge--2 1--A--3
        #  |         2 \           |         2
        #   ----3->1    |           ----3->1
        #  |            |
        #   ----4 ---- (4)(5->4)
        E = contract(E,A,([0,2,4],[0,1,4])) if contains_mpo else \
            contract(E,A,([0,2],[0,1]))
        if verbosity>0: print("E=EA "+str(E.size()))

        #   -------T1--0
        #  |       |
        #  |       |
        # edge-----A--3->1
        #  |       2 \(4->2)
        #  |       0
        #   --1 1--T2--2 (2->3)
        T2 = env.T[(c,(0,1))]
        E = contract(E,T2,([1,2],[1,0]))
        if len(E.size())==4: E= contiguous(permute(E,(0,1,3,2)))
        if verbosity>0: print("E=ET "+str(E.size()))     
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return E

def apply_TM_2sO_2sChannel(coord, direction, state, env, edge, op=None, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction in which the transfer operator is applied
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times (D^2)^2 \times \chi`
    :param op: two-site operator to be inserted within the two-site transfer matrix
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
    
    Applies a single instance of the two-site "transfer matrix" with site r=(x,y) to 
    the ``edge`` tensor by contracting the following network, or its corresponding 
    rotation depending on the ``direction``::

        direction:  right=(1,0)                       down=(0,1)

                 -----T--------------------     --------edge-------------   
                |     |                        |    |          |         |
               edge--(a(r)^+ o1 a(r))------    T--(a(r)^+)---(a(r+x)^+)--T
                |     |        \               |\   o1---------o2        |
                |----(a(r+y)^+) o2 a(r+y)--    | --a(r)-------a(r+x)-----T
                |     |                        |    |          |         |
                 -----T--------------------

    The two-site operator is first decomposed into a simple MPO o1--o2
    (TODO case where op comes with extra MPO index)::
        
         s1'  s2'    s1'      s2'
        |  op   | = |o1|-----|o2|
         s1   s2     s1       s2  

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``o1``, ``o2``. The transfer matrix
    is always grown from left to right or from top to bottom.
    """

    # TODO stronger verification
    op_1, op_2= None, None
    if op is not None:
        if len(op.size())==4:
            # pre-process ``op``
            # TODO possibly truncate/compress according to the vanishingly small singular values
            dims_op= op.size()
            op_mat= view(contiguous(permute(op,(0,2,1,3))),(dims_op[0]**2,dims_op[0]**2))
            op_1, s, op_2= truncated_svd_gesdd(op_mat, op_mat.size(0))
            op_1= view(op_1, (dims_op[0],dims_op[0],s.size()[0]))
            op_2= conj(transpose(op_2))
            op_2= contiguous(permute(view(op_2, (s.size()[0],dims_op[0],dims_op[0])), (1,2,0)))
        else:
            raise ValueError(f"Invalid op: rank {op.size()}")

    def shift_coord(c,d=(0,0)):
        c0= (c[0]+d[0],c[1]+d[1])
        return c0, state.vertexToSite(c0)

    # Four basic cases of passed op
    def get_aXa(a, op):
        # a - on-site tensor
        # op - operator
        dims_a= a.size()
        dims_op= None if op is None else op.size()
        if op is None:
            # identity
            A= einsum('nefgh,nabcd->eafbgchd',a,conj(a))
            A= view(contiguous(A), (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2))
        elif len(dims_op)==2:
            # one-site operator
            A= einsum('nefgh,nabcd->eafbgchd',einsum('mefgh,mn->nefgh',a,op),conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2))
        elif len(dims_op)==3:
            # edge operators of some MPO within the transfer matrix
            #
            # 0                   0
            # |                   |
            # op--2 ... or ... 2--op
            # |                   |
            # 1                   1
            #
            # assume the last index of the op is the MPO dimension.
            # It will become the last index of the resulting edge 
            A= einsum('nefghl,nabcd->eafbgchdl',einsum('mefgh,mnl->nefghl',a,op),conj(a))
            A= view(contiguous(A), \
                (dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1))
        if verbosity>0: print(f"aXa {A.size()}")
        return A

    dir_r=(1,0)
    dir_b=(0,1)
    c0, c = shift_coord(coord)
    if direction == (0,-1): #up
        raise ValueError("Direction: "+str(direction)+"not implemented")
    elif direction == (-1,0): #left
        raise ValueError("Direction: "+str(direction)+"not implemented")
    elif direction == (0,1): #down
        T1 = env.T[(c,(-1,0))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #  --edge-- ------
        # 0   1->2 2->3  3->4
        # 0
        # T1--2->1
        # 1->0
        E = contract(T1,edge,([0],[0]))
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
        A= get_aXa(a, op_1)
        #  -------edge---- ------    
        # |        2      3->1   4->2
        # |        0                
        # T1--1 1--A--3->4
        # |        |\             
        # 0     3<-2 (4->5)        
        E = contract(E,A,([1,2],[1,0]))
        if verbosity>0: print("E=edgeTA "+str(E.size()))

        c0, c= shift_coord(c0, dir_r)
        a= state.site(c)
        A= get_aXa(a, op_2)
        #  ----edge------ ------
        # |     |        1      2->1
        # |     |        0
        # T1----A--4 1---A--3->4 
        # |     |\      /|
        # 0  2<-3 (5)(4) 2->3
        E = contract(E,A,([1,4],[0,1])) if op is None else \
            contract(E,A,([1,4,5],[0,1,4]))
        if verbosity>0: print("E=edgeTAA "+str(E.size()))

        T2 = env.T[(c,(1,0))]
        #  ----edge----- -------
        # |     |       |       1
        # |     |       |       0
        # T1----A=======A--4 1--T2
        # |     |       |       2->4
        # 0     2->1    3->2
        E = contract(E,T2,([1,4],[0,1]))
        if verbosity>0: print("E=edgeTAAT "+str(E.size()))
    elif direction == (1,0): #right
        T1 = env.T[(c,(0,-1))]
        # Assume index structure of ``edge`` tensor to be as follows
        # 
        #       -- 0
        # edge |-- 1
        #      |---2
        #       -- 3
        #
        #   ----0 0--T1--2->1
        #  |         1->0
        # edge--1->2
        #  |
        #   ----2->3
        #  |
        #   ----3->4
        E = contract(T1,edge,([0],[0]))
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
        A= get_aXa(a, op_1)

        #   ---------T1--1->0
        #  |         0
        #  |         0
        # edge--2 1--A--3->4
        #  |      3<-2 \
        #   ----3->1   (4->5)
        #  |            
        #   ----4->2
        E = contract(E,A,([0,2],[0,1]))
        if verbosity>0: print("E=edgeTA "+str(E.size()))

        c0, c= shift_coord(c0, dir_b)
        a= state.site(c)
        A= get_aXa(a, op_2)
        #   ---------T1--0
        #  |         |
        # edge-------A--4->2
        #  |         | \
        #  |         3 (5)
        #  |         0 (4)
        #  |         | /     
        #   ----1 1--A--2->3
        #  |         3->4   
        #   ----2->1
        E = contract(E,A,([1,3],[1,0])) if op is None else \
            contract(E,A,([1,3,5],[1,0,4]))
        if verbosity>0: print("E=edgeTAA "+str(E.size()))

        T2 = env.T[(c,(0,1))]
        #   ---------T1--0
        #  |         |
        # edge-------A--2->1
        #  |         |   
        #   ---------A--3->2
        #  |         3
        #  |         0   
        #   ----1 1--T2--2->3
        E = contract(E,T2,([1,3],[1,0]))
        if verbosity>0: print("E=edgeTAAT "+str(E.size()))     
    else:
        raise ValueError("Invalid direction: "+str(direction))

    return E

def apply_TM_2sO_1sChannel(coord, direction, state, env, edge, op=None, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: direction in which the transfer operator is applied
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: two-site operator to be inserted into the two consecutive
               transfer matrices
    :param verbosity: logging verbosity
    :type coord: tuple(int,int)
    :type direction: tuple(int,int)
    :type state: IPEPS
    :type env: ENV
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with two transfer matrices (and operator ``op``, if any) applied.
             The resulting tensor has an identical index structure as the 
             original ``edge``
    :rtype: torch.tensor
    
    Applies two transfer matrices to the ``edge`` tensor, including the two-site operator
    ``op``. The applied transfer matrices depend on the selected site r=(x,y) and  
    the ``direction`` of the growth. The following network is contracted::

         -----T-------------------T--------------------------
        |     |                   |
       edge--(a(r)^+ op_l a(r))==(a(r+dir)^+ op_r a(r+dir))--
        |     |                   |
         -----T-------------------T--------------------------

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``op_l`` and ``op_r`` if ``op`` is supplied.
    The ``op_l`` and ``op_r`` are given by the SVD decomposition of the two-site operator
    ``op``::

         0  1        0           1          0            1->0
         |  |  SVD   |           |          |            |
        | op |  =  |op_l|--(S--|op^~_r|) = |op_l|--2 2--|op_r|
         |  |        |           |          |            |
         2  3        2           3          2->1         3->1
    """
    # TODO stronger verification
    if op is not None:
        assert(len(op.size())==4)

        # pre-process ``op``
        # TODO possibly truncate/compress according to the vanishingly small singular values
        dims_op= op.size()
        op_mat= view(contiguous(permute(op,(0,2,1,3))), (dims_op[0]**2,dims_op[0]**2))
        op_l, s, op_r= truncated_svd_gesdd(op_mat, op_mat.size(0))
        op_l= view(op_l, (dims_op[0],dims_op[0],s.size()[0]))
        op_r= conj(transpose(op_r))*s[:,None]
        op_r= contiguous(permute(view(op_r, (s.size()[0],dims_op[0],dims_op[0])), (1,2,0)))
    else:
        op_l=None
        op_r=None

    E= apply_TM_1sO(coord, direction, state, env, edge, op=op_l, verbosity=verbosity)
    c1= (coord[0]+direction[0],coord[1]+direction[1])
    E= apply_TM_1sO(c1, direction, state, env, E, op=op_r, verbosity=verbosity)

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
    :type state: IPEPS
    :type env: ENV
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
    :type dist: int
    :type rl_0: tuple(function(tuple(int,int))->torch.Tensor, function(tuple(int,int))->torch.Tensor)
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
    E0 = get_edge(c0, rev_d, state, env, verbosity=verbosity) if rl_0 is None else rl_0[0](c0)
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

        # and finally apply edge in the direction of growth
        if rl_0 is None:
            E12= apply_edge(c0, direction, state, env, E12, verbosity=verbosity)
            E00= apply_edge(c0, direction, state, env, E0, verbosity=verbosity)
        else:
            E12= torch.tensordot(E12,rl_0[1](c0),([0,1,2],[0,1,2]))
            E00= torch.tensordot(E0,rl_0[1](c0),([0,1,2],[0,1,2]))
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = E0.abs().max()
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf

def corrf_2sOH2sOH_E1(coord, direction, state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: orientation of correlation function
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op1: first two-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    two-site operator :math:`\text{get_op2}(r)=O_2`
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

    Computes the correlation function :math:`\langle O_1(0) O_2(r) \rangle`
    of two horizontaly-oriented two-site operators by contracting 
    the following network::

        C-----T-------T---------- ... ---T---- ... -----T-------T-------------C
        |     |       |                  |              |       |             |
        |   /-a(0)^+--a(1)^+-\           a(i)^+      /--a(r)^+--a(r+1)^+--\   |
        T--<    ( op 1 )      >-- ... --<|>-- ... --<  (gen_op2(r))        >--T 
        |   \-a(0)----a(1)---/           a(i))       \--a(r)----a(r+1)----/   |
        |     |       |                  |              |       |             |
        C-----T-------T---------- ... ---T---- ... -----T-------T-------------C

    for increasingly large distance ``r`` up to ``dist+1``. The ``op1`` is 
    applied at site 0=(x,y) & 1, the transfer matrices are applied in the 
    ``direction`` up to site :math:`r=(x,y) + \text{dist} \times \text{direction}`
    """
    def shift_c(c,d,r=1):
        return (c[0]+r*d[0],c[1]+r*d[1])

    c0= coord
    rev_d = (-direction[0],-direction[1]) # opposite direction
    E0 = get_edge(c0, rev_d, state, env, verbosity=verbosity)
    # Apply double transfer matrix with operator op2
    #
    #   -- 0     ----T-------T---- 0
    #  |        |  --A^+     A^+
    #  |        | /  |--op1--|   \  
    # E1-- 1 = E0----A       A---- 1
    #   -- 2     ----T-------T---- 2
    E1 = apply_TM_2sO_1sChannel(c0, direction, state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_2sO_1sChannel(c0, direction, state, env, E0, verbosity=verbosity)

    c0= shift_c(c0,direction,r=2)
    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
    for r in range(dist+1):
        # close the end of the network by appending final double transfer matrix 
        # with op2
        #
        #       C--T--T-- [ --T-- ]^r --T--T--
        # E12 = T--|O1|-- [ --A-- ]   --|O2|--
        #       C--T--T-- [ --T-- ]   --T--T--
        E12= apply_TM_2sO_1sChannel(c0, direction, state, env, E1, op=get_op2(r), verbosity=verbosity)
        #c12_e= shift_c(c0,direction)
        #print(f"c0: {c0} c12_e: {c12_e}")
        # grow normalization network by single TM
        E0 = apply_TM_1sO(c0, direction, state, env, E0, verbosity=verbosity)
        # grow network with only a op1 operator by single TM
        E1 = apply_TM_1sO(c0, direction, state, env, E1, verbosity=verbosity)
        c0= shift_c(c0,direction)

        E12= apply_edge(c0, direction, state, env, E12, verbosity=verbosity)
        # grow normalization network by additional TM to match the length
        # of the network with op2 applied
        E00= apply_TM_1sO(c0, direction, state, env, E0, verbosity=verbosity)
        E00= apply_edge(c0, direction, state, env, E00, verbosity=verbosity)
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = E0.abs().max()
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf

def corrf_2sOV2sOV_E2(coord, direction, state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param coord: tuple (x,y) specifying vertex on a square lattice
    :param direction: orientation of correlation function
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op1: first two-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    two-site operator :math:`\text{get_op2}(r)=O_2`
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
    
    Computes the four-point correlation function :math:`\langle O_1(0) O_2(r) \rangle`,
    where both O_1 and O_2 are two-site operator by contracting the following network::

        C-----T--------------- ... ----T------------- ... -----T-------------C
        |     |                        |                       |             |
        T--(a(0)^+-----a(0))-- ... --a(i)^+(ai)------ ... --(a(r)^+a(r)------T
        |          op1                 |                   ( gen_op2(r))     |
        T--(a(y)^+-----a(y))-- ... --a(i+y)^+a(i+y)-- ... --(a(r+y)^+a(r+y)--T
        |     |                        |                       |             | 
        C-----T--------------- ... ----T------------- ... -----T-------------C

    for increasingly large distance ``r`` up to ``dist``. The ``op1`` is 
    applied within the transfer matrix defined by site 0=(x,y). The transfer matrices 
    are applied in the ``direction`` up to site :math:`r=(x,y) + \text{dist} \times \text{direction}`. 
    """
    def shift_c(c,d):
        return (c[0]+d[0],c[1]+d[1])

    c0= coord
    rev_d = (-direction[0],-direction[1]) # opposite direction
    E0 = get_edge_2(c0, rev_d, state, env, verbosity=verbosity)
    # Apply transfer matrix with operator op1
    #
    #   -- 0     -- -----T--------- 0
    # E1-- 1 = E0-- --(A^+ op1 A)-- 1
    #   -- 2     -- --(A^+ op1 A)-- 2
    #   -- 3     -- -----T--------- 3
    E1 = apply_TM_2sO_2sChannel(c0, direction, state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_2sO_2sChannel(c0, direction, state, env, E0, verbosity=verbosity)

    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
    for r in range(dist+1):
        # close the end of the network by appending final transfer matrix 
        # with op2
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        c0= shift_c(c0,direction)
        E12= apply_TM_2sO_2sChannel(c0, direction, state, env, E1, op=get_op2(r), verbosity=verbosity)
        # and corresponding normalization network
        E0 = apply_TM_2sO_2sChannel(c0, direction, state, env, E0, verbosity=verbosity)
        # and to network with only a op1 operator
        E1 = apply_TM_2sO_2sChannel(c0, direction, state, env, E1, verbosity=verbosity)

        # and finally apply edge of opposite direction
        E12= apply_edge(c0, direction, state, env, E12, verbosity=verbosity)
        E00= apply_edge(c0, direction, state, env, E0, verbosity=verbosity)
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = E0.abs().max()
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf