import numpy as np
from tn_interface_abelian import contract, permute

def get_edge(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V_ABELIAN
    :type env: ENV_C4V_ABELIAN
    :type verbosity: int
    :return: edge of the width-1 channel
    :rtype: Tensor
    
    Build initial edge by contracting the following network::

            C--0(-)
            |
        E = T--1(-) 1,2
            |     
            C--2(-) 3
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]

    # C--1->0
    # 0
    # 0
    # T--2,3
    # 1
    E = contract(C,T,([0],[0]))
    if verbosity>0: print(f"E=CT \n{E}")
    
    # C--0
    # |
    # T--2,3->1,2
    # 1
    # 0
    # C--1->3
    E = contract(E,C,([1],[0]))
    if verbosity>0: print(f"E=CT \n{E}")
    return E

def apply_edge(state, env, vec, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type vec: torch.Tensor
    :type verbosity: int
    :return: scalar resulting from the contraction of ``vec`` with an edge
             built from environment
    :rtype: torch.tensor
    
    Contracts ``vec`` tensor with the corresponding edge by contracting 
    the following network::

                   ---C
                  |   |     
        scalar = vec--T
                  |   |
                   ---C
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    
    if vec.s[0]*C.s[0]==1:
        # flip_signature
        C= C.flip_signature()
        T= T.flip_signature()

    # Assume index structure of ``edge`` tensor to be as follows
    # 
    #      -- 0
    # vec |-- 1,2
    #      -- 3
    #
    #   ---0 0------C
    #  |            1->3
    # vec--1,2->0,1
    #  |
    #   ---3->2
    S = contract(vec,C,([0],[0]))
    if verbosity>0: print("S=vecC "+str(S.size()))

    #   ------------C 
    #  |            3
    #  |            0
    # vec--0,1 2,3--T   
    #  |            1
    #   ---2->0
    S = contract(S,T,([0,1,3],[2,3,0]))
    if verbosity>0: print("S=ST "+str(S.size()))

    #   -------C
    #  |       |
    # edge-----T
    #  |       1
    #  |       1
    #   --0 0--C
    S = contract(S,C,([0,1],[0,1]))
    if verbosity>0: print("S=SC "+str(S.size()))

    return S

def apply_TM_1sO(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor corresponding to the edge of width-1 channel
    :param op: operator to be inserted into transfer matrix
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V_ABELIAN
    :type env: ENV_C4V_ABELIAN
    :type edge: yastn.Tensor
    :type op: yastn.Tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied.
             The resulting tensor has an identical index structure as the 
             original ``edge`` 
    :rtype: tensor
    
    Applies a single instance of the "transfer matrix" to the ``edge`` tensor  
    by contracting the following network::

         -----T----------
        |     |      
       edge--(a^+ op a)--
        |     |     
         -----T----------

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``op`` (if supplied).
    """
    # TODO stronger verification
    if op is not None:
        assert op.ndim==2,"1-site operator is expected to have two indices"

    T= env.T[env.keyT]
    a= state.site()

    _FLIP_SIGNATURE= False
    if edge.s[0]*T.s[0]==1:
        # flip_signature
        _FLIP_SIGNATURE= True
        T= T.flip_signature()
        a= a.flip_signature()

    # Assume index structure of ``edge`` tensor to be as follows
    # 
    #       -- 0(+)
    # edge |-- 1(-) 1,2
    #       -- 2(+) 3
    #
    #   ----0 0--T--1->2
    #  |         2->3
    # edge--1->0
    #  |
    #   ----2->1

    #      ----0 0--T--1->3
    #     |         2->4
    # 3--edge--1->0
    # ->2 |
    #      ----2->1

    #      ----0 0--T--1->4
    #     |         2,3->5,6
    # 4--edge--1,2->0,1
    # ->3 |
    #      ----3->2
    E = contract(edge,T,([0],[0]))
    if verbosity>0: print(f"E=CT \n{E}")

    # TODO - more efficent contraction with uncontracted-double-layer on-site tensor
    #        Possibly reshape indices 1,2 of E, which are to be contracted with 
    #        on-site tensor and contract bra,ket in two steps instead of creating 
    #        double layer tensor
    #    /         /
    # --a-- <=  --a--
    #  /|s       /|0(+)
    #   op        |0(-)
    # s'|/        op
    # --a--       |1(+)->0(+) 
    #  /
    #
    # where X is Id or op
    # if _FLIP_SIGNATURE:
    #     a_dl= contract(op,a, ([1],[0])) if op else a
    # else:
    a_dl= contract(op,a, ([0],[0])) if op else a
    a_dl= contract(a_dl,a, ([0],[0]), conj=(0,1)) # mefgh,mabcd->efghabcd
    a_dl= a_dl.transpose((0,4,1,5,2,6,3,7)) # efghabcd->eafbgchd

    #   ---------T--2->1 
    #  |         3
    #  |         0
    # edge--0 1--A--3   
    #  |         2
    #   ----1->0
    # E = contract(E,a_dl,([0,3],[1,0]))

    #      ---------T--3->2 
    #     |         4
    #     |         0
    # 2--edge--0 1--A--3->4   
    # ->1 |         2->3
    #      ----1->0
    # E = contract(E,a_dl,([0,4],[1,0]))

    #      -------------T--4->2 
    #     |             5,6
    #     |             0,1
    # 3--edge--0,1 2,3--A--6,7->5,6
    # ->1 |             4,5->3,4
    #      ----2->0
    E = contract(E,a_dl,([0,1,4,5],[2,3,0,1]))
    # E = contract(E,a_dl,([0,1,5,6],[2,3,0,1]))
    if verbosity>0: print(f"E=CT \n{E}")

    #   -------T--1->0
    #  |       |
    #  |       |
    # edge-----A--3->1
    #  |       2
    #  |       2
    #   --0 0--T--1->2
    # E = contract(E,T,([0,2],[0,2]))

    #      -------T--2->1
    #     |       |
    #     |       |
    # 1--edge-----A--4->2
    # ->0 |       3
    #     |       2
    #      --0 0--T--1->3
    # E = contract(E,T,([0,3],[0,2]))
    # E = E.transpose((1,2,3,0))

    #      -------T--2->1
    #     |       |
    #     |       |
    # 1--edge-----A--5,6->2,3
    # ->0 |       3,4
    #     |       2,3
    #      --0 0--T--1->4
    E = contract(E,T,([0,2,3],[0,2,3]))
    # E = contract(E,T,([0,3,4],[0,2,3]))
    # E = E.transpose((1,2,3,4,0))
    if verbosity>0: print(f"E=CT \n{E}")

    return E

def corrf_1sO1sO(state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param op1: first one-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    one-site opreator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type state: IPEPS_ABELIAN_C4V
    :type env: ENV_ABELIAN_C4V
    :type op1: yastn.Tensor
    :type get_op2: function(int)->yastn.Tensor
    :type dist: int
    :type verbosity: int
    :return: vector ``corrf`` of length ``dist`` holding the values of 
             correlation function :math:`\langle O_1(0) O_2(r) \rangle` for :math:`r \in [1,dist]`
    :rtype: torch.tensor
    
    Computes the two-point correlation function :math:`\langle O_1(0) O_2(r) \rangle`
    by contracting the following network::

        C-----T---------- ... -----T---- ... --------T-------------C
        |     |                    |                 |             |
        T--(a^+ op_1 a)-- ... --(a^+a)-- ... --(a^+ gen_op2(r) a)--T
        |     |                    |                 |             | 
        C-----T---------- ... -----T---- ... --------T-------------C

    for increasingly large distance ``r`` up to ``dist``.
    """
    E0 = get_edge(state, env, verbosity=verbosity)
    # Apply transfer matrix with operator op1
    #
    #   -- 0     -- -----T--------- 0
    # E1-- 1 = E0-- --(A^+ op1 A)-- 1
    #   -- 2     -- -----T--------- 2
    E1 = apply_TM_1sO(state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_1sO(state, env, E0, verbosity=verbosity)

    corrf=np.empty(dist+1, dtype="float64")
    for r in range(dist+1):
        # close the end of the network by appending final transfer matrix 
        # with op2
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        E12= apply_TM_1sO(state, env, E1, op=get_op2(r), verbosity=verbosity)
        # and corresponding normalization network
        E0 = apply_TM_1sO(state, env, E0, verbosity=verbosity)
        # and to network with only a op1 operator
        E1 = apply_TM_1sO(state, env, E1, verbosity=verbosity)

        E12= apply_edge(state, env, E12, verbosity=verbosity)
        E00= apply_edge(state, env, E0, verbosity=verbosity)
        corrf[r]= E12.to_number()/E00.to_number()

        import pdb; pdb.set_trace()

        # normalize by largest element of E0
        max_elem_E0 = E0.norm(p='inf')
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf