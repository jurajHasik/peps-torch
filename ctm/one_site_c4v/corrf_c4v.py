import torch
from ipeps import IPEPS
from ctm.one_site_c4v.env_c4v import ENV_C4V

def get_edge(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type verbosity: int
    :return: tensor with indices :math:`\chi \times D^2 \times \chi`
    :rtype: torch.tensor
    
    Build initial edge by contracting the following network::

            C--
            |
        E = T--
            |     
            C--
    """
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    # C--1->0
    # 0
    # 0
    # T--2
    # 1
    E = torch.tensordot(C,T,([0],[0]))
    if verbosity>0: print("E=CT "+str(E.size()))
    # C--0
    # |
    # T--2->1
    # 1
    # 0
    # C--1->2
    E = torch.tensordot(E,C,([1],[0]))
    if verbosity>0: print("E=CTC "+str(E.size()))
    return E

def apply_edge(state, env, vec, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type vec: torch.tensor
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
	# Assume index structure of ``edge`` tensor to be as follows
	# 
	#      -- 0
	# vec |-- 1
	#      -- 2
	#
	#   ---0 0--C
	#  |        1->2
	# vec--1->0    
	#  |
	#   ---2->1
    S = torch.tensordot(vec,C,([0],[0]))
    if verbosity>0: print("S=vecC "+str(S.size()))

	#   --------C 
	#  |        2
	#  |        0
	# vec--0 2--T   
	#  |        1
	#   ---1->0
    S = torch.tensordot(S,T,([0,2],[2,0]))
    if verbosity>0: print("S=ST "+str(S.size()))

	#   -------C
	#  |       |
	# edge-----T
	#  |       1
	#  |       1
	#   --0 0--C
    S = torch.tensordot(S,C,([0,1],[0,1]))
    if verbosity>0: print("S=SC "+str(S.size()))

    return S

def apply_TM_1sO(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: operator to be inserted into transfer matrix
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied.
             The resulting tensor has an identical index structure as the 
             original ``edge`` 
    :rtype: torch.tensor
    
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
    	assert(len(op.size())==2)

    T = env.T[env.keyT]
	# Assume index structure of ``edge`` tensor to be as follows
	# 
	#       -- 0
	# edge |-- 1
	#       -- 2
	#
	#   --0 0--T--1->2
	#  |       2->3
	# edge--1->0
	#  |
	#   --2->1
    E = torch.tensordot(edge,T,([0],[0]))
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
    a= next(iter(state.sites.values()))
    dims_a = a.size()
    X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device) if op is None else op
    A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

	#   ---------T--2->1 
	#  |         3
	#  |         0
	# edge--0 1--A--3   
	#  |         2
	#   ----1->0
    E = torch.tensordot(E,A,([0,3],[1,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

	#   -------T--1->0
	#  |       |
	#  |       |
	# edge-----A--3->1
	#  |       2
	#  |       2
	#   --0 0--T--1->2
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))

    return E

def apply_TM_2sO(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: two-site operator to be inserted into the two consecutive
               transfer matrices
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with two transfer matrices (and operator ``op``, if any) applied.
             The resulting tensor has an identical index structure as the 
             original ``edge``
    :rtype: torch.tensor
    
    Applies two transfer matrices to the ``edge`` tensor, including the two-site operator
    ``op`` by contracting the following network::

         -----T-------------T------------
        |     |             |
       edge--(a^+ op_l a)==(a^+ op_r a)--
        |     |             |
         -----T-------------T------------

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``op_l`` and ``op_r`` if ``op`` is supplied.
    The ``op_l`` and ``op_r`` are given by the SVD decomposition of two-site operator
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
        op_mat= op.permute(0,2,1,3).contiguous().reshape(dims_op[0]**2,dims_op[0]**2)
        op_l, s, op_r= torch.svd(op_mat)
        op_l= op_l.reshape(dims_op[0],dims_op[0],s.size()[0])
        op_r= torch.einsum('i,ij->ij',s,op_r.t()).reshape(s.size()[0],dims_op[0],dims_op[0])
        op_r= op_r.permute(1,2,0).contiguous()

    T = env.T[env.keyT]
	# Assume index structure of ``edge`` tensor to be as follows
	# 
	#       -- 0
	# edge |-- 1
	#       -- 2
	#
	#   ----0 0--T--1->2
	#  |         2->3
	# edge--1->0
	#  |
	#   ----2->1
    E = torch.tensordot(edge,T,([0],[0]))
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
    a= next(iter(state.sites.values()))
    dims_a = a.size()
    X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device)[:,:,None] if op is None else op_l
    A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

	#   ---------T--2->1 
	#  |         3 4
	#  |         0/
	# edge--0 1--A--3   
	#  |         2
	#   ----1->0
    E = torch.tensordot(E,A,([0,3],[1,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

	#   -------T--1->0
	#  |       | 4->2
	#  |       |/
	# edge-----A--3->1
	#  |       2
	#  |       2
	#   --0 0--T--1->3
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))

    #   ----0 0----T--1->3
	#  |----2->1   2->4
	# edge--1->0
	#  |
	#   ----3->2
    E = torch.tensordot(E,T,([0],[0]))
    if verbosity>0: print("E=ET "+str(E.size()))

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
    X= torch.eye(dims_a[0], dtype=a.dtype, device=a.device)[:,:,None] if op is None else op_r
    A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

	#   ---------T--3->1
	#  |         4
	#  |----1 4-\0
	# edge--0 1--A--3   
	#  |         2
	#   ----2->0
    E = torch.tensordot(E,A,([0,1,4],[1,4,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

	#   -------T--1->0
	#  |       |
	#  |       |
	# edge-----A--3->1
	#  |       2
	#  |       2
	#   --0 0--T--1->2
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))

    return E

def corrf_1sO1sO(state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param op1: first one-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    one-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
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

    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
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
    	corrf[r]= E12/E00

    	# normalize by largest element of E0
    	max_elem_E0 = torch.max(torch.abs(E0))
    	E0=E0/max_elem_E0
    	E1=E1/max_elem_E0

    return corrf

def corrf_2sO2sO_H(state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param op1: first two-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    two-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type state: IPEPS
    :type env: ENV_C4V
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
    :type dist: int
    :type verbosity: int
    :return: vector ``corrf`` of length ``dist`` holding the values of 
             correlation function :math:`\langle O_1(0) O_2(r) \rangle` for :math:`r \in [2,dist+1]`
    :rtype: torch.tensor
    
    Computes the correlation function :math:`\langle O_1(0) O_2(r) \rangle`
    of two horizontaly-oriented two-site operators by contracting 
    the following network::

        C-----T----T------- ... -----T---- ... ----------T----T--------C
        |     |    |                 |                   |    |        |
        |   /-a^+--a^+-\             |                /--a^+--a^+---\  |
        T--< ( op 1 )   >-- ... --(a^+a)-- ... ------< (gen_op2(r)) >--T 
        |   \-a----a---/             |                \--a----a-----/  |
        |     |    |                 |                   |    |        | 
        C-----T----T------- ... -----T---- ... ----------T----T--------C

    for increasingly large distance ``r`` up to ``dist+1``.
    """
    E0 = get_edge(state, env, verbosity=verbosity)
    # Apply double transfer matrix with operator op2
    #
    #   -- 0     ----T-------T---- 0
    #  |        |  --A^+     A^+
    #  |        | /  |--op1--|   \  
    # E1-- 1 = E0----A       A---- 1
    #   -- 2     ----T-------T---- 2
    E1 = apply_TM_2sO(state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_2sO(state, env, E0, verbosity=verbosity)

    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
    for r in range(dist+1):
    	# close the end of the network by appending final double transfer matrix 
    	# with op2
    	#
    	#       C--T--T-- [ --T-- ]^r --T--T--
    	# E12 = T--|O1|-- [ --A-- ]   --|O2|--
    	#       C--T--T-- [ --T-- ]   --T--T--
    	E12= apply_TM_2sO(state, env, E1, op=get_op2(r), verbosity=verbosity)
    	# grow normalization network by single TM
    	E0 = apply_TM_1sO(state, env, E0, verbosity=verbosity)
		# grow network with only a op1 operator by single TM
    	E1 = apply_TM_1sO(state, env, E1, verbosity=verbosity)

    	E12= apply_edge(state, env, E12, verbosity=verbosity)
    	# grow normalization network by additional TM to match the length
    	# of the network with op2 applied
    	E00= apply_TM_1sO(state, env, E0, verbosity=verbosity)
    	E00= apply_edge(state, env, E00, verbosity=verbosity)
    	corrf[r]= E12/E00

    	# normalize by largest element of E0
    	max_elem_E0 = torch.max(torch.abs(E0))
    	E0=E0/max_elem_E0
    	E1=E1/max_elem_E0

    return corrf