import torch
from ipeps.ipeps_c4v import IPEPS_C4V
from ctm.one_site_c4v.env_c4v import ENV_C4V

def get_edge(state, env, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
    #   C--1->0
    #   0
    # A 0
    # | T--2
    # | 1
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

def get_edge_L(state, env, l=1, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param l: length of the edge
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type l: int
    :type verbosity: int
    :return: tensor with indices :math:`\chi \times (D^2)^l \times \chi`
    :rtype: torch.tensor

    Build an edge of length l by contracting a following network::
   
               ---->  
        E = A C--T--...--T--C   |
            | |  |       |  |   V
              0  1       l  l+1
    """
    C= env.C[env.keyC]
    T= env.T[env.keyT]
    #  <----           ---->
    # 0--T--1  =>  0<-1--T--0->2
    #    2               2->1
    T= T.permute(1,2,0).contiguous()
    E= C
    for i in range(1,l+1):
        #           ---->
        #  E-----i 0--T--2->i+1
        # /|\         |
        # ...(i-1)    1->i
        E= torch.tensordot(E,T,([i],[0]))
        if verbosity>0: print("E=C"+i*"T"+f" {E.size()}")
    #  E--l+1 0--C
    # /|\        |
    # ...l       1->l+1   
    E= torch.tensordot(E,C,([l+1],[0]))
    if verbosity>0: print("E=C"+l*"T"+f"C {E.size()}")
    
    return E

def apply_edge(state, env, vec, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
    #  |        1    |
    # vec--0 2--T    |  
    #  |        0->1 V
    #   ---1->0
    S = torch.tensordot(S,T,([0,2],[2,1]))
    if verbosity>0: print("S=ST "+str(S.size()))

    #   -------C
    #  |       |
    # edge-----T
    #  |       1
    #  |       0
    #   --0 1--C
    S = torch.tensordot(S,C,([0,1],[1,0]))
    if verbosity>0: print("S=SC "+str(S.size()))

    return S

#TODO check complex
def apply_edge_L(state, env, vec, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param vec: tensor of dimensions :math:`\chi \times (D^2)^l \times \chi`
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type vec: torch.tensor
    :type verbosity: int
    :return: scalar resulting from the contraction of ``vec`` with an edge
             built from environment
    :rtype: torch.tensor
    
    Contracts ``vec`` tensor with the corresponding edge by contracting 
    the following network::

                     ---0   l+1----C
                 A  |              | |     
        scalar = | vec--1     l----T |
                 |  |    ...       | V
                    |---l     1----T
                     ---l+1     0--C
    """
    l= len(vec.size())-2
    E= get_edge_L(state, env, l=l, verbosity=verbosity)

    inds= list(range(len(vec.size())))
    S = torch.tensordot(vec,E,(inds,inds[::-1]))
    if verbosity>0: print("S "+str(S.size()))

    return S

def apply_TM_1sO(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: operator to be inserted into transfer matrix
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
            
             -->
         -----T----------
        |     |     
       edge--(a^+ op a)--
        |     |     
         -----T----------
             <--

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
    #        ---->  
    #   --0 1--T--0->2
    #  |       2->3
    # edge--1->0
    #  |
    #   --2->1
    E = torch.tensordot(edge,T,([0],[1]))
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
    A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,X,a.conj()).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)

    #          ---->
    #   ---------T--2->1 
    #  |         3
    #  |         0
    # edge--0 1--A--3   
    #  |         2
    #   ----1->0
    E = torch.tensordot(E,A,([0,3],[1,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

    #        ---->
    #   -------T--1->0
    #  |       |
    #  |       |
    # edge-----A--3->1
    #  |       2
    #  |       2
    #   --0 0--T--1->2
    #        <----  
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))

    return E

def apply_TM_1sO_2(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times (D^2)^2 \times \chi`
    :param op: two-site operator to be inserted within the two-site transfer matrix
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type edge: torch.tensor
    :type op: torch.tensor
    :type verbosity: int
    :return: ``edge`` with a single instance of the transfer matrix applied 
             The resulting tensor has an identical index structure as the 
             original ``edge``
    :rtype: torch.tensor
    
    Applies a single instance of the two-site "transfer matrix" to 
    the ``edge`` tensor by contracting the following network, or its corresponding 
    rotation depending on the ``direction``::

                 -----T----------
                |     |          
               edge--(a^+ o1 a)--
                |     |   |      
                |----(a^+ o2 a)--
                |     |          
                 -----T----------

    The two-site operator is first decomposed into a simple MPO o1--o2
    (TODO case where op comes with an extra MPO index)::
        
         s1'  s2'    s1'      s2'
        |  op   | = |o1|-----|o2|
         s1   s2     s1       s2  

    where the physical indices `s` and `s'` of the on-site tensor :math:`a` 
    and it's hermitian conjugate :math:`a^\dagger` are contracted with 
    identity :math:`\delta_{s,s'}` or ``o1``, ``o2``.
    """

    # TODO stronger verification
    op_1, op_2= None, None
    if op is not None:
        if len(op.size())==4:
            # pre-process ``op``
            # TODO possibly truncate/compress according to the vanishingly small singular values
            dims_op= op.size()
            op_mat= op.permute(0,2,1,3).contiguous().reshape(dims_op[0]**2,dims_op[0]**2)
            op_1, s, op_2= torch.svd(op_mat)
            op_1= op_1.reshape(dims_op[0],dims_op[0],s.size()[0])
            op_2= torch.einsum('i,ij->ij',s,op_2.t()).reshape(s.size()[0],dims_op[0],dims_op[0])
            op_2= op_2.permute(1,2,0).contiguous()
        else:
            raise ValueError(f"Invalid op: rank {op.size()}")

    # Four basic cases of passed op
    def get_aXa(a, op):
        # a - on-site tensor
        # op - operator
        dims_a= a.size()
        dims_op= None if op is None else op.size()
        if op is None:
            # identity
            A= torch.einsum('nefgh,nabcd->eafbgchd',a,a.conj()).contiguous()\
                .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)
        elif len(dims_op)==2:
            # one-site operator
            A= torch.einsum('mefgh,mn,nabcd->eafbgchd',a,op,a.conj()).contiguous()\
                .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2)
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
            A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,op,a.conj()).contiguous()\
                .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)
        if verbosity>0: print(f"aXa {A.size()}")
        return A

    a= next(iter(state.sites.values()))
    T= env.T[env.keyT]
    # Assume index structure of ``edge`` tensor to be as follows
    # 
    #       -- 0
    # edge |-- 1
    #      |---2
    #       -- 3
    #
    #          ---->
    #   ----0 1--T--0
    #  |         2->1
    # edge--1->2
    #  |
    #   ----2->3
    #  |
    #   ----3->4
    E = torch.tensordot(T,edge,([1],[0]))
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
    A= get_aXa(a, op_1)

    #   ---------T--0
    #  |         1
    #  |         0
    # edge--2 1--A--3->4
    #  |      3<-2 \
    #   ----3->1   (4->5)
    #  |            
    #   ----4->2
    E = torch.tensordot(E,A,([1,2],[0,1]))
    if verbosity>0: print("E=edgeTA "+str(E.size()))

    A= get_aXa(a, op_2)
    #   ---------T--0
    #  |         |
    # edge-------A--4->2
    #  |         | \
    #  |         3 (5)
    #  |         0 (4)
    #  |         | /     
    #   ----1 1--A--2->3
    #  |         3->4   
    #   ----2->1
    E = torch.tensordot(E,A,([1,3],[1,0])) if op is None else \
        torch.tensordot(E,A,([1,3,5],[1,0,4]))
    if verbosity>0: print("E=edgeTAA "+str(E.size()))

    #          ---->
    #   ---------T--0
    #  |         |
    # edge-------A--2->1
    #  |         |   
    #   ---------A--3->2
    #  |         3
    #  |         2   
    #   ----1 0--T2--1->3
    #          <-----
    E = torch.tensordot(E,T,([1,3],[0,2]))
    if verbosity>0: print("E=edgeTAAT "+str(E.size()))

    return E

def apply_TM_2sO(state, env, edge, op=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param edge: tensor of dimensions :math:`\chi \times D^2 \times \chi`
    :param op: two-site operator to be inserted into the two consecutive
               transfer matrices
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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
    #          ---->
    #   ----0 1--T--0->2
    #  |         2->3
    # edge--1->0
    #  |
    #   ----2->1
    E = torch.tensordot(edge,T,([0],[1]))
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
    A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a.conj()).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

    #   ---------T--2->1 
    #  |         3 4
    #  |         0/
    # edge--0 1--A--3   
    #  |         2
    #   ----1->0
    E = torch.tensordot(E,A,([0,3],[1,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

    #        ----> 
    #   -------T--1->0
    #  |       | 4->2
    #  |       |/
    # edge-----A--3->1
    #  |       2
    #  |       2
    #   --0 0--T--1->3
    #        <----
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))
                
    #            ---->
    #   ----0 1----T--0->3
    #  |----2->1   2->4
    # edge--1->0
    #  |
    #   ----3->2
    E = torch.tensordot(E,T,([0],[1]))
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
    A= torch.einsum('mefgh,mnl,nabcd->eafbgchdl',a,X,a.conj()).contiguous()\
        .view(dims_a[1]**2, dims_a[2]**2, dims_a[3]**2, dims_a[4]**2, -1)

    #          ---->
    #   ---------T--3->1
    #  |         4
    #  |----1 4-\0
    # edge--0 1--A--3   
    #  |         2
    #   ----2->0
    E = torch.tensordot(E,A,([0,1,4],[1,4,0]))
    if verbosity>0: print("E=EA "+str(E.size()))

    #        ----> 
    #   -------T--1->0
    #  |       |
    #  |       |
    # edge-----A--3->1
    #  |       2
    #  |       2
    #   --0 0--T--1->2
    #        <----
    E = torch.tensordot(E,T,([0,2],[0,2]))
    if verbosity>0: print("E=ET "+str(E.size()))

    return E

def corrf_1sO1sO(state, env, op1, get_op2, dist, rl_0=None, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param op1: first one-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    one-site opreator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param rl_0: right and left edges of the two-point function network. These
                 are expected to be rank-3 tensor compatible with transfer operator indices.
                 Typically provided by leading eigenvectors of transfer matrix.
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
    :type dist: int
    :type rl_0: tuple(torch.Tensor, torch.Tensor)
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
    E0 = get_edge(state, env, verbosity=verbosity) if rl_0 is None else rl_0[0]
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

        if rl_0 is None:
            E12= apply_edge(state, env, E12, verbosity=verbosity)
            E00= apply_edge(state, env, E0, verbosity=verbosity)
        else:
            E12= torch.tensordot(E12,rl_0[1],([0,1,2],[0,1,2]))
            E00= torch.tensordot(E0,rl_0[1],([0,1,2],[0,1,2]))
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = torch.max(torch.abs(E0))
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf

def corrf_2sOH2sOH_E1(state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param state: underlying 1-site C4v symmetric wavefunction
    :param env: C4v symmetric environment corresponding to ``state``
    :param op1: first two-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    two-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
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

def corrf_2sOV2sOV_E2(state, env, op1, get_op2, dist, verbosity=0):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param op1: first two-site operator :math:`O_1`
    :param get_op2: function returning (position-dependent) second
                    two-site operator :math:`\text{get_op2}(r)=O_2`
    :param dist: maximal distance of correlation function
    :param verbosity: logging verbosity
    :type state: IPEPS_C4V
    :type env: ENV_C4V
    :type op1: torch.tensor
    :type get_op2: function(int)->torch.tensor
    :type dist: int
    :type verbosity: int
    :return: vector ``corrf`` of length ``dist`` holding the values of 
             correlation function :math:`\langle O_1(0) O_2(r) \rangle` 
             for :math:`r \in [1,dist]`
    :rtype: torch.tensor
    
    Computes the four-point correlation function :math:`\langle O_1(0) O_2(r) \rangle`,
    where both O_1 and O_2 are two-site operators by contracting the following network::

        C-----T------ ... ----T----- ... -----T---------C
        |     |               |               |         |
        T--(a^+--a)-- ... --(a^+a)-- ... --(a^+--a)-----T
        |    op1              |           (gen_op2(r))  |
        T--(a^+--a)-- ... --(a^+a)-- ... --(a^+--a)-----T
        |     |               |               |         | 
        C-----T------ ... ----T----- ... -----T---------C

    for increasingly large distance ``r`` up to ``dist``.
    """
    E0 = get_edge_L(state, env, l=2, verbosity=verbosity)
    # Apply transfer matrix with operator op1
    #
    #   -- 0     -- -----T--------- 0
    # E1-- 1 = E0-- --(a^+ op1 a)-- 1
    #   -- 2     -- --(a^+ op1 a)-- 2
    #   -- 3     -- -----T--------- 3
    E1 = apply_TM_1sO_2(state, env, E0, op=op1, verbosity=verbosity)
    # used for normalization
    E0 = apply_TM_1sO_2(state, env, E0, verbosity=verbosity)

    corrf=torch.empty(dist+1,dtype=state.dtype,device=state.device)
    for r in range(dist+1):
        # close the end of the network by appending final transfer matrix 
        # with op2
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        E12= apply_TM_1sO_2(state, env, E1, op=get_op2(r), verbosity=verbosity)
        # and corresponding normalization network
        E0 = apply_TM_1sO_2(state, env, E0, verbosity=verbosity)
        # and to network with only a op1 operator
        E1 = apply_TM_1sO_2(state, env, E1, verbosity=verbosity)

        # and finally apply edge of opposite direction
        E12= apply_edge_L(state, env, E12, verbosity=verbosity)
        E00= apply_edge_L(state, env, E0, verbosity=verbosity)
        corrf[r]= E12/E00

        # normalize by largest element of E0
        max_elem_E0 = torch.max(torch.abs(E0))
        E0=E0/max_elem_E0
        E1=E1/max_elem_E0

    return corrf