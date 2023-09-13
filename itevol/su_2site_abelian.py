import yastn.yastn as yastn
from tn_interface_abelian import contract

def run_seq_2s(state, gate_seq, su_opts={"weight_inv_cutoff": 1.0e-14, \
    "max_D_total": 1, "log_level": 0}):
    for bond,gate in gate_seq:
        apply_gate_2s(state,bond,gate,su_opts)

    return state

def truncated_svd(M, axes, sU=1, D_total=2**32):

    def truncation_f(S):
        return yastn.linalg.truncation_mask_multiplets(S, keep_multiplets=True, D_total=D_total,\
            tol=1.0e-8, tol_block=1.0e-14, eps_multiplet=1.0e-10)

    return yastn.linalg.svd_with_truncation(M, axes, sU=sU, mask_f=truncation_f)

def apply_gate_2s(state,bond,gate,su_opts):
    # dxy_w_s1s2 = xy_s2-xy_s1
    xy_s1, dxy_w_s1s2, xy_s2= bond
    dxy_w_s2s1= (-dxy_w_s1s2[0],-dxy_w_s1s2[1])

    weight_inv_cutoff= su_opts["weight_inv_cutoff"]
    max_D_total= su_opts["max_D_total"]
    log_level= su_opts["log_level"]

    # get outer weights of t_s1 and t_s2
    dxy_w_to_ind= dict({(0,-1): 1, (-1,0): 2, (0,1): 3, (1,0): 4})
    full_dxy=set(dxy_w_to_ind.keys())
    outer_w_dxy_s1= full_dxy - set(( dxy_w_s1s2, ))
    outer_w_dxy_s2= full_dxy - set(( dxy_w_s2s1, ))

    # absorb outer weights into t_s1 and t_s2
    A= state.site(xy_s1)
    for dxy_w in outer_w_dxy_s1:
        w= state.weight((xy_s1, dxy_w))
        _match_diag_signature= 1 if -w.get_signature()[1]==A.get_signature()[dxy_w_to_ind[dxy_w]] else 0
        A= contract(w, A, ([_match_diag_signature],[dxy_w_to_ind[dxy_w]]))
        #                                0  1  2  ....     N     1  2      0      N
        # jk,ik * i0,i1,...,ik,...,iN -> jk,i0,i1,...,,...,iN -> i0,i1,...,jk,...,iN 
        ind_l= list(range(1,A.ndim))
        ind_l.insert(dxy_w_to_ind[dxy_w], 0)
        A= A.transpose(ind_l)
        #print(f"{(xy_s1, dxy_w)} {([1],[dxy_w_to_ind[dxy_w]])} {ind_l}")

    B= state.site(xy_s2)
    for dxy_w in outer_w_dxy_s2:
        w= state.weight((xy_s2, dxy_w))
        _match_diag_signature= 1 if -w.get_signature()[1]==B.get_signature()[dxy_w_to_ind[dxy_w]] else 0
        B= contract(w,B, ([_match_diag_signature],[dxy_w_to_ind[dxy_w]]))

        ind_l= list(range(1,B.ndim))
        ind_l.insert(dxy_w_to_ind[dxy_w], 0)
        B= B.transpose(ind_l)
        # print(f"{(xy_s2, dxy_w)} {([1],[dxy_w_to_ind[dxy_w]])} {ind_l}")

    outer_inds_s1= set([1,2,3,4]) - set((dxy_w_to_ind[dxy_w_s1s2],))
    outer_inds_s2= set([1,2,3,4]) - set(( dxy_w_to_ind[dxy_w_s2s1], ))
    # get reduced on-site tensors
    #
    #   |     |                 |      |
    # --A--W--B-- -> --xA--SA--rA--W--rB--SB--xB--
    #
    # choice of signature
    #                                               1   
    #   |                                           |
    # --A--(s) -> --xA--(s) 0(-s)--SA--(s)1 0(-s)--rA--(s)2
    #                   
    #                    0
    #      |             |                          
    # (s)--B-- -> 1(s)--rB--(-s)2 0(s)--SB--(-s)1 0(s)--xB--
    W= state.weight((xy_s1, dxy_w_s1s2))
    xA, SA, rA= yastn.linalg.svd(A, (list(outer_inds_s1),[0, dxy_w_to_ind[dxy_w_s1s2]]), \
        sU=-A.get_signature()[dxy_w_to_ind[dxy_w_s1s2]])
    rB, SB, xB= yastn.linalg.svd(B, ([0, dxy_w_to_ind[dxy_w_s2s1]], \
        list(outer_inds_s2)), sU=-B.get_signature()[dxy_w_to_ind[dxy_w_s2s1]])

    # contract
    #                    0      1 
    #                    |______|
    #                   |____g___|
    #                    2      3
    #    | |             |      |  
    # --|_M_|-- = --SA--rA--W--rB--SB--
    M= contract(SA,rA,([1],[0]))
    _match_diag_signature= 1 if -W.get_signature()[1]==M.get_signature()[2] else 0
    M= contract(M,W,([2],[_match_diag_signature]))
    rB= contract(rB,SB,([2],[0]))
    #        1             0->2                1       2
    # 0--(SA-rA-W)--2 1--(rB-SB)--2->3 -> 0--(SA-rA-W-rB-SB)--3
    M= contract(M,rB,([2],[1]))
    #         0_______1
    #        |_________|
    #         2       3
    #         1       2
    # 0->2--(SA-rA-W-rB-SB)--3
    M= contract(gate,M,([2,3],[1,2]))

    # split and truncate
    #     0              1
    #     |              |
    # 1--nA--2 --W-- 0--nB--2
    nA, W, nB= truncated_svd(M, ([0,2],[1,3]), sU=A.get_signature()[dxy_w_to_ind[dxy_w_s1s2]],\
        D_total=max_D_total)

    # normalize new weight
    W= W / W.norm(p="inf")

    # reabsorb nA, nB back to xA, xB
    #                           0
    # (other)--xA--(ndim-1) 1--nA--2->1 -> 
    A= contract(nA,xA, ([1],[xA.ndim-1]))
    l_ind= [0]+list(range(2,xA.ndim+1))
    l_ind.insert(dxy_w_to_ind[dxy_w_s1s2],1)
    A= A.transpose(l_ind)
    # print(f"{dxy_w_to_ind[dxy_w_s1s2]} {l_ind}")

    #     1           0
    # 0--nB--2 -> 1--nB--2 0--xB--(other) 
    nB= nB.transpose((1,0,2))
    B= contract(nB,xB,([2],[0]))
    l_ind= [0]+list(range(2,xB.ndim+1))
    l_ind.insert(dxy_w_to_ind[dxy_w_s2s1],1)
    B= B.transpose(l_ind)
    # print(f"{dxy_w_to_ind[dxy_w_s2s1]} {l_ind}")

    # apply inverse of the previous weights
    for dxy_w in outer_w_dxy_s1:
        w= state.weight((xy_s1, dxy_w)).reciprocal(cutoff=weight_inv_cutoff)
        _match_diag_signature= 1 if -w.get_signature()[1]==A.get_signature()[dxy_w_to_ind[dxy_w]] else 0
        A= contract(w,A, ([_match_diag_signature],[dxy_w_to_ind[dxy_w]]))
        #                                0  1  2  ....     N     1  2      0      N
        # jk,ik * i0,i1,...,ik,...,iN -> jk,i0,i1,...,,...,iN -> i0,i1,...,jk,...,iN 
        ind_l= list(range(1,A.ndim))
        ind_l.insert(dxy_w_to_ind[dxy_w], 0)
        A= A.transpose(ind_l)
        # print(f"{(xy_s1, dxy_w)} {([1],[dxy_w_to_ind[dxy_w]])} {ind_l}")

    for dxy_w in outer_w_dxy_s2:
        w= state.weight((xy_s2, dxy_w)).reciprocal(cutoff=weight_inv_cutoff)
        _match_diag_signature= 1 if -w.get_signature()[1]==B.get_signature()[dxy_w_to_ind[dxy_w]] else 0
        B= contract(w,B, ([_match_diag_signature],[dxy_w_to_ind[dxy_w]]))
        ind_l= list(range(1,B.ndim))
        ind_l.insert(dxy_w_to_ind[dxy_w], 0)
        B= B.transpose(ind_l)
        # print(f"{(xy_s2, dxy_w)} {([1],[dxy_w_to_ind[dxy_w]])} {ind_l}")

    state.sites[ state.vertexToSite(xy_s1) ]= A
    state.sites[ state.vertexToSite(xy_s2) ]= B
    state.weights[ (state.vertexToSite(xy_s1),dxy_w_s1s2) ]= W
    state.weights[ (state.vertexToSite(xy_s2),dxy_w_s2s1) ]= W

    return state