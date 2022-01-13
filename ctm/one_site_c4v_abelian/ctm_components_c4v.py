from tn_interface_abelian import contract, permute

def c2x2_dl(A, C, T, verbosity=0):
    #               ----->
    # C--1(-1) (+1)1--T--0->1(+1)
    # 0(-1)           2(-1)
    C2x2= contract(C, T, ([1],[1]))

    #        ---->
    #   C------T--1->0(+1)
    #   0(-1)  2->1(-1)
    # A 0(+1)
    # | T--2->3(-1)
    # | 1->2(+1)
    C2x2 = contract(C2x2, T, ([0],[0]))

    # C---------------T--0(+1)
    # |               1(-1)
    # |               0(+1)
    # T--3(-1) (+1)1--A--3(+1)
    # 2->1(+1)        2(+1)
    C2x2 = contract(C2x2, A, ([1,3],[0,1]))

    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2= C2x2.fuse_legs(axes=((1,2),(0,3)))
    
    # C2x2--1(+1)
    # |
    # 0(+1)
    if verbosity>1: print(C2x2)

    return C2x2

def c2x2_sl(a, C, T, verbosity=0):
    # C--1(-1)(+1)0--T--1(+1)
    # 0(-1)      (-1)2,3
    c2x2= contract(C, T, ([1],[0]))

    # C------T--1->3(+) => C------T--3(+1)
    # 0(-1)  2,3->4,5(-1)  |      4,5
    # 0(+1)                |
    # T--2,3->1,2(-)       T--1,2(-)
    # 1->0(+1)             0(+)
    c2x2= contract(T, c2x2, ([0],[0]))
    # c2x2= permute(c2x2,(0,2,3,1))

    # Open indices connecting Ts to on-site tensor. The unmerged index pairs are ordered 
    # as ket,bra
    #
    # The T tensor is corresponding to the B-sublattice, and hence has switched
    # signatures of legs connecting to on-site tensor: (-) on ket and (+) on bra
    #
    # C-------T--1(+1)
    # |       2->2(-),3(+)
    # T--3->3(-),4(+)->4(-),5(+)
    # |
    # 0(+1)
    # c2x2= c2x2.ungroup_leg(3, T._leg_fusion_data[2])
    # c2x2= c2x2.ungroup_leg(2, T._leg_fusion_data[2])

    # C---------------T--3->2(+1)  => C------------T_|--2->1(+1)
    # |            (-)4 \5->3(+)      |            | \
    # |            (+)1               T------------a-------6->3(+)
    # T----1(-)(+)2---a--4->6(+)      |\      (+)4/|  3(+)
    # |\2->1(+)       |\0->4(+)       | \     (-)0-|-\1(-) 
    # |                3->5(+)        |  \(+)1 (-)2---a*---4->5(-)
    # 0(+1)                           |            |  | 
    #                                 0(+)   (+)2<-5  3->4(-)
    #
    c2x2= contract(c2x2, a, ([4,1], [1,2]))
    c2x2= contract(c2x2, a.conj(), ([3,1,4], [1,2,0]))

    # C-----T--1->3(+1)                  C----T---3(+)
    # |     |                        =>  |    |
    # T----a*a--3(+),5(-)->4(+),5(-)     T---a*a--4(+),5(-)
    # |     |                            |    |
    # 0(+1) 2(+),4(-)->1(+),2(-)         0(+) 1(+),2(-) 
    c2x2= permute(c2x2, (0,2,4,1,3,5))

    return c2x2