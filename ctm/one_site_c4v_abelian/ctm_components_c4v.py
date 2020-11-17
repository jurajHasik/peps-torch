from tn_interface_abelian import contract

def c2x2_dl(A, C, T, verbosity=0):
    # # C--1(-1) (+1)0--T--1(+1)
    # # 0(-1)           2(-1)
    # C2x2= contract(C, T, ([1],[0]))
    # # C2x2= contract(C, T.conj().change_signature((1,1,-1)), ([1],[0]))

    # # C--1(-1) (+1)1--T--0->1(+1)
    # # 0(-1)           2(-1)
    # # C2x2 = contract(C, T, ([1],[1]))

    # # C------T--1->0(+1)
    # # 0(-1)  2->1(-1)
    # # 0(+1)
    # # T--2->3(-1)
    # # 1->2(+1)
    # C2x2 = contract(C2x2, T, ([0],[0]))

    # # C---------------T--0(+1)
    # # |               1(-1)
    # # |               0(+1)
    # # T--3(-1) (+1)1--A--3(+1) 
    # # 2->1(+1)        2(+1)
    # C2x2 = contract(C2x2, A, ([1,3],[0,1]))
    
    lo0, lo1= C._leg_fusion_data[0], C._leg_fusion_data[1]  
    C= C.ungroup_leg(1, lo1)
    C= C.ungroup_leg(0, lo0)

    lo0, lo1, lo_aux= T._leg_fusion_data[0], T._leg_fusion_data[1], T._leg_fusion_data[2] 
    T= T.ungroup_leg(2, lo_aux)
    T= T.ungroup_leg(1, lo1)
    T= T.ungroup_leg(0, lo0)

    T= T.change_signature((-1,1, 1,-1, 1,-1))
    # C--2(+1),3(-1) 0(+1->-1),1(-1->+1)--T--2(+1),3(-1) 
    # |                                   |   
    # 0(+1),1(-1)                         4(+1),5(-1)
    C2x2= contract(C,T, ([2,3],[0,1]))

    # C-------------T--2->0(+1),3->1(-1) 
    # |             |
    # 0(+1),1(-1)   4->2(+1),5->3(-1)
    # 0(+1->-1),1(-1->+1)
    # |
    # T--4->6(+1),5->7(-1)
    # |
    # 2->4(+1),3->5(-1)
    C2x2= contract(C2x2,T, ([0,1],[0,1]))

    lo0, lo1, lo2, lo3= A._leg_fusion_data[0],A._leg_fusion_data[1],A._leg_fusion_data[2],A._leg_fusion_data[3]
    A= A.ungroup_leg(1, lo1)
    A= A.ungroup_leg(0, lo0)

    C2x2= C2x2.change_signature((1,-1, -1,1, 1,-1, -1,1))
    # C-----------------------------------T--0(+1),1(-1) 
    # |                                   |
    # |                                   2(+1->-1),3(-1->+1)
    # |                                   0(+1),1(-1)
    # |                                   |
    # T--6(+1->-1),7(-1->+1) 2(+1),3(-1)--A--5(+1)
    # |                                   |
    # 4->2(+1),5->3(-1)                   4(+1)
    C2x2 = contract(C2x2, A, ([2,3,6,7],[0,1,2,3]))

    # CT    |--0,1
    # TA____|--5
    # |    |
    # 2,3  4
    C2x2= C2x2.transpose((2,3,4, 0,1,5))
    C2x2, lo1= C2x2.group_legs((3,4,5), new_s=-1)
    C2x2, lo0= C2x2.group_legs((0,1,2), new_s=-1)

    # permute 0123->1203
    # reshape (12)(03)->01
    # C2x2= C2x2.transpose((1,2,0,3))
    # C2x2, lo1= C2x2.group_legs((2,3), new_s=-1)
    # C2x2, lo0= C2x2.group_legs((0,1), new_s=-1)
    C2x2._leg_fusion_data[1]= lo1
    C2x2._leg_fusion_data[0]= lo0

    # C2x2--1(-1)
    # |
    # 0(-1)
    if verbosity>1: print(C2x2)

    return C2x2

def c2x2_sl(a, C, T, verbosity=0):
    # C--1 0--T--1
    # 0       2
    C2x2 = torch.tensordot(C, T, ([1],[0]))

    # C------T--1->0
    # 0      2->1
    # 0
    # T--2->3
    # 1->2
    C2x2 = torch.tensordot(C2x2, T, ([0],[0]))

    # C-------T--0
    # |       1
    # |       0
    # T--3 1--A--3 
    # 2->1    2
    # C2x2 = torch.tensordot(C2x2, A, ([1,3],[0,1]))

    # 4) double-layer tensor contraction - layer by layer
    # 4i) untangle the fused D^2 indices
    # 
    # C-----T--0
    # |     1->1,2
    # |  
    # T--3->4,5
    # 2->3
    C2x2= C2x2.view(C2x2.size()[0],a.size()[1],a.size()[1],C2x2.size()[2],\
        a.size()[2],a.size()[2])

    # 4ii) first layer "bra" (in principle conjugate)
    # 
    # C---------T----0
    # |         |\---2->1
    # |         1    
    # |         1 /0->4
    # T----4 2--a--4->6 
    # | |       3->5
    # |  --5->3
    # 3->2
    C2x2= torch.tensordot(C2x2, a,([1,4],[1,2]))

    # 4iii) second layer "ket"
    # 
    # C----T----------0
    # |    |\-----\
    # |    |       1
    # |    |/4 0\  |
    # T----a----------6->3 
    # | |  |      \1
    # |  -----3 2--a--4->5
    # |    |       3->4
    # |    |
    # 2->1 5->2
    C2x2= torch.tensordot(C2x2, a,([1,3,4],[1,2,0]))

    # 4iv) fuse pairs of aux indices
    #
    # C----T----0
    # |    |\
    # T----a----3\ 
    # | |  |\|    ->3
    # |  ----a--5/
    # |    | |
    # 1   (2 4)->2 
    # 
    # and simultaneously
    # permute 0123->1203
    # reshape (12)(03)->01
    C2x2= C2x2.permute(1,2,4,0,3,5).contiguous().view(C2x2.size()[1]*(a.size()[3]**2),\
        C2x2.size()[0]*(a.size()[4]**2))

    # C2x2--1
    # |
    # 0
    if verbosity>1: print(C2x2)

    return C2x2