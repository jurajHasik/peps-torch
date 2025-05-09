import itertools
import numpy as np
import torch
import yastn.yastn as yastn
from tn_interface_abelian import contract, permute
try:
    from scipy.sparse.linalg import LinearOperator
    from scipy.sparse.linalg import eigs
    from scipy.sparse.linalg import eigsh
except:
    print("Warning: Missing scipy. ARNOLDISVD is not available.")
from ctm.one_site_c4v_abelian import corrf_c4v


def get_Top_spec_c4v(n, state, env_c4v, verbosity=0):
    raise NotImplementedError()
    # 0) build edge and get symmetric structure
    # (+)0--
    # (-)1--E 
    # (+)2--
    E= corrf_c4v.get_edge(state, env_c4v, verbosity)

    # get symmetry structure of the (grouped) edge and create a dummy Nx1 vector
    # for serialization
    # (+)--M--(+) (-)--V0--(+) = (+)--V1--(+)
    Cs, Ds= (l.t for l in E.get_legs(native=True)), (l.D for l in E.get_legs(native=True))
    # get all the possible sectors given by fusion of E-legs
    Cs1= E.s[0]*np.asarray(Cs[0])
    for x in [s*np.asarray(c) for s,c in zip(E.s[1:], Cs[1:])]:
        X= np.add.outer(Cs1, x)
        Cs1= np.unique(X.reshape(len(Cs1)*len(x),1),axes=0)
    Cs1= tuple(map(tuple,Cs1))

    # build dummy Nx1 vector
    Cs= Cs + (Cs1,)
    Ds= Ds + (tuple([1]*len(Cs1)),)
    v0= yastn.zeros(config= E.config, s=np.concatenate((E.s,[1])), n=0, t=Cs, D=Ds)
    # r1d, meta= v0.compress_to_1d()
    r1d, meta= E.compress_to_1d()
    N= r1d.numel()

    # multiply vector by transfer-op and pass the result back to numpy
    # Assume the following index structure
    #  --0 (chi)
    # v--1 (D^2)
    #  --2 (chi)
    def _mv(v):
        V1d= torch.as_tensor(v, dtype=r1d.dtype, device=r1d.device)
        # bring 1d vector into edge structure
        # (+)0--
        # (-)1--E--(-) dummy-index
        # (+)2--
        # pdb.set_trace()
        V= yastn.decompress_from_1d(V1d, E.config, meta)
        V= corrf_c4v.apply_TM_1sO(state,env_c4v,V,verbosity=verbosity)
        # V= V.flip_signature(inplace=True)
        # bring the edge back into plain 1d representation
        V1d, _meta= V.compress_to_1d()
        return V1d.detach().cpu().numpy()

    T= LinearOperator((N,N), matvec=_mv)
    # vals= eigs(T, k=n, v0=None, return_eigenvectors=False)
    vals= eigsh(T, k=n, return_eigenvectors=False) 

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    vals= np.copy(vals[::-1]) # descending order
    vals= (1.0/np.abs(vals[0])) * vals

    L= torch.zeros((n,2), dtype=r1d.dtype, device=r1d.device)
    L[:,0]= torch.as_tensor(np.real(vals))
    L[:,1]= 0
    # L[:,1]= torch.as_tensor(np.imag(vals))

    return L


def get_full_EH_spec_exactTM(state):

    def get_max_ev(S):
        ev_list=[]
        for c,b in S.A.items():
            ev_list.extend( [ [b[i].item(), c] for i in range(len(b))] )
        # find largest spectral value across all sectors
        ev_list.sort(key=lambda x: abs(x[0]), reverse=True)
        return ev_list[0]

    def print_EVs(S,normalize=True):
        # print spectrum (value, charge)
        ev_list=[]
        for c,b in S.A.items():
            ev_list.extend( [ [b[i].item(), c] for i in range(len(b))] )
        # find largest spectral value across all sectors
        ev_list.sort(key=lambda x: abs(x[0]), reverse=True)
        max_ev= ev_list[0][0]
        if normalize:
            for i in range(len(ev_list)): ev_list[i][0]= ev_list[i][0]/max_ev 
        for ev in ev_list: print(ev)
    
    def get_eigenvecs(U,S,V,c,ev_ind):
        # given sector charge and eigenvalue index, retrieve left and right eigenvectors
        Pr= yastn.zeros(config=S.config, s=S.get_signature(), 
            n=S.get_tensor_charge(), t=list(zip(c)), D=[1]+list(S[c].size()) )
        Pl= yastn.zeros(config=S.config, s=S.get_signature(), 
            n=S.get_tensor_charge(), t=list(zip(c)), D=list(S[c].size())+[1] )
        
        Pr[c][0,ev_ind]=1.0
        Pl[c][ev_ind,0]=1.0

        R= yastn.tensordot(Pr,V,([1],[0]))
        L= yastn.tensordot(U,Pl,([1],[0]))
        return L,R

    def get_leading_ev(U,S,V,c=None,ev_ind=0,verbosity=0):
        # if c is provided, find the leading eigenvectors in the charge c sector
        # otherwise, find the leading eigenvalue across all sectors and return
        # its eigenvectors

        # report largest eigenvalues in each sector and the dimension of the sector
        if verbosity>0:
            highest_evs= [ (c,max(S[c]),S[c].size()) for c in S.A.keys() ]
            highest_evs.sort(key=lambda x: x[1], reverse=True)
            for row in highest_evs: print(f"{row[0]} {row[1]} {row[2]}")

        # get charge c_max of the sector, which contains the leading eigenvalue
        max_val=-np.inf
        c_max= c
        if c_max is None:
            for c in S.A.keys():
                if max_val<max(S[c]): c_max, max_val=c, max(S[c])
        if verbosity>0:
            print(S[c_max])

        L,R= get_eigenvecs(U,S,V,c=c_max,ev_ind=ev_ind)
        return L,R,c_max

    # compute exact transfer matrix
    # get double layer tensor
    a_dl= yastn.tensordot(state.site(),state.site(),([0],[0]),conj=(0,1))
    a_dl_fused= a_dl.fuse_legs( axes=((0,4),(1,5),(2,6),(3,7)) )
    
    # build open transfer matrix, unmerge fused (bra,ket) pairs and fuse into
    # indices accumulating bra and ket
    #
    #    -0                0                    0
    # -2--A-- -4  =>  2,3--A--6,7  =>  2<-2,4--|A|--6,8->4 (bra)
    #     1                |           3<-3,5--|A|--7,9->5 (ket)
    #     1                |                    1
    # -3--A-- -5      4,5--A--8,9
    #    -1                1
    #
    TM2_open= yastn.ncon([a_dl_fused,a_dl_fused.flip_signature()],[[-0,-2,1,-4],[1,-3,-1,-5]])
    TM2_open= TM2_open.unfuse_legs(axes=(2,3,4,5))
    TM2_open= TM2_open.fuse_legs(axes=(0,1,(2,4),(3,5),(6,8),(7,9)))

    # Decompose
    # 
    #    1
    #   /A\
    # -0 | -1
    #   \A/
    #    1
    TM2_closed= yastn.ncon([TM2_open.fuse_legs(axes=(0,1,(2,3),(4,5)))],[[1,1,-0,-1]])
    print(f"TM2_closed size {TM2_closed.size}")
    # signatures on left & right legs are identical
    U,S,V= TM2_closed.svd((0,1))

    import pdb; pdb.set_trace()

    L,R,c_max= get_leading_ev(U,S,V,verbosity=1)
    #
    #     0                   0
    #  1--R => 1(bra),2(ket)--R => 0<-0,1--R--2 <=> 0--exp(EH)--1
    #
    # leg 0 is dummy leg, carrying the charge
    # 
    R= R.unfuse_legs(axes=1)
    R= R.fuse_legs(axes=((0,1),2))
    _,Sr,_= R.svd((0,1))
    # L= L.unfuse_legs(axes=0)
    # L= L.unfuse_legs(axes=(0,1))
    # L= L.fuse_legs(axes=((0,2),(1,3,4)))
    # _,Sl,_= L.svd((0,1))

    import pdb; pdb.set_trace()

    print_EVs(Sr)
    # print_EVs(Sl)

    import pdb; pdb.set_trace()

    for i in range(1,S[c_max].size(0)):
        L,R= get_eigenvecs(U,S,V,c_max,i)
        R= R.unfuse_legs(axes=1)
        R= R.fuse_legs(axes=((0,1),2))
        _,Sr,_= R.svd((0,1))

        ev_val, ev_charge= get_max_ev(Sr)
        print(f"{i} TM_ev= {S[c_max][i]} ev= {ev_val} {ev_charge}")
        if ev_charge[0]%2!=0:
            print_EVs(Sr)
            break

    #    0            0
    # 2--A--4  =>  2--A--3
    # 3--A--5         A
    #    1            1 
    TM2_open= TM2_open.fuse_legs(axes=(0,1,(2,3),(4,5)))

    #     1
    # -0--A-- -2  =>  0<-0,1--A--2,3->1
    #     A                   A
    #     2                   A
    #     2                   A   
    # -1--A-- -3
    #     A
    #     1
    TM4_closed= yastn.ncon([TM2_open,TM2_open],[[1,2,-0,-2],[2,1,-1,-3]])
    TM4_closed= TM4_closed.fuse_legs(axes=((0,1),(2,3)))
    U,S,V= TM4_closed.svd((0,1))
    print(f"TM4_closed size {TM4_closed.size}")
    L,R,c_max= get_leading_ev(U,S,V,verbosity=1)
    #
    #    0         0                         0    
    # 1--R => 1,2--R => 1(b),2(k),3(b),4(k)--R => 0<-(0,1,3)--R--(2,4)->1 
    #                    
    R= R.unfuse_legs(axes=1)
    R= R.unfuse_legs(axes=(1,2))
    R= R.fuse_legs(axes=((0,1,3),(2,4)))
    _,Sr,_= R.svd((0,1))
    print_EVs(Sr)

    import pdb; pdb.set_trace()

    for i in range(1,S[c_max].size(0)):
        L,R= get_eigenvecs(U,S,V,c_max,i)
        R= R.unfuse_legs(axes=1)
        R= R.unfuse_legs(axes=(1,2))
        R= R.fuse_legs(axes=((0,1,3),(2,4)))
        _,Sr,_= R.svd((0,1))

        ev_val, ev_charge= get_max_ev(Sr)
        print(f"{i} TM_ev= {S[c_max][i]} ev= {ev_val} {ev_charge}")
        if ev_charge[0]%2!=0:
            print_EVs(Sr)
            break

    #     1         0
    # -0--A-- -3 -> A--1-> 1,2
    #     A         A--2-> 3,4
    #     2
    #     2
    # -1--A-- -4    A--3-> 5,6
    #     A         A--4
    #     3
    #     3
    # -2--A-- -5    A--5
    #     A         A--6-> 11,12
    #     1
    # TM6_closed= yastn.ncon([TM2_open,TM2_open,TM2_open],[[1,2,-0,-3],[2,3,-1,-4],[3,1,-2,-5]])
    # TM6_closed= TM6_closed.fuse_legs(axes=((0,1,2),(3,4,5)))
    # U,S,V= TM6_closed.svd((0,1))
    # L,R= get_leading_ev(U,S,V)
    # R= R.unfuse_legs(axes=1)
    # R= R.unfuse_legs(axes=(1,2,3))
    # R= R.unfuse_legs(axes=(1,2,3,4,5,6))
    # R= R.fuse_legs(axes=((0,1,3,5,7,9,11),(2,4,6,8,10,12)))

    # _,Sl,_= R.svd((0,1))
    # print_EVs(Sl)

    # for i in range(1,19):
    #     L,R= get_leading_ev(U,S,V, c=(0,0), ev_ind=i)
    #     R= R.unfuse_legs(axes=1)
    #     R= R.unfuse_legs(axes=(1,2,3))
    #     R= R.unfuse_legs(axes=(1,2,3,4,5,6))
    #     R= R.fuse_legs(axes=((0,1,3,5,7,9,11),(2,4,6,8,10,12)))
    #     _,Sr_c00_1,_= R.svd((0,1))

    #     print(f"{i} ",end="")
    #     print_EVs(Sr_c00_1)