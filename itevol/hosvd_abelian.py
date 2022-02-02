import yast.yast as yast
from tn_interface_abelian import contract, permute, conj

def hosvd(A,itebd_tol,bond_dim,keep_multiplet):
    #A is n(1) x ... x n(d) tensor
    #U is d cell array with U(k) being the left singular vector of a's mode-k unfolding
    #S is n(1) x ... x n(d) tensor : A x1 U(1) x2 U(2) ... xd U(d)
    S = A
    U_set = []
    lambda_set = []

    #index 0
    #AA=torch.einsum('abc,dbc->ad', A, A.conj())
    #AA=yast.ncon([A,A.conj()],[[-1+1,0+1,1+1],[-2+1,0+1,1+1]])
    #u,lamb,_=torch.svd(AA)
    u, lamb, V=yast.svd(A, axes=((0,1),(2,3,4,5)), sU=1, tol=itebd_tol, \
        D_total=bond_dim, untruncated_S=False, keep_multiplets=keep_multiplet)
    lamb=lamb/lamb.norm(p='inf')
    #print(lamb.to_dense())
    U_set.append(u)
    lambda_set.append(lamb)
    #S=torch.einsum('ab,bcd->acd', u.t().conj(), S)
    S=yast.ncon([u.conj(),S],[[1+1,2+1,-1+1],[1+1,2+1,-2+1,-3+1,-4+1,-5+1]])

    #index 1
    #AA=torch.einsum('bac,bdc->ad', A, A.conj())
    #AA=yast.ncon([A,A.conj()],[[0+1,-1+1,1+1],[0+1,-2+1,1+1]])
    #u,lamb,_=torch.svd(AA)
    u, lamb, V=yast.svd(A.transpose(axes=(2,3,0,1,4,5)), axes=((0,1),(2,3,4,5)), sU=1,\
        tol=itebd_tol, D_total=bond_dim, untruncated_S=False, keep_multiplets=keep_multiplet)
    lamb=lamb/lamb.norm(p='inf')
    U_set.append(u)
    lambda_set.append(lamb)
    #S=torch.einsum('ac,bcd->bad', u.t().conj(), S)
    S=yast.ncon([u.conj(),S],[[1+1,2+1,-2+1],[-1+1,1+1,2+1,-3+1,-4+1]])

    #index 2
    #AA=torch.einsum('bca,bcd->ad', A, A.conj())
    #AA=yast.ncon([A,A.con()],[[0+1,1+1,-1+1],[0+1,1+1,-2+1]])
    #u,lamb,_=torch.svd(AA)
    u, lamb, V=yast.svd(A.transpose(axes=(4,5,0,1,2,3)), axes=((0,1),(2,3,4,5)), sU=1,\
        tol=itebd_tol, D_total=bond_dim, untruncated_S=False, keep_multiplets=keep_multiplet)
    lamb=lamb/lamb.norm(p='inf')
    U_set.append(u)
    lambda_set.append(lamb)
    #S=torch.einsum('ad,bcd->bca', u.t().conj(), S)
    S=yast.ncon([u.conj(),S],[[1+1,2+1,-3+1],[-1+1,-2+1,1+1,2+1]])

    # for l in lambda_set:
    #     if sum(l.get_leg_structure(0).values())>bond_dim:
    #         import pdb; pdb.set_trace()

    return S, U_set, lambda_set





