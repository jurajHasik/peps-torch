import numpy
import torch

def hosvd(A):
    #A is n(1) x ... x n(d) tensor
    #U is d cell array with U(k) being the left singular vector of a's mode-k unfolding
    #S is n(1) x ... x n(d) tensor : A x1 U(1) x2 U(2) ... xd U(d)
    S = A
    U_set = []
    lambda_set = []
    shape = A.shape
    size = numpy.prod(shape)

    #index 0
    AA=torch.einsum('abc,dbc->ad', A, A.conj())
    u,lamb,_=torch.svd(AA)
    lamb=torch.diag(torch.sqrt(lamb))
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ab,bcd->acd', u.t().conj(), S)

    #index 1
    AA=torch.einsum('bac,bdc->ad', A, A.conj())
    u,lamb,_=torch.svd(AA)
    lamb=torch.diag(torch.sqrt(lamb))
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ac,bcd->bad', u.t().conj(), S)

    #index 2
    AA=torch.einsum('bca,bcd->ad', A, A.conj())
    u,lamb,_=torch.svd(AA)
    lamb=torch.diag(torch.sqrt(lamb))
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ad,bcd->bca', u.t().conj(), S)



    return S, U_set, lambda_set


def hosvd1(A):
    #A is n(1) x ... x n(d) tensor
    #U is d cell array with U(k) being the left singular vector of a's mode-k unfolding
    #S is n(1) x ... x n(d) tensor : A x1 U(1) x2 U(2) ... xd U(d)
    S = A
    U_set = []
    lambda_set = []
    shape = A.shape
    size = numpy.prod(shape)

    #index 0
    u,lamb,_=torch.svd(S.reshape(shape[0], shape[1]*shape[2]))
    lamb=torch.diag(lamb)
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ab,bcd->acd', u.t().conj(), S)

    #index 1
    u,lamb,_=torch.svd(torch.permute(S, (1,0,2)).reshape(shape[0], shape[1]*shape[2]))
    lamb=torch.diag(lamb)
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ac,bcd->bad', u.t().conj(), S)

    #index 2
    u,lamb,_=torch.svd(torch.permute(S, (2,0,1)).reshape(shape[0], shape[1]*shape[2]))
    lamb=torch.diag(lamb)
    lamb=lamb/lamb[0,0]
    if u.dtype==torch.complex128:
        lamb=lamb.type(torch.complex128)
    U_set.append(u)
    lambda_set.append(lamb)
    S=torch.einsum('ad,bcd->bca', u.t().conj(), S)

    # print('check hosvd')
    # check_hosvd(A, S, U_set, lambda_set)

    return S, U_set, lambda_set


def check_hosvd(A, S, U_set, lambda_set):

    S0=torch.einsum('ab,bcd->acd', U_set[0], S)
    S1=torch.einsum('ac,bcd->bad', U_set[1], S0)
    S2=torch.einsum('ad,bcd->bca', U_set[2], S1)
    print(torch.linalg.norm(A))
    print(torch.linalg.norm(S2-A))

    lambda0=(torch.einsum('ab,bcd->acd', U_set[0].conj().t(), A))
    lambda0_square=torch.einsum('abc,dbc->ad', lambda0, lambda0.conj())
    lambda0_square=lambda0_square/lambda0_square[0,0]
    l0=lambda_set[0]
    print(torch.linalg.norm(torch.einsum('ab,bc->ac', l0/l0[0,0], l0/l0[0,0])-lambda0_square))
    
    lambda1=(torch.einsum('ac,bcd->bad', U_set[1].conj().t(), A))
    lambda1_square=torch.einsum('abc,adc->bd', lambda1, lambda1.conj())
    lambda1_square=lambda1_square/lambda1_square[0,0]
    l1=lambda_set[1]
    print(torch.linalg.norm(torch.einsum('ab,bc->ac', l1/l1[0,0], l1/l1[0,0])-lambda1_square))

    lambda2=(torch.einsum('ad,bcd->bca', U_set[2].conj().t(), A))
    lambda2_square=torch.einsum('abc,abd->cd', lambda2, lambda2.conj())
    lambda2_square=lambda2_square/lambda2_square[0,0]
    l2=lambda_set[2]
    print(torch.linalg.norm(torch.einsum('ab,bc->ac', l2/l2[0,0], l2/l2[0,0])-lambda2_square))