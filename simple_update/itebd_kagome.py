import torch
import numpy as np
from simple_update.hosvd import hosvd
from ipeps.ipess_kagome import *
import time

@torch.no_grad()
def trotter_gate(state, H, dt):
    gate=torch.matrix_exp(H*(-dt))
    gate_half=torch.matrix_exp(H*(-dt/2))
    return gate, gate_half

def pinv(A, itebd_tol):
    # A is diagonal elements, not the full matrix
    A=A/A[0]
    B=A*0
    A_keep=A[abs(A) > itebd_tol]
    B[abs(A) > itebd_tol]=1/A_keep
    return B


def Tri_T_dn(T_d, B_a, B_b, B_c, lambda_up_a, lambda_up_b, lambda_up_c, gate, itebd_tol):
    B_c_new=torch.einsum('uji,ik->ujk', B_c, lambda_up_c)
    B_b_new=torch.einsum('vkc,cm->vkm', B_b, lambda_up_b)
    B_a_new=torch.einsum('wld,dn->wln', B_a, lambda_up_a)
    
    #start=time.clock()
    #this does not cost much time, for D=50, it takes 0.02 second
    A= torch.einsum('jkl,uji,vkc,wld->uivcwd', T_d, B_c_new, B_b_new, B_a_new)
    #end=time.clock()
    #print (end-start)

    d=torch.Tensor.size(B_a)[0]
    D=torch.Tensor.size(B_a)[1]
    gate=gate.reshape(d,d,d,d,d,d)
    A=torch.einsum('abeuvw,uivcwd->aibced', gate, A)
    A=A.reshape(d*D, d*D, d*D)
    S, U_set, lambda_set=hosvd(A)

    S_trun=S[0:D, 0:D, 0:D]
    B_c_new=U_set[0][0:d*D,0:D]
    B_b_new=U_set[1][0:d*D,0:D]
    B_a_new=U_set[2][0:d*D,0:D]
    lambda_dn_c=lambda_set[0][0:D,0:D]
    lambda_dn_b=lambda_set[1][0:D,0:D]
    lambda_dn_a=lambda_set[2][0:D,0:D]

    lambda_up_c_inv=torch.diag(pinv(torch.diag(lambda_up_c), itebd_tol))
    lambda_up_b_inv=torch.diag(pinv(torch.diag(lambda_up_b), itebd_tol))
    lambda_up_a_inv=torch.diag(pinv(torch.diag(lambda_up_a), itebd_tol))

    B_c_new=torch.permute(B_c_new.reshape(d,D,D), [0,2,1])
    B_b_new=torch.permute(B_b_new.reshape(d,D,D), [0,2,1])
    B_a_new=torch.permute(B_a_new.reshape(d,D,D), [0,2,1])

    B_c_new=torch.einsum('uji,ik->ujk', B_c_new, lambda_up_c_inv)
    B_b_new=torch.einsum('vkc,cm->vkm', B_b_new, lambda_up_b_inv)
    B_a_new=torch.einsum('wld,dn->wln', B_a_new, lambda_up_a_inv)

    return B_a_new, B_b_new, B_c_new, lambda_dn_a, lambda_dn_b, lambda_dn_c, S_trun

def Tri_T_up(T_u, B_a, B_b, B_c, lambda_dn_a, lambda_dn_b, lambda_dn_c, gate, itebd_tol):
    B_c_new=torch.einsum('uji,jk->uki', B_c, lambda_dn_c)
    B_b_new=torch.einsum('vka,km->vma', B_b, lambda_dn_b)
    B_a_new=torch.einsum('wlb,ln->wnb', B_a, lambda_dn_a)
    
    A= torch.einsum('iab,uji,vka,wlb->ujvkwl', T_u, B_c_new, B_b_new, B_a_new)

    d=torch.Tensor.size(B_a)[0]
    D=torch.Tensor.size(B_a)[1]
    gate=gate.reshape(d,d,d,d,d,d)
    A=torch.einsum('abeuvw,uivcwd->aibced', gate, A)
    A=A.reshape(d*D, d*D, d*D)
    S, U_set, lambda_set=hosvd(A)

    S_trun=S[0:D, 0:D, 0:D]
    B_c_new=U_set[0][0:d*D,0:D]
    B_b_new=U_set[1][0:d*D,0:D]
    B_a_new=U_set[2][0:d*D,0:D]
    lambda_up_c=lambda_set[0][0:D,0:D]
    lambda_up_b=lambda_set[1][0:D,0:D]
    lambda_up_a=lambda_set[2][0:D,0:D]

    lambda_dn_c_inv=torch.diag(pinv(torch.diag(lambda_dn_c), itebd_tol))
    lambda_dn_b_inv=torch.diag(pinv(torch.diag(lambda_dn_b), itebd_tol))
    lambda_dn_a_inv=torch.diag(pinv(torch.diag(lambda_dn_a), itebd_tol))

    B_c_new=B_c_new.reshape(d,D,D)
    B_b_new=B_b_new.reshape(d,D,D)
    B_a_new=B_a_new.reshape(d,D,D) 

    B_c_new=torch.einsum('uji,jk->uki', B_c_new, lambda_dn_c_inv)
    B_b_new=torch.einsum('vka,km->vma', B_b_new, lambda_dn_b_inv)
    B_a_new=torch.einsum('wlb,ln->wnb', B_a_new, lambda_dn_a_inv)

    return B_a_new, B_b_new, B_c_new, lambda_up_a, lambda_up_b, lambda_up_c, S_trun

def itebd_step(state, lambdas, itebd_tol, gate, posit):
    
    if posit=='dn':
        B_a_new, B_b_new, B_c_new, lambda_dn_a, lambda_dn_b, lambda_dn_c, S_trun=Tri_T_dn(state.ipess_tensors['T_d'], state.ipess_tensors['B_a'], state.ipess_tensors['B_b'], state.ipess_tensors['B_c'], \
            lambdas['lambda_up_a'], lambdas['lambda_up_b'], lambdas['lambda_up_c'], gate, itebd_tol)
        state.ipess_tensors['T_d']=S_trun/(torch.max(abs(S_trun)))
        state.ipess_tensors['B_a']=B_a_new
        state.ipess_tensors['B_b']=B_b_new
        state.ipess_tensors['B_c']=B_c_new
        lambdas['lambda_dn_a']=lambda_dn_a
        lambdas['lambda_dn_b']=lambda_dn_b
        lambdas['lambda_dn_c']=lambda_dn_c
    elif posit=='up':
        B_a_new, B_b_new, B_c_new, lambda_up_a, lambda_up_b, lambda_up_c, S_trun=Tri_T_up(state.ipess_tensors['T_u'], state.ipess_tensors['B_a'], state.ipess_tensors['B_b'], state.ipess_tensors['B_c'], \
            lambdas['lambda_dn_a'], lambdas['lambda_dn_b'], lambdas['lambda_dn_c'], gate, itebd_tol)
        state.ipess_tensors['T_u']=S_trun/(torch.max(abs(S_trun)))
        state.ipess_tensors['B_a']=B_a_new
        state.ipess_tensors['B_b']=B_b_new
        state.ipess_tensors['B_c']=B_c_new
        lambdas['lambda_up_a']=lambda_up_a
        lambdas['lambda_up_b']=lambda_up_b
        lambdas['lambda_up_c']=lambda_up_c
    return state, lambdas

def itebd(state, lambdas, H, itebd_tol, tau, dt):
    gate, gate_half=trotter_gate(state, H, dt)
    state, lambdas=itebd_step(state, lambdas, itebd_tol, gate_half, 'dn')
    for cs in range(0,round(tau/dt)):
        #start=time.clock()
        state, lambdas=itebd_step(state, lambdas, itebd_tol, gate, 'up')
        state, lambdas=itebd_step(state, lambdas, itebd_tol, gate, 'dn')
        #end=time.clock()
        #print (end-start)
    state, lambdas=itebd_step(state, lambdas, itebd_tol, gate_half, 'up')


    state=IPESS_KAGOME_GENERIC({'T_u': state.ipess_tensors['T_u'], 'B_a': state.ipess_tensors['B_a'], 'T_d': state.ipess_tensors['T_d'],\
                    'B_b': state.ipess_tensors['B_b'], 'B_c': state.ipess_tensors['B_c']})
    return state, lambdas