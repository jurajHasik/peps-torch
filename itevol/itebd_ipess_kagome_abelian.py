import torch
import numpy as np
from itevol.hosvd_abelian import hosvd
from ipeps.ipess_kagome_abelian import *
import examples.abelian.settings_U1_torch as settings_U1
import yastn.yastn as yastn
import time

def trotter_gate(H,dt):
    D, U= yastn.linalg.eigh(H, axes=([0],[1]))
    Da= D.exp(-dt)
    Db= D.exp(-dt/2)
    gate= U.tensordot(Da, ([1],[0]))
    gate= gate.tensordot(U, ([1,1]), conj=(0,1))

    gate_half= U.tensordot(Db, ([1],[0]))
    gate_half= gate_half.tensordot(U, ([1,1]), conj=(0,1))
    return gate, gate_half

def pinv(A, itebd_tol):
    # A is diagonal elements, not the full matrix
    A=A/A[0]
    B=A*0
    A_keep=A[abs(A) > itebd_tol]
    B[abs(A) > itebd_tol]=1/A_keep
    return B

def Tri_T_dn(T_d, B_a, B_b, B_c, lambda_up_a, lambda_up_b, lambda_up_c, gate,\
    itebd_tol,bond_dim,keep_multiplet):
    #B_c_new=torch.einsum('uji,ik->ujk', B_c, lambda_up_c)
    B_c_new=yastn.ncon([B_c, lambda_up_c],[[-1+1,-2+1,1+1],[1+1,-3+1]])
    #B_b_new=torch.einsum('vkc,cm->vkm', B_b, lambda_up_b)
    B_b_new=yastn.ncon([B_b, lambda_up_b],[[-1+1,-2+1,1+1],[1+1,-3+1]])
    #B_a_new=torch.einsum('wld,dn->wln', B_a, lambda_up_a)
    B_a_new=yastn.ncon([B_a, lambda_up_a],[[-1+1,-2+1,1+1],[1+1,-3+1]])
    
    #start=time.clock()
    #this does not cost much time, for D=50, it takes 0.02 second
    #A= torch.einsum('jkl,uji,vkc,wld->uivcwd', T_d, B_c_new, B_b_new, B_a_new)
    A=yastn.ncon([T_d, B_c_new, B_b_new, B_a_new],[[1+1,2+1,3+1],[-1+1,1+1,-2+1],[-3+1,2+1,-4+1],[-5+1,3+1,-6+1]])
    #end=time.clock()
    #print (end-start)

    #d=torch.Tensor.size(B_a)[0]
    #D=torch.Tensor.size(B_a)[1]
    #gate=gate.reshape(d,d,d,d,d,d)
    gate=gate.unfuse_legs(axes=(0,1))
    #A=torch.einsum('abeuvw,uivcwd->aibced', gate, A)
    A=yastn.ncon([gate, A],[[-1+1,-3+1,-5+1,1+1,2+1,3+1],[1+1,-2+1,2+1,-4+1,3+1,-6+1]])
    #A=A.reshape(d*D, d*D, d*D)
    #A=A.fuse_legs(axes=((0,1),(2,3),(4,5)))
    S_trun, U_set, lambda_set=hosvd(A,itebd_tol,bond_dim,keep_multiplet)

    B_c_new=U_set[0]
    B_b_new=U_set[1]
    B_a_new=U_set[2]
    lambda_dn_c=lambda_set[0]
    lambda_dn_b=lambda_set[1]
    lambda_dn_a=lambda_set[2]

    lambda_up_c_inv=lambda_up_c.reciprocal(cutoff=itebd_tol)
    lambda_up_b_inv=lambda_up_b.reciprocal(cutoff=itebd_tol)
    lambda_up_a_inv=lambda_up_a.reciprocal(cutoff=itebd_tol)

    B_c_new=B_c_new.transpose(axes=(0,2,1))
    B_b_new=B_b_new.transpose(axes=(0,2,1))
    B_a_new=B_a_new.transpose(axes=(0,2,1))

    B_c_new=yastn.ncon([B_c_new, lambda_up_c_inv],[[-1+1,-2+1,1+1],[1+1,-3+1]])
    B_b_new=yastn.ncon([B_b_new, lambda_up_b_inv],[[-1+1,-2+1,1+1],[1+1,-3+1]])
    B_a_new=yastn.ncon([B_a_new, lambda_up_a_inv],[[-1+1,-2+1,1+1],[1+1,-3+1]])

    #print(lambda_dn_a.to_dense())
    return B_a_new, B_b_new, B_c_new, lambda_dn_a, lambda_dn_b, lambda_dn_c, S_trun

def Tri_T_up(T_u, B_a, B_b, B_c, lambda_dn_a, lambda_dn_b, lambda_dn_c, gate,\
    itebd_tol,bond_dim,keep_multiplet):
    #B_c_new=torch.einsum('uji,jk->uki', B_c, lambda_dn_c)
    B_c_new=yastn.ncon([B_c, lambda_dn_c],[[-1+1,1+1,-3+1],[1+1,-2+1]])
    #B_b_new=torch.einsum('vka,km->vma', B_b, lambda_dn_b)
    B_b_new=yastn.ncon([B_b, lambda_dn_b],[[-1+1,1+1,-3+1],[1+1,-2+1]])
    #B_a_new=torch.einsum('wlb,ln->wnb', B_a, lambda_dn_a)
    B_a_new=yastn.ncon([B_a, lambda_dn_a],[[-1+1,1+1,-3+1],[1+1,-2+1]])
    
    #A= torch.einsum('iab,uji,vka,wlb->ujvkwl', T_u, B_c_new, B_b_new, B_a_new)
    A=yastn.ncon([T_u, B_c_new, B_b_new, B_a_new],[[1+1,2+1,3+1],[-1+1,-2+1,1+1],[-3+1,-4+1,2+1],[-5+1,-6+1,3+1]])
    #gate=gate.reshape(d,d,d,d,d,d)
    gate=gate.unfuse_legs(axes=(0,1))
    #A=torch.einsum('abeuvw,uivcwd->aibced', gate, A)
    A=yastn.ncon([gate, A],[[-1+1,-3+1,-5+1,1+1,2+1,3+1],[1+1,-2+1,2+1,-4+1,3+1,-6+1]])
    #A=A.reshape(d*D, d*D, d*D)
    #A=A.fuse_legs(axes=((0,1),(2,3),(4,5)))
    S_trun, U_set, lambda_set=hosvd(A,itebd_tol,bond_dim,keep_multiplet)

    B_c_new=U_set[0]
    B_b_new=U_set[1]
    B_a_new=U_set[2]
    lambda_up_c=lambda_set[0]
    lambda_up_b=lambda_set[1]
    lambda_up_a=lambda_set[2]

    lambda_dn_c_inv=lambda_dn_c.reciprocal(cutoff=itebd_tol)
    lambda_dn_b_inv=lambda_dn_b.reciprocal(cutoff=itebd_tol)
    lambda_dn_a_inv=lambda_dn_a.reciprocal(cutoff=itebd_tol)

    B_c_new=yastn.ncon([B_c_new, lambda_dn_c_inv],[[-1+1,1+1,-3+1],[1+1,-2+1]])
    B_b_new=yastn.ncon([B_b_new, lambda_dn_b_inv],[[-1+1,1+1,-3+1],[1+1,-2+1]])
    B_a_new=yastn.ncon([B_a_new, lambda_dn_a_inv],[[-1+1,1+1,-3+1],[1+1,-2+1]])

    #print(lambda_up_a.to_dense())
    return B_a_new, B_b_new, B_c_new, lambda_up_a, lambda_up_b, lambda_up_c, S_trun

def itebd_step(state, lambdas, itebd_tol, gate, posit, bond_dim, keep_multiplet):
    
    if posit=='dn':
        B_a_new, B_b_new, B_c_new, lambda_dn_a, lambda_dn_b, lambda_dn_c,S_trun=\
            Tri_T_dn(state.ipess_tensors['T_d'], state.ipess_tensors['B_a'], \
                state.ipess_tensors['B_b'], state.ipess_tensors['B_c'], \
                lambdas['lambda_up_a'], lambdas['lambda_up_b'], lambdas['lambda_up_c'], 
                gate, itebd_tol, bond_dim, keep_multiplet)
        state.ipess_tensors['T_d']=S_trun/S_trun.norm(p='inf')
        state.ipess_tensors['B_a']=B_a_new
        state.ipess_tensors['B_b']=B_b_new
        state.ipess_tensors['B_c']=B_c_new
        lambdas['lambda_dn_a']=lambda_dn_a
        lambdas['lambda_dn_b']=lambda_dn_b
        lambdas['lambda_dn_c']=lambda_dn_c
    elif posit=='up':
        B_a_new, B_b_new, B_c_new, lambda_up_a, lambda_up_b, lambda_up_c, S_trun=\
            Tri_T_up(state.ipess_tensors['T_u'], state.ipess_tensors['B_a'], \
                state.ipess_tensors['B_b'], state.ipess_tensors['B_c'], \
            lambdas['lambda_dn_a'], lambdas['lambda_dn_b'], lambdas['lambda_dn_c'], \
            gate, itebd_tol, bond_dim, keep_multiplet)
        state.ipess_tensors['T_u']=S_trun/S_trun.norm(p='inf')
        state.ipess_tensors['B_a']=B_a_new
        state.ipess_tensors['B_b']=B_b_new
        state.ipess_tensors['B_c']=B_c_new
        lambdas['lambda_up_a']=lambda_up_a
        lambdas['lambda_up_b']=lambda_up_b
        lambdas['lambda_up_c']=lambda_up_c
    return state, lambdas

def itebd(state, lambdas, H, itebd_tol, tau, dt, bond_dim, keep_multiplet):
    gate, gate_half=trotter_gate(H, dt)
    #import pdb; pdb.set_trace()

    state, lambdas=itebd_step(state, lambdas, itebd_tol, gate_half, 'dn', bond_dim, keep_multiplet)
    for cs in range(0,round(tau/dt)):
        #start=time.clock()
        state, lambdas=itebd_step(state, lambdas, itebd_tol, gate, 'up', bond_dim, keep_multiplet)
        state, lambdas=itebd_step(state, lambdas, itebd_tol, gate, 'dn', bond_dim, keep_multiplet)
        #end=time.clock()
        #print (end-start)
    state, lambdas=itebd_step(state, lambdas, itebd_tol, gate_half, 'up', bond_dim, keep_multiplet)

    state= IPESS_KAGOME_GENERIC_ABELIAN(state.engine, 
        {'T_u': state.ipess_tensors['T_u'], 'B_a': state.ipess_tensors['B_a'], \
         'T_d': state.ipess_tensors['T_d'], 'B_b': state.ipess_tensors['B_b'],\
         'B_c': state.ipess_tensors['B_c']})
    return state, lambdas