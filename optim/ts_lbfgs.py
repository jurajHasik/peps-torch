import torch
import numpy as np
import groups.su2 as su2

############################ Basic functions ##################################
def build_gate(j_label, j, tau, device):
    """Compute the gate that is applied to the reduced density matrix."""
    # Spin-spin operator
    #   s1|   |s2
    #     |   |
    #   [  S.S  ]
    #     |   |
    #  s1'|   |s2'
    s2 = su2.SU2(2, dtype=torch.float64, device=device)
    expr_kron = 'ij,ab->iajb'
    if j_label == 'j1':
        # S_1 * S_2 but with S_2 rotated of pi to respect the metric
        SS = - torch.einsum(expr_kron, s2.SZ(), s2.SZ())\
            - 0.5*(torch.einsum(expr_kron, s2.SP(), s2.SP())
                   + torch.einsum(expr_kron, s2.SM(), s2.SM()))
    else:
        SS = torch.einsum(expr_kron, s2.SZ(), s2.SZ())\
            + 0.5*(torch.einsum(expr_kron, s2.SP(), s2.SM())\
                   + torch.einsum(expr_kron, s2.SM(), s2.SP()))
    SS = SS.view(4,4).contiguous()
    # Diagonalization of SS and creation of Hamiltonian Ha
    eig_va, eig_vec = torch.symeig(SS, eigenvectors=True)
    eig_va = torch.exp(-0.5*tau*j*eig_va)
    U = torch.tensor(eig_vec)
    D = torch.diag(torch.tensor(eig_va))
    # SS = U*D*U.T
    gate = torch.einsum('ij,jk,lk->il', U, D, U)
    gate = gate.view(2,2,2,2).contiguous()
    return gate

def single_layer_trotter_decompo_ansatz(j1, tau0, global_args):
    # very special first step
    #
    # build tower from gates exp(-tau_0 S.S) decomposed as MPO
    #
    #    |
    # --MPO
    #    |        |/
    #   MPO-- = --T--
    #    | /     /|
    #   MPO
    #    |
    #   MPO
    #   /|
    #
    expSS_NN= build_gate('j1', j1, tau0, global_args.device)

    # permute indices
    #
    # 0        1     0        2    
    # exp(-tauSS) => exp(-tauSS) => reshape into matrix (0,1)--exp(-tauSS)--(2,3)
    # 2        3     1        3
    expSS_12= expSS_NN.view(2,2,2,2).permute(0,2,1,3).reshape(4,4)

    # print("\n")
    # print("Using symmetric hermitian decomposition")
    # # diagonalize to split into MPO
    # #
    # # (0,1)--U--D--U--(2,3)
    # D,U= np.linalg.eigh(expSS_12)
    # print("D of expSS_12")
    # print(D)

    # # split into 2-site MPO where diagonal D is split symmetrically. This necessarily
    # # involves going to complex numbers as D has some negative values
    # #
    # # 0                       2
    # # U--sqrt(D)-- --sqrt(D)--U
    # # 1                       3
    # MPO= (U@np.diag(np.sqrt(D+0.j))).reshape(2,2,4)

    # # verify we get back the expected operator
    # print("MPO--MPO")
    # print(np.tensordot(MPO,MPO,([2],[2])).reshape(4,4))

    # # build tower
    # #    
    # #        0              0
    # #        |              |
    # # 1<-2--MPO       = 1--MPO
    # #        |1            MPO--3
    # #        |0             |
    # #       MPO--2->3       2
    # #        |
    # #        1->2
    # #
    # X= np.tensordot(MPO,MPO,([1],[0]))

    # # get the difference
    # print("symmetry error (eigendecomp)")
    # print(np.linalg.norm(X-X.transpose(0,3,2,1)))

    # diagonalize to split into MPO
    #
    # (0,1)--U--S--Vh--(2,3)
    U,S,Vh= torch.linalg.svd(expSS_12)
    # sort in ascending fashion
    U= U.flip(1)
    S= S.flip(0)
    Vh= Vh.flip(0)

    # split into two MPOs MPO1, MPO2 since U is not equivalent to Vh
    #
    # 0                          1
    # U--sqrt(S)--2  0--sqrt(S)--Vh
    # 1                          2
    MPO1= (U@torch.diag(torch.sqrt(S))).view(2,2,4)
    MPO2= (torch.diag(torch.sqrt(S))@Vh).view(4,2,2)

    # build tower
    #    
    #        0               0
    #        |               |
    # 3<-2--MPO1       = 3--MPO1
    #        |1             MPO1--5
    #        |0              |  2
    #       MPO1--2->5       | /
    #        |              MPO1
    #        1  2->2        MPO1
    #        0 /           / |  
    #       MPO1          4  1
    #        1
    #        0
    #       MPO1
    #      / |
    #  4<-2  1
    #
    X= torch.einsum('xyl,yvr,vwu,wzd->xzuldr',MPO1,MPO1,MPO1,MPO1).contiguous()

    # X is not C4v-symmetric due to Trotter error
    return X

def C2x2_LU(tensor1, tensor2, C, T):
    C2x1 = torch.tensordot(C, T, ([1],[0]))
    C2x2 = torch.tensordot(C2x1, T, ([0],[0]))
    C2x2 = C2x2.view(C2x2.size()[0],tensor1.size()[1],tensor2.size()[1],
                    C2x2.size()[2],tensor1.size()[1],tensor2.size()[1])
    C2x2 = torch.tensordot(C2x2, tensor1,([1,4],[1,2]))
    C2x2 = torch.tensordot(C2x2, tensor2,([1,3],[1,2]))
    C2x2 = C2x2.permute(1,3,6,0,4,7,2,5).contiguous().view(
            C2x2.size()[1],(tensor1.size()[3]**2),
            C2x2.size()[0]*(tensor1.size()[4]**2),tensor1.size()[0],tensor1.size()[0])
    return C2x2
    
def C2x2_RU(tensor1, tensor2, C, T):
    return C2x2_LU(tensor1.permute(0,1,4,3,2).contiguous(),
                   tensor2.permute(0,1,4,3,2).contiguous(), C, T)

############################ 2 sites functions ###############################
def rdm2x1_sl_2sites(tensor1, tensor2, env):
    """Return a tensor([2,2,2,2])."""
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    C2x1 = torch.tensordot(C, T, ([1],[0]))
    left_half = torch.tensordot(C2x1, C2x2_LU(tensor1, tensor2, C, T), ([0,2],[0,1]))
    rdm = torch.tensordot(left_half,left_half,([0,1],[0,1]))
    rdm = rdm.permute(0,2,1,3).contiguous()
    rdm = torch.einsum('abcdafch', rdm.view((*[2]*8)).contiguous())
    return rdm

def const_w2_2sites(tensor, env, gate):
    rdm2 = rdm2x1_sl_2sites(tensor, tensor, env=env)
    # Compute \omega_2, norm of tensor2*gate
    w2 = torch.einsum('cdij,ijkl,klcd', rdm2, gate, gate)
    return w2

def cost_function_2sites(tensor1, tensor2, env, gate, w2):
    rdm0 = rdm2x1_sl_2sites(tensor2, tensor2, env=env)
    rdm1 = rdm2x1_sl_2sites(tensor1, tensor2, env=env)
    # Compute \omega_0, norm of tensor2
    w0 = torch.einsum('abab', rdm0)
    # Compute \omega_1, overlap of tensor1 and tensor2*gate
    w1 = torch.einsum('abcd, cdab', rdm1, gate)
    cost_function = w1/torch.sqrt(w0*w2)
    return cost_function

def optimization_2sites(onsite1, params_j, env, gate,
                        const_w2, cost_function,
                        params_opt):
    # Parameters
    permutation = params_j['permutation']; new_symmetry = params_j['new_symmetry']
    max_iter = params_opt['max_iter']; threshold = params_opt['threshold']
    patience = params_opt['patience']; noise = params_opt['noise']
    optimizer_class = params_opt['optimizer_class']; lr = params_opt['lr']
    # Normalize tensor
    onsite1.normalize()
    # Permute the on site tensors according to the bond considered
    onsite1.permute(permutation)
    # Create on site tensor to optimize
    onsite2 = onsite1.copy(); onsite2.convert(new_symmetry)
    # Initialize coefficients to optimize
    onsite2.add_noise(noise=noise)
    onsite2.coeff = torch.tensor(onsite2.coeff, dtype=onsite2.dtype, requires_grad=True)
    optimizer = optimizer_class([onsite2.coeff], max_iter=max_iter, lr=lr,\
        tolerance_grad=params_opt['tolerance_grad'], tolerance_change=params_opt['tolerance_change'] )
    # Compute the constant \omega_2
    w2 = const_w2(onsite1.site(), env, gate)
    # Criterion for convergence of the L-BFGS optimizer
    n_bad_steps = 0; best_loss = float('inf')
    threshold = threshold; patience = patience
    loc_history=[]

    def closure():
        optimizer.zero_grad()
        loss = -cost_function(onsite1.site(), onsite2.site(), env, gate, w2)
        # Compute gradient
        loss.backward()
        # might be too much 
        loc_history.append( (loss.item(), max(abs(onsite2.coeff.grad)).item()) )
        # Clip norm gradients to 1.0 to garantee they are not exploding
        torch.nn.utils.clip_grad_norm_(onsite2.coeff, 1.0)
        return loss

    for i in range(1,max_iter):     
        # Update value of the coefficients
        loss_res = optimizer.step(closure)
        # Convergence of the optimizer
        if abs(loss_res.item() - best_loss) > threshold:
                best_loss = loss_res.item()
                n_bad_steps = 0
        else:
                n_bad_steps += 1
        if n_bad_steps > patience:
                break
            
    # Return optimized tensor
    onsite2.coeff = onsite2.coeff.detach(); onsite2.unpermute(permutation)
    return onsite2, loc_history

####################### Plaquette functions for NNN term ######################
def rdm2x2_sl_NNN_plaquette(tensor1, tensor2, tensor_off, diag, env):
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    if diag == 'diag':
        C2x2off = C2x2_LU(tensor_off, tensor_off, C, T)
        C2x2diag = C2x2_RU(tensor1, tensor2, C, T)
    if diag == 'off':
        C2x2off = C2x2_RU(tensor_off, tensor_off, C, T)
        C2x2diag = C2x2_LU(tensor1, tensor2, C, T)   
    C2x2c = torch.einsum('abii->ab', 
                C2x2off.view([C2x2off.size(0)*C2x2off.size(1)]+list(C2x2off.size()[2:])))
    C2x2diag = C2x2diag.view(T.size()[1]*(tensor1.size()[2]**2),
                         T.size()[1]*(tensor2.size()[3]**2),tensor_off.size()[0]**2)
    C2x2diag = torch.einsum('ab,bci->aci', C2x2c, C2x2diag)
    rdm = torch.einsum('abi,baj->ij', C2x2diag, C2x2diag)
    rdm = rdm.view(tuple([tensor1.size()[0] for i in range(4)]))
    rdm = rdm.permute(0,2,1,3).contiguous()
    rdm = torch.einsum('abcdafch', rdm.view((*[2]*8)).contiguous())
    return rdm

def const_w2_NNN_plaquette(tensor, tensor_off, diag, env, gate):
    rdm2 = rdm2x2_sl_NNN_plaquette(tensor, tensor, tensor_off, diag, env)
    # Compute \omega_2, norm of tensor2*gate
    w2 = torch.einsum('cdij,ijkl,klcd', rdm2, gate, gate)
    return w2

def cost_function_NNN_plaquette(tensor1, tensor2, tensor_off, diag, env, gate, w2):
    rdm0 = rdm2x2_sl_NNN_plaquette(tensor2, tensor2, tensor_off, diag, env=env)
    rdm1 = rdm2x2_sl_NNN_plaquette(tensor1, tensor2, tensor_off, diag, env=env)
    # Compute \omega_0, norm of tensor2
    w0 = torch.einsum('abab', rdm0)
    # Compute \omega_1, overlap of tensor1 and tensor2*gate
    w1 = torch.einsum('abcd, cdab', rdm1, gate)
    cost_function = w1/torch.sqrt(w0*w2)
    return cost_function


####################### 1 site functions for NNN term #########################
def rdm1x1_sl(tensor1, tensor2, env):
    """Return a tensor([2,2,2,2])."""
    C = env.C[env.keyC]
    T = env.T[env.keyT]
    CTC = torch.tensordot(C,T,([0],[0]))
    CTC = torch.tensordot(CTC,C,([1],[0]))
    rdm = torch.tensordot(CTC,T,([2],[0]))
    dimsA = tensor1.size()
    a = torch.einsum('mefgh,nabcd->eafbgchdmn', tensor1, tensor2).contiguous()\
        .view(dimsA[1]**2, dimsA[2]**2, dimsA[3]**2, dimsA[4]**2, dimsA[0], dimsA[0])
    rdm = torch.tensordot(rdm,a,([1,3],[1,2]))
    rdm = torch.tensordot(T,rdm,([0,2],[0,2]))
    rdm = torch.tensordot(rdm,CTC,([0,1,2],[0,2,1]))
    rdm = torch.einsum('abac', rdm.view((*[2]*4)).contiguous())
    return rdm

def const_w2_NNN_1site(tensor, env, gate):
    rdm2 = rdm2x1_sl_2sites(tensor, tensor, env=env)
    # Compute \omega_2, norm of tensor2*gate
    w2 = torch.einsum('cdij,ijkl,klcd', rdm2, gate, gate)
    return w2

def cost_function_NNN_1site(tensor1, tensor2, env, gate, w2):
    rdm0 = rdm2x1_sl_2sites(tensor2, tensor2, env=env)
    rdm1 = rdm2x1_sl_2sites(tensor1, tensor2, env=env)
    # Compute \omega_0, norm of tensor2
    w0 = torch.einsum('abab', rdm0)
    # Compute \omega_1, overlap of tensor1 and tensor2*gate
    w1 = torch.einsum('abcd, cdab', rdm1, gate)
    cost_function = w1/torch.sqrt(w0*w2)
    return cost_function

######### Not working ##########
# =============================================================================
# def rdm2x2_sl_NNN_plaquette(tensor1, tensor2, env):
#     """Tensor 1 and 2 are the diagonal onsite tensors. Tensor 3 and 4 are on
#     the off-diagonal."""
#     C = env.C[env.keyC]
#     T = env.T[env.keyT]
#     C2x2LU = C2x2_LU(tensor1, tensor1, C, T)
#     C2x2RU = C2x2_RU(tensor1, tensor2, C, T)
#     C2x2c = torch.einsum('abii->ab', 
#                 C2x2LU.view([C2x2LU.size(0)*C2x2LU.size(1)]+list(C2x2LU.size()[2:])))
#     C2x2RU = C2x2RU.view(T.size()[1]*(tensor1.size()[2]**2),
#                          T.size()[1]*(tensor2.size()[3]**2),tensor1.size()[0]**2)
#     C2x2RU = torch.einsum('ab,bci->aci', C2x2c, C2x2RU)
#     rdm = torch.einsum('abi,baj->ij', C2x2RU, C2x2RU)
#     rdm = rdm.view(tuple([tensor1.size()[0] for i in range(4)]))
#     rdm = rdm.permute(0,2,1,3).contiguous()
#     rdm = torch.einsum('abcdafch', rdm.view((*[2]*8)).contiguous())
#     return rdm
# 
# def const_w2_NNN_plaquette(tensor, env, gate):
#     rdm2 = rdm2x2_sl_NNN_plaquette(tensor, tensor, env=env)
#     # Compute \omega_2, norm of tensor2*gate
#     w2 = torch.einsum('cdij,ijkl,klcd', rdm2, gate, gate)
#     return w2
# 
# def cost_function_NNN_plaquette(tensor1, tensor2, env, gate, w2):
#     rdm0 = rdm2x2_sl_NNN_plaquette(tensor2, tensor2, env=env)
#     rdm1 = rdm2x2_sl_NNN_plaquette(tensor1, tensor2, env=env)
#     # Compute \omega_0, norm of tensor2
#     w0 = torch.einsum('abab', rdm0)
#     # Compute \omega_1, overlap of tensor1 and tensor2*gate
#     w1 = torch.einsum('abcd, cdab', rdm1, gate)
#     cost_function = w1/torch.sqrt(w0*w2)
#     return cost_function
# =============================================================================
