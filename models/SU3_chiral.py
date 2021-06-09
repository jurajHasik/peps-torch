import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import rdm_kagome
from ctm.generic import corrf
from math import sqrt
from numpy import exp
import itertools


# function (n1,n2,n3) --> s that maps the basis of states in the fundamental irrep of SU(3) (states n=0,1,2) for the three sites of the unit cell to a single physical index s=0...26
# NB: the 3 sites are labeled as:
#        1---3
#         \ /
#          2
def fmap(n1, n2, n3):
    return n3 + 3 * n2 + 9 * n1


# reverse mapping s --> (n1, n2, n3)
def fmap_inv(s):
    n1 = s // 9
    n2 = (s - 9 * n1) // 3
    n3 = s - 9 * n1 - 3 * n2
    return (n1, n2, n3)


exchange_bond = torch.zeros((3, 3, 3, 3), dtype=torch.complex128, device=cfg.global_args.device)
for i in range(3):
    for j in range(3):
        exchange_bond[i, j, j, i] = 1.

exchange_bond_triangle = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128, device=cfg.global_args.device)
for i in range(3):
    for j in range(3):
        for k in range(3):
            # 1--2
            exchange_bond_triangle[i, j, k, j, i, k] = 1.
            # 2--3
            exchange_bond_triangle[i, j, k, i, k, j] = 1.
            # 3--1
            exchange_bond_triangle[i, j, k, k, j, i] = 1.

permute_triangle = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128, device=cfg.global_args.device)
permute_triangle_inv = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128, device=cfg.global_args.device)
for i in range(3):
    for j in range(3):
        for k in range(3):
            # anticlockwise (direct)
            permute_triangle[i, j, k, j, k, i] = 1.
            # clockwise (inverse)
            permute_triangle_inv[i, j, k, k, i, j] = 1.

# define the matrices associated with the observables \lambda_3 and \lambda_8 for the 3 sites
lambda_3 = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
lambda_8 = 1. / sqrt(3.) * torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -2.]], dtype=torch.complex128, device=cfg.global_args.device)
lambda_3_1 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
lambda_3_2 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
lambda_3_3 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
lambda_8_1 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
lambda_8_2 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
lambda_8_3 = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
for s in range(27):
    n1, n2, n3 = fmap_inv(s)
    lambda_3_1[s, s] = lambda_3[n1, n1]
    lambda_3_2[s, s] = lambda_3[n2, n2]
    lambda_3_3[s, s] = lambda_3[n3, n3]
    lambda_8_1[s, s] = lambda_8[n1, n1]
    lambda_8_2[s, s] = lambda_8[n2, n2]
    lambda_8_3[s, s] = lambda_8[n3, n3]


class SU3_CHIRAL():

    def __init__(self, Kr=0., Ki=0., j1=0., j2=0., global_args=cfg.global_args):
        self.j1 = j1
        self.j2 = j2
        self.Kr = Kr
        self.Ki = Ki
        print('Hamiltonian coupling constants:')
        print("Kr = {}".format(Kr))
        print("Ki = {}".format(Ki))
        print("j1 = {}".format(j1))
        print("j2 = {}".format(j2))
        self.dtype = global_args.dtype
        self.device = global_args.device
        self.phys_dim = 27
        self.h_triangle = (Kr+1j*Ki) * permute_triangle + (Kr-1j*Ki) * permute_triangle_inv + self.j1 * exchange_bond_triangle
        self.h_triangle = self.h_triangle.to(self.device)

    # Energy terms

    def energy_triangle_dn(self, state, env, force_cpu=False):
        norm_wf = rdm.rdm2x2_id((0, 0), state, env, force_cpu=force_cpu)
        e_dn = rdm.rdm2x2_dn_triangle((0, 0), state, env, operator=self.h_triangle, force_cpu=force_cpu) / norm_wf
        return torch.real(e_dn)

    def energy_triangle_up(self, state, env, force_cpu=False):
        norm_wf = rdm.rdm2x2_id((0, 0), state, env, force_cpu=force_cpu)
        e_up = rdm.rdm2x2_up_triangle((0, 0), state, env, operator=self.h_triangle, force_cpu=force_cpu) / norm_wf
        return torch.real(e_up)

    def energy_triangle_dn_v2(self, state, env, force_cpu=False):
        rdm_dn= rdm_kagome.rdm2x2_up_triangle_open(\
            (0, 0), state, env, force_cpu=force_cpu)
        e_dn= torch.einsum('ijkmno,mnoijk', rdm_dn, self.h_triangle )
        return torch.real(e_dn)

    def energy_triangle_up_v2(self, state, env, force_cpu=False):
        e_up= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_triangle, force_cpu=force_cpu)
        return torch.real(e_up)


    def energy_nnn(self, state, env, force_cpu=False):
        if self.j2 == 0:
            return 0.
        else:
            vNNN = self.P_bonds_nnn(state, env, force_cpu=force_cpu)
            return(self.j2*(vNNN[0]+vNNN[1]+vNNN[2]+vNNN[3]+vNNN[4]+vNNN[5]))

    # Observables

    def P_dn(self, state, env, force_cpu=False):
        norm_wf = rdm.rdm2x2_id((0, 0), state, env, force_cpu=force_cpu)
        vP_dn = rdm.rdm2x2_dn_triangle((0, 0), state, env, operator=permute_triangle, force_cpu=force_cpu) / norm_wf
        return vP_dn

    def P_up(self, state, env, force_cpu=False):
        norm_wf = rdm.rdm2x2_id((0, 0), state, env, force_cpu=force_cpu)
        vP_up = rdm.rdm2x2_up_triangle((0, 0), state, env, operator=permute_triangle, force_cpu=force_cpu) / norm_wf
        return vP_up

    def P_bonds_nnn(self, state, env, force_cpu=False):
        norm_wf = rdm.rdm2x2_id((0, 0), state, env, force_cpu=force_cpu)
        vNNN1_12, vNNN1_31 = rdm.rdm2x2_nnn_1((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        vNNN2_32, vNNN2_21 = rdm.rdm2x2_nnn_2((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        vNNN3_31, vNNN3_23 = rdm.rdm2x2_nnn_3((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        return torch.real(vNNN1_12 / norm_wf), torch.real(vNNN2_21 / norm_wf), torch.real(vNNN1_31 / norm_wf), torch.real(vNNN3_31 / norm_wf), torch.real(vNNN2_32 / norm_wf), torch.real(vNNN3_23 / norm_wf)

    def P_bonds_nn(self, state, env):
        id_matrix = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        # bond 2--3
        bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bond_op[fmap(i,j,k),fmap(i,k,j)] = 1.
        vP_23 = rdm.rdm1x1((0,0), state, env, operator=bond_op) / norm_wf
        # bond 1--3
        bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bond_op[fmap(i,j,k),fmap(k,j,i)] = 1.
        vP_13 = rdm.rdm1x1((0, 0), state, env, operator=bond_op) / norm_wf
        # bond 1--2
        bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    bond_op[fmap(i,j,k),fmap(j,i,k)] = 1.
        vP_12 = rdm.rdm1x1((0, 0), state, env, operator=bond_op) / norm_wf
        return(torch.real(vP_23), torch.real(vP_13), torch.real(vP_12))

    def eval_lambdas(self, state, env):
        # computes the expectation value of the SU(3) observables \lambda_3 and \lambda_8 for the three sites of the unit cell
        id_matrix = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        color3_1 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_1) / norm_wf
        color3_2 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_2) / norm_wf
        color3_3 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_3) / norm_wf
        color8_1 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_1) / norm_wf
        color8_2 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_2) / norm_wf
        color8_3 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_3) / norm_wf
        return (color3_1, color3_2, color3_3), (color8_1, color8_2, color8_3)
