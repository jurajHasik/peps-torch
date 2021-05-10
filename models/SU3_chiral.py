import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
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


exchange_bond = torch.zeros((3, 3, 3, 3), dtype=torch.complex128)
for i in range(3):
    for j in range(3):
        exchange_bond[i, j, j, i] = 1.

exchange_bond_triangle = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128)
for i in range(3):
    for j in range(3):
        for k in range(3):
            # 1--2
            exchange_bond_triangle[i, j, j, i, k, k] = 1.
            # 2--3
            exchange_bond_triangle[i, i, j, k, k, j] = 1.
            # 3--1
            exchange_bond_triangle[i, k, j, j, k, i] = 1.

permute_triangle = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128)
permute_triangle_inv = torch.zeros((3, 3, 3, 3, 3, 3), dtype=torch.complex128)
for i in range(3):
    for j in range(3):
        for k in range(3):
            # anticlockwise (direct)
            permute_triangle[i, j, k, j, k, i] = 1.
            # clockwise (inverse)
            permute_triangle_inv[i, j, k, k, i, j] = 1.

# define the matrices associated with the observables \lambda_3 and \lambda_8 for the 3 sites
lambda_3 = torch.tensor([[1., 0., 0.], [0., -1., 0.], [0., 0., 0.]], dtype=torch.complex128)
lambda_8 = 1. / sqrt(3.) * torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., -2.]], dtype=torch.complex128)
lambda_3_1 = torch.eye(27, dtype=torch.complex128)
lambda_3_2 = torch.eye(27, dtype=torch.complex128)
lambda_3_3 = torch.eye(27, dtype=torch.complex128)
lambda_8_1 = torch.eye(27, dtype=torch.complex128)
lambda_8_2 = torch.eye(27, dtype=torch.complex128)
lambda_8_3 = torch.eye(27, dtype=torch.complex128)
for s in range(27):
    n1, n2, n3 = fmap_inv(s)
    lambda_3_1[s, s] = lambda_3[n1, n1]
    lambda_3_2[s, s] = lambda_3[n2, n2]
    lambda_3_3[s, s] = lambda_3[n3, n3]
    lambda_8_1[s, s] = lambda_8[n1, n1]
    lambda_8_2[s, s] = lambda_8[n2, n2]
    lambda_8_3[s, s] = lambda_8[n3, n3]


class SU3_CHIRAL():

    def __init__(self, theta=0., j1=0., j2=0., global_args=cfg.global_args):
        r"""
        :param theta: rotation angle of the hamiltonian which parametrizes the ratio between real and imaginary permutations
        :param global_args: global configuration
        :type theta: float
        :type global_args: GLOBALARGS
        """
        self.j1 = j1
        self.j2 = j2
        self.theta = theta
        self.dtype = global_args.dtype
        self.device = global_args.device
        self.phys_dim = 27
        self.h_triangle = exp(1j * self.theta) * permute_triangle + exp(
            -1j * self.theta) * permute_triangle_inv + self.j1 * exchange_bond_triangle

    def P_dn(self, state, env):
        vP_dn = rdm.rdm2x2_dn_triangle((0, 0), state, env, operator=permute_triangle) / state.norm_wf
        return vP_dn

    def P_up(self, state, env):
        vP_up = rdm.rdm2x2_up_triangle((0, 0), state, env, operator=permute_triangle) / state.norm_wf
        return vP_up

    def energy_nnn(self, state, env):
        if self.j2 == 0.:
            return 0.
        else:
            vNNN1 = rdm.rdm2x2_nnn_1((0, 0), state, env, operator=exchange_bond)
            vNNN2 = rdm.rdm2x2_nnn_2((0, 0), state, env, operator=exchange_bond)
            vNNN3 = rdm.rdm2x2_nnn_3((0, 0), state, env, operator=exchange_bond)
            return torch.real(self.j2 * (vNNN1 + vNNN2 + vNNN3) / state.norm_wf)

    def energy_triangle_dn(self, state, env):
        e_dn = rdm.rdm2x2_dn_triangle((0, 0), state, env, operator=self.h_triangle) / state.norm_wf
        return torch.real(e_dn)

    def energy_triangle_up(self, state, env):
        e_up = rdm.rdm2x2_up_triangle((0, 0), state, env, operator=self.h_triangle) / state.norm_wf
        return torch.real(e_up)

    def eval_lambdas(self, state, env):
        # computes the expectation value of the SU(3) observables \lambda_3 and \lambda_8 for the three sites of the unit cell
        id_matrix = torch.eye(27, dtype=torch.complex128)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        color3_1 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_1) / norm_wf
        color3_2 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_2) / norm_wf
        color3_3 = rdm.rdm1x1((0, 0), state, env, operator=lambda_3_3) / norm_wf
        color8_1 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_1) / norm_wf
        color8_2 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_2) / norm_wf
        color8_3 = rdm.rdm1x1((0, 0), state, env, operator=lambda_8_3) / norm_wf
        return (color3_1, color3_2, color3_3), (color8_1, color8_2, color8_3)
