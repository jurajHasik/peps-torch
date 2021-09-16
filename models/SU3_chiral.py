import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.pess_kagome import rdm_kagome
from ctm.generic import corrf
from math import sqrt
from numpy import exp
import itertools

def _cast_to_real(t, check=True, imag_eps=1.0e-8):
    if t.is_complex():
        assert abs(t.imag) < imag_eps,"unexpected imaginary part "+str(t.imag)
        return t.real
    return t

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

su3_gens= torch.zeros(3,3,8, dtype= torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,0]= torch.tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,1]= torch.tensor([[0., -1.j, 0.], [1.j, 0., 0.], [0., 0., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,2]= lambda_3
su3_gens[:,:,3]= torch.tensor([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,4]= torch.tensor([[0., 0., -1.j], [0., 0., 0.], [1.j, 0., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,5]= torch.tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,6]= torch.tensor([[0., 0., 0.], [0., 0., -1.j], [0., 1.j, 0.]], dtype=torch.complex128, device=cfg.global_args.device)
su3_gens[:,:,7]= lambda_8

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
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = 3
        self.id_downT = torch.eye(27, dtype=self.dtype, device=self.device)
        self.h_triangle = (Kr+1j*Ki) * permute_triangle + (Kr-1j*Ki) * permute_triangle_inv + self.j1 * exchange_bond_triangle
        self.h_triangle = self.h_triangle.to(self.device)
        _tmp_l_labels = ["l3","l8","l3_1","l3_2","l3_3","l8_1","l8_2","l8_3"]
        _tmp_l_op= [lambda_3, lambda_8, lambda_3_1, lambda_3_2, lambda_3_3, lambda_8_1, lambda_8_2, lambda_8_3]
        self.obs_ops= { l: op.to(self.device) for l,op in zip(_tmp_l_labels, _tmp_l_op)}
        self.su3_gens= su3_gens.to(self.device)
    # Energy terms

    def energy_triangle_dn(self, state, env, force_cpu=False):
        e_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_triangle, force_cpu=force_cpu)
        return _cast_to_real(e_dn)

    def energy_triangle_up(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open(\
            (0, 0), state, env, force_cpu=force_cpu)
        e_up= torch.einsum('ijkmno,mnoijk', rdm_up, self.h_triangle )
        return _cast_to_real(e_up)

    def energy_nnn(self, state, env, force_cpu=False):
        if self.j2 == 0:
            return 0.
        else:
            vNNN = self.P_bonds_nnn(state, env, force_cpu=force_cpu)
            return(self.j2*(vNNN[0]+vNNN[1]+vNNN[2]+vNNN[3]+vNNN[4]+vNNN[5]))

    # Observables

    def P_dn(self, state, env, force_cpu=False):
        vP_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
            operator=permute_triangle, force_cpu=force_cpu)
        return vP_dn

    def P_up(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
        vP_up= torch.einsum('ijkmno,mnoijk', rdm_up, permute_triangle)
        return vP_up

    def P_bonds_nnn(self, state, env, force_cpu=False):
        norm_wf = rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, \
            self.id_downT, force_cpu=force_cpu)
        vNNN1_12, vNNN1_31 = rdm_kagome.rdm2x2_nnn_1((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        vNNN2_32, vNNN2_21 = rdm_kagome.rdm2x2_nnn_2((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        vNNN3_31, vNNN3_23 = rdm_kagome.rdm2x2_nnn_3((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
        return _cast_to_real(vNNN1_12 / norm_wf), _cast_to_real(vNNN2_21 / norm_wf), \
            _cast_to_real(vNNN1_31 / norm_wf), _cast_to_real(vNNN3_31 / norm_wf), \
            _cast_to_real(vNNN2_32 / norm_wf), _cast_to_real(vNNN3_23 / norm_wf)

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

    def eval_su3_gens(self, state, env):
        id_matrix2 = torch.eye(9, dtype=torch.complex128, device=cfg.global_args.device)
        id_matrix2 = id_matrix2.reshape(3,3,3,3)
        id_matrix = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        # site 0
        l8_ops_0= torch.einsum('ijx,klmn->ikmjlnx', self.su3_gens, id_matrix2).contiguous()
        l8_ops_0= l8_ops_0.reshape(27,27,8)
        for x in range(8):
            l8_x_1x1= rdm.rdm1x1((0, 0), state, env, operator=l8_ops_0[:,:,x]) / norm_wf
            print(f"{x} {l8_x_1x1}")

        # site 1
        l8_ops_1= torch.einsum('ijx,klmn->kimljnx', self.su3_gens, id_matrix2).contiguous()
        l8_ops_1= l8_ops_1.reshape(27,27,8)
        for x in range(8):
            l8_x_1x1= rdm.rdm1x1((0, 0), state, env, operator=l8_ops_1[:,:,x]) / norm_wf
            print(f"{x} {l8_x_1x1}")

        # site 2
        l8_ops_2= torch.einsum('ijx,klmn->kmilnjx', self.su3_gens, id_matrix2).contiguous()
        l8_ops_2= l8_ops_2.reshape(27,27,8)
        for x in range(8):
            l8_x_1x1= rdm.rdm1x1((0, 0), state, env, operator=l8_ops_2[:,:,x]) / norm_wf
            print(f"{x} {l8_x_1x1}")

    def eval_obs(self,state,env,force_cpu=True):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]
        """
        selected_ops= ["l3_1","l3_2","l3_3","l8_1","l8_2","l8_3"]
        id_matrix = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        obs= {}
        with torch.no_grad():
            for label in selected_ops:
                obs_val= rdm.rdm1x1((0, 0), state, env, operator=self.obs_ops[label]) / norm_wf
                obs[f"{label}"]= obs_val #_cast_to_real(obs_val)

        # prepare list with labels and values
        return list(obs.values()), list(obs.keys())