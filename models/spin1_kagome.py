import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.pess_kagome import rdm_kagome
from ctm.generic import corrf
import groups.su2 as su2
from models.spin_half_kagome import S_HALF_KAGOME
from math import sqrt
from numpy import exp
import itertools

_cast_to_real= rdm._cast_to_real
# def _cast_to_real(t, check=True, imag_eps=1.0e-12):
#     if t.is_complex():
#         assert abs(t.imag) < imag_eps,"unexpected imaginary part "+str(t.imag)
#         return t.real
#     return t

class S1_KAGOME(S_HALF_KAGOME):

    def __init__(self, j1=1., JD=0, j1sq=0, j2=0, j2sq=0, jtrip=0.,\
        jperm=0+0j, h=0, global_args=cfg.global_args):
        r"""
        H = J_1 \sum_{<ij>} S_i.S_j + J_{1sq} \sum_{<ij>} (S_i.S_j)^2
            + J_2 \sum_{<<ij>>} S_i.S_j + J_{2sq} \sum_{<<ij>>} (S_i.S_j)^2
            - J_{trip} \sum_t (S_{t_1} \times S_{t_2}).S_{t_3}
            + J_{perm} \sum_t P_t + J*_{perm} \sum_t P^{-1}_t
        """
        super().__init__(j1=j1, JD=JD, j1sq=j1sq, j2=j2, j2sq=j2sq, jtrip=jtrip,\
            jperm=jperm, h=h, phys_dim=3, global_args=global_args)

    def energy_nnn(self, state, env, force_cpu=False):
        if self.j2 == 0:
            return 0.
        else:
            vNNN = self.P_bonds_nnn(state, env, force_cpu=force_cpu)
            return(self.j2*(vNNN[0]+vNNN[1]+vNNN[2]+vNNN[3]+vNNN[4]+vNNN[5]))

    # Observables
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
        pd3= self.phys_dim**3
        id_matrix = torch.eye(pd3, dtype=torch.complex128, device=cfg.global_args.device)
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
        # bond 2--3
        bond_op = torch.zeros((pd3, pd3), dtype=torch.complex128, device=cfg.global_args.device)
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