import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import corrf
from ctm.generic import rdm_kagome  # modified by Yi, 06/15/21
from math import sqrt
from tn_interface import einsum, mm
from tn_interface import view, permute, contiguous
import itertools
import numpy as np


def _cast_to_real(t):
    return t.real if t.is_complex() else t


class KAGOME():
    def __init__(self, phys_dim=3, j=0.0, k=1.0, h=0.0, global_args=cfg.global_args):
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = phys_dim
        self.j = j  # 2-site permutation coupling constant
        self.k = k  # 3-site ring exchange coupling constant
        self.h = h  # 3-site ring exchange coupling constantNN coupling between A & C in the same unit cell

        # For now, only one-site
        self.obs_ops = self.get_obs_ops()
        self.h2, self.h3_l, self.h3_r = self.get_h()

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"] = irrep.SZ()
        obs_ops["sp"] = irrep.SP()
        obs_ops["sm"] = irrep.SM()
        obs_ops["SS"] = irrep.SS()

        return obs_ops

    def get_h(self):
        pd = self.phys_dim
        irrep = su2.SU2(pd, dtype=self.dtype, device=self.device)
        # identity operator on two spin-S spins
        idp2x2 = torch.eye(pd**2, dtype=self.dtype,device=self.device)
        SS = irrep.SS()
        SS = SS.view(pd**2, pd**2)
        perm_2site = SS + SS@SS - idp2x2
        # Reshape back into rank-4 tensor for later use with reduced density matrices
        perm_2site = perm_2site.view(pd, pd, pd, pd).contiguous()

        h2 = perm_2site
        h3_l = torch.einsum('ijal,lkbc->ijkabc', perm_2site, perm_2site)
        h3_r = torch.einsum('ijal,klbc->ikjabc', perm_2site, perm_2site)

        return h2, h3_l, h3_r

    def energy_1site(self, state, env):
        pd = self.phys_dim
        energy = 0.0
        idp = torch.eye(pd, dtype=self.dtype, device=self.device) / pd  # trace(idp * rdm) = 1
        for coord, site in state.sites.items():
            # intra-cell 2-site exchange terms
            rdm1x1_ab = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B')).view(pd, pd, pd, pd)
            energy += self.j * torch.einsum('ijab,ijab', rdm1x1_ab, self.h2)
            rdm1x1_bc = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('B', 'C')).view(pd, pd, pd, pd)
            energy += self.j * torch.einsum('ijab,ijab', rdm1x1_bc, self.h2)
            rdm1x1_ac = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'C')).view(pd, pd, pd, pd)
            energy += self.j * torch.einsum('ijab,ijab', rdm1x1_ac, self.h2)

            # intra-cell 3-site ring exchange terms
            rdm1x1 = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B', 'C')).view(pd, pd, pd, pd, pd, pd)
            energy += _cast_to_real((self.k + self.h * 1j) * torch.einsum('ijkabc,ijkabc', rdm1x1, self.h3_l))
            energy += _cast_to_real((self.k - self.h * 1j) * torch.einsum('ijkabc,ijkabc', rdm1x1, self.h3_r))

            # inter-cell 2-site exchange terms
            rdm2x2_ab = rdm_kagome.rdm2x2_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=(), sites_to_keep_01=(), sites_to_keep_11=('A'))
            energy += _cast_to_real(self.j * torch.einsum('ijklabcd,ilad,jb,kc', rdm2x2_ab, self.h2, idp, idp))
            rdm1x2_bc = rdm_kagome.rdm2x1_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=('C'))
            energy += _cast_to_real(self.j * torch.einsum('ijab,ijab', rdm1x2_bc, self.h2))
            rdm2x1_ac = rdm_kagome.rdm1x2_kagome(coord, state, env, sites_to_keep_00=('C'), sites_to_keep_01=('A'))
            energy += _cast_to_real(self.j * torch.einsum('ijab,ijab', rdm2x1_ac, self.h2))

            # inter-cell 3-site ring exchange terms
            rdm2x2_ring = rdm_kagome.rdm2x2_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=('C'), sites_to_keep_01=(), sites_to_keep_11=('A'))
            energy += _cast_to_real((self.k + self.h * 1j) * torch.einsum('ijklabcd,lijdab,kc', rdm2x2_ring, self.h3_l, idp))
            energy += _cast_to_real((self.k - self.h * 1j) * torch.einsum('ijklabcd,lijdab,kc', rdm2x2_ring, self.h3_r, idp))
        energy_per_site = energy/len(state.sites.items())
        energy_per_site = _cast_to_real(energy_per_site)
        return energy_per_site

    def eval_obs(self, state, env):
        pd = self.phys_dim
        chirality = self.h3_l - self.h3_r
        idp = torch.eye(pd, dtype=self.dtype, device=self.device) / pd

        obs = dict()
        with torch.no_grad():
            for coord, site in state.sites.items():
                rdm1x1 = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B', 'C')).view(pd, pd, pd, pd, pd, pd)
                obs["chirality_dn"] = torch.einsum('ijkabc,ijkabc', rdm1x1, chirality)
                obs["chirality_dn"] = _cast_to_real(obs["chirality_dn"] * 1j)
                rdm2x2_ring = rdm_kagome.rdm2x2_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=('C'), sites_to_keep_01=(), sites_to_keep_11=('A'))
                obs["chirality_up"] = torch.einsum('ijklabcd,lijdab,kc', rdm2x2_ring, chirality, idp)
                obs["chirality_up"] = _cast_to_real(obs["chirality_up"] * 1j)

                rdm1x1_ab = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'B')).view(pd, pd, pd, pd)
                obs["avg_bonds_dn"] = torch.einsum('ijab,ijab', rdm1x1_ab, self.h2)
                rdm1x1_bc = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('B', 'C')).view(pd, pd, pd, pd)
                obs["avg_bonds_dn"] += torch.einsum('ijab,ijab', rdm1x1_bc, self.h2)
                rdm1x1_ac = rdm_kagome.rdm1x1_kagome(coord, state, env, sites_to_keep=('A', 'C')).view(pd, pd, pd, pd)
                obs["avg_bonds_dn"] += torch.einsum('ijab,ijab', rdm1x1_ac, self.h2)
                obs["avg_bonds_dn"] = _cast_to_real(obs["avg_bonds_dn"]) / 3.0

                rdm2x2_ab = rdm_kagome.rdm2x2_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=(), sites_to_keep_01=(), sites_to_keep_11=('A'))
                obs["avg_bonds_up"] = torch.einsum('ijklabcd,ilad,jb,kc', rdm2x2_ab, self.h2, idp, idp)
                obs["avg_bonds_up"] = 0
                rdm1x2_bc = rdm_kagome.rdm2x1_kagome(coord, state, env, sites_to_keep_00=('B'), sites_to_keep_10=('C'))
                obs["avg_bonds_up"] += torch.einsum('ijab,ijab', rdm1x2_bc, self.h2)
                rdm2x1_ac = rdm_kagome.rdm1x2_kagome(coord, state, env, sites_to_keep_00=('C'), sites_to_keep_01=('A'))
                obs["avg_bonds_up"] += torch.einsum('ijab,ijab', rdm2x1_ac, self.h2)
                obs["avg_bonds_up"] = _cast_to_real(obs["avg_bonds_up"]) / 3.0

        # prepare list with labels and values
        obs_labels = ["avg_bonds_dn", "avg_bonds_up", "chirality_dn", "chirality_up"]
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels
