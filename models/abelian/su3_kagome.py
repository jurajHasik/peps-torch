from math import sqrt
import itertools
import numpy as np

import torch
import yamps.yast as yast
import config as cfg
import groups.su3_abelian as su3

from ctm.generic.env import ENV
from ctm.pess_kagome import rdm_kagome

def _cast_to_real(t, check=True, imag_eps=1.0e-8):
    if check and t.is_complex():
        assert abs(t.imag) < imag_eps,"unexpected imaginary part "+str(t.imag)
    return t.real if t.is_complex() else t

class KAGOME_SU3_U1xU1():
    def __init__(self, settings, j=0.0, k=1.0, h=0.0, global_args=cfg.global_args):
        r"""
        :param j: 2-site permutation coupling
        :param k: real part of 3-site ring exchange coupling
        :param h: imaginary part of 3-site ring exchange coupling

        The SU(3) Hamiltonian on Kagome lattice with explicit U(1)xU(1) symmetry 
        
        .. math:: H = J \sum_ij P_ij + K \sum_t_up,t_down (P_ijk + P^-1_ijk) 
                                     + ih \sum_t_up,t_down (P_ijk - P^-1_ijk)
    
        or in parametrization through angles

        .. math:: H = cos \phi \sum_ij P_ij 
                      + sin \phi \sum_t_up,t_down exp(i\theta) P_ijk + exp(-i\theta)P^-1_ijk
        
        where :math:`J = cos \phi,\ K = sin \phi cos \theta,\ h= sin \phi sin \theta`.
        The \phi= 0.5pi and \theta=0 corresponds to the AKLT point.
        """
        assert settings.sym.NSYM==2, "U(1)xU(1) abelian symmetry is assumed"
        self.engine= settings
        self.dtype=settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim = 3
        self.j = j  # 
        self.k = k  # 
        self.h = h  # 3-site ring exchange coupling constant (imaginary part)

        # For now, only one-site
        self.obs_ops = self.get_obs_ops()
        self.perm2, self.perm2_tri, self.perm3_l, self.perm3_r, self.h2_tri, \
            self.h3_tri, self.h_tri = self.get_h()

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su3.SU3_DEFINING_U1xU1(self.engine)
        obs_ops["tz"] = irrep.TZ()
        obs_ops["tp"] = irrep.TP()
        obs_ops["tm"] = irrep.TM()
        obs_ops["vp"] = irrep.VP()
        obs_ops["vm"] = irrep.VM()
        obs_ops["up"] = irrep.UP()
        obs_ops["um"] = irrep.UM()
        obs_ops["y"] = irrep.Y()

        return obs_ops

    def get_h(self):
        irrep= su3.SU3_DEFINING_U1xU1(self.engine)
        id2= yast.tensordot(irrep.I(),irrep.I(),([],[])).transpose(axes=(0,2,1,3))
        
        perm2 = 2 * irrep.C1() + (1./3) * id2
        perm2 = perm2.remove_zero_blocks(rtol=1e-14, atol=0)
        
        # 0    1
        # |-P2-|                        0  1   2
        # 2    3          =>  2<->3  => |--P3--|
        #      0    1->3                3  4   5
        #      |-P2-|
        #      2->4 3->5
        perm3_l = yast.tensordot(perm2, perm2, ([3],[0])).transpose(axes=(0,1,3,2,4,5))
        perm3_r = yast.tensordot(perm2, perm2, ([3],[1])).transpose(axes=(0,3,1,2,4,5))

        # 0    1     0->4    4->2    0  1     2 
        # |-P2-| (x) I    => 2->3 => |--P2xI--|
        # 2    3     1->5    3->4    3  4     5
        perm2xI= yast.tensordot(perm2, irrep.I(), ([],[])).transpose(axes=(0,1,4,2,3,5))
        perm2_tri = perm2xI + perm2xI.transpose(axes=(2,0,1,5,3,4)) \
            + perm2xI.transpose(axes=(0,2,1,3,5,4))

        h2_tri = self.j * perm2_tri
        h3_tri = (self.k + self.h * 1j) * perm3_l + (self.k - self.h * 1j) * perm3_r
        h_tri = (h2_tri + h3_tri)

        return perm2, perm2_tri, perm3_l, perm3_r, h2_tri, h3_tri, h_tri

    def energy_1site(self, state, env):
        pd = self.phys_dim
        energy = 0.0
        idp = torch.eye(pd, dtype=self.dtype, device=self.device)
        
        # intra-cell (down triangle)
        norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, torch.einsum('ia,jb,kc->ijkabc', idp, idp, idp))
        energy += rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.h_tri) / norm

        # inter-cell (up triangle)
        rdm2x2_ring = rdm_kagome.rdm2x2_kagome((0,0), state, env, sites_to_keep_00=('B'), sites_to_keep_10=('C'),
                                                   sites_to_keep_01=(), sites_to_keep_11=('A'))
        energy += torch.einsum('ijlabd,lijdab', rdm2x2_ring, self.h_tri)
        energy_per_site = energy / (len(state.sites.items()) * 3.0)
        energy_per_site = _cast_to_real(energy_per_site)
        return energy_per_site

    def eval_obs(self, state, env, force_cpu=False):
        pd = self.phys_dim
        chirality = 1j * (self.perm3_l - self.perm3_r)
        idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
        obs = dict()
        with torch.no_grad():
            norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
            obs["chirality_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, chirality) / norm
            obs["chirality_dn"] = _cast_to_real(obs["chirality_dn"])
            obs["avg_bonds_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.perm2_tri) / norm
            obs["avg_bonds_dn"] = _cast_to_real(obs["avg_bonds_dn"]) / 3.0

            rdm2x2_ring = rdm_kagome.rdm2x2_up_triangle_open((0,0), state, env, force_cpu=force_cpu)
            obs["chirality_up"] = torch.einsum('ijlabc,ijlabc', rdm2x2_ring, chirality)
            obs["chirality_up"] = _cast_to_real(obs["chirality_up"])
            obs["avg_bonds_up"] = torch.einsum('ijlabc,ijlabc', rdm2x2_ring, self.perm2_tri)
            obs["avg_bonds_up"] = _cast_to_real(obs["avg_bonds_up"]) / 3.0

            obs.update(self.eval_generators(state, env, force_cpu=force_cpu))

        # prepare list with labels and values
        obs_labels = ["avg_bonds_dn", "avg_bonds_up", "chirality_dn", "chirality_up"]\
            + ["m2_A", "m2_B", "m2_C"]
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_obs_2x2subsystem(self, state, env, force_cpu=False):
        chirality = 1j * (self.perm3_l - self.perm3_r)
        obs = dict()
        with torch.no_grad():
            obs["chirality_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, env, chirality,\
                force_cpu=force_cpu)
            obs["chirality_dn"] = _cast_to_real(obs["chirality_dn"])
            obs["e_t_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, env, self.h_tri,\
                force_cpu=force_cpu)
            obs["e_t_dn"] = _cast_to_real(obs["e_t_dn"])
            obs["avg_bonds_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state, env, self.perm2_tri,\
                force_cpu=force_cpu)
            obs["avg_bonds_dn"] = _cast_to_real(obs["avg_bonds_dn"])/3

            rdm2x2_ring = rdm_kagome.rdm2x2_up_triangle_open((0,0), state, env, force_cpu=force_cpu)
            obs["chirality_up"] = torch.einsum('ijlabc,abcijl', rdm2x2_ring, chirality)
            obs["chirality_up"] = _cast_to_real(obs["chirality_up"])
            obs["e_t_up"] = torch.einsum('ijlabc,abcijl', rdm2x2_ring, self.h_tri)
            obs["e_t_up"] = _cast_to_real(obs["e_t_up"])
            obs["avg_bonds_up"] = torch.einsum('ijlabc,abcijl', rdm2x2_ring, self.perm2_tri)
            obs["avg_bonds_up"] = _cast_to_real(obs["avg_bonds_up"])/3

            obs.update(self.eval_generators(state, env, force_cpu=force_cpu))

        # prepare list with labels and values
        obs_labels = ["e_t_dn","e_t_up","avg_bonds_dn","avg_bonds_up","chirality_dn","chirality_up"]\
            + ["m2_A", "m2_B", "m2_C"]
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def energy_down_t_1x1subsystem(self, state, env, force_cpu=False):
        r"""
        Evaluate the energy contribution from the down triangle on 1x1 subsystem
        embedded in the environment.
        """
        pd = self.phys_dim
        idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
        norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
        energy_dn = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.h_tri) / norm
        return energy_dn

    def energy_triangles_2x2subsystem(self, state, env, force_cpu=False):
        r"""
        Evaluate energy per site by computing contributions from down
        and up triangles, both defined on 2x2 subsystem embedded in the environment. 
        """
        # intra-cell (down)
        energy_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_tri, force_cpu=force_cpu)
        energy_dn = _cast_to_real(energy_dn)
        # inter-cell (up)
        # rdm2x2_up = rdm_kagome.rdm2x2_up_triangle_open(\
        #     (0,0), state, env, force_cpu=force_cpu, sym_pos_def=False)
        rdm2x2_up = rdm_kagome.rdm2x2_kagome(\
            (0,0), state, env, sites_to_keep_00=(), sites_to_keep_10=('B'),\
            sites_to_keep_01=('A'), sites_to_keep_11=('C'),
            force_cpu=force_cpu, sym_pos_def=False)
        energy_up = torch.einsum('ijlabc,abcijl', rdm2x2_up, self.h_tri)
        energy_up = _cast_to_real(energy_up)
        energy_per_site= (energy_dn + energy_up)/3
        return energy_dn, energy_up

    def energy_per_site_2x2subsystem(self,state,env,force_cpu=False):
        e_down, e_up= self.energy_triangles_2x2subsystem(state, env, force_cpu=force_cpu)
        e_per_site= (e_down+e_up)/3
        return e_per_site

    def eval_generators(self, state, env, force_cpu=False):
        pd = self.phys_dim
        idp2= torch.eye(pd**2, dtype=self.dtype, device=self.device)
        idp3= torch.eye(pd**3, dtype=self.dtype, device=self.device)
        idp2= idp2.view(pd,pd,pd,pd)
        gens = {"A": torch.zeros(8, dtype=self.dtype, device=self.device), 
            "B": torch.zeros(8, dtype=self.dtype, device=self.device),
            "C": torch.zeros(8, dtype=self.dtype, device=self.device)}
        
        # vector of su(3) generators in Gell-Mann basis
        norm= _cast_to_real(rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3))
        J= self.obs_ops["J"]
        with torch.no_grad():
            # for site_type in ['A', 'B', 'C']:
            #     rdm1x1 = rdm_kagome.rdm1x1_kagome((0,0), state, env, sites_to_keep=(site_type))
            #     for i in range(J.size(0)):
            #         gens[site_type][i]= torch.einsum('ij,ji', rdm1x1, J[i,:,:])
            
            for i in range(J.size(0)):
                gens['A'][i]= _cast_to_real(rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                    torch.einsum('ab,ijkl->aijbkl',J[i,:,:],idp2))) / norm
            for i in range(J.size(0)):
                gens['B'][i]= _cast_to_real(rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                    torch.einsum('ab,ijkl->iajkbl',J[i,:,:],idp2))) / norm
            for i in range(J.size(0)):
                gens['C'][i]= _cast_to_real(rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                    torch.einsum('ab,ijkl->ijaklb',J[i,:,:],idp2))) / norm
            for site_type in ['A', 'B', 'C']:
                gens[f"m2_{site_type}"]= torch.dot(gens[site_type],gens[site_type])
        return gens

    def eval_C2(self, state, env, force_cpu=False):
        pd = self.phys_dim
        idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
        irrep_su3_def = su3.SU3_DEFINING(dtype=self.dtype, device=self.device)
        c2 = irrep_su3_def.C2()
        c2_list = dict({"C2_dn": 0., "C2_up": 0.})
        with torch.no_grad():
            norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
            c2_list["C2_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, c2) / norm

            rdm2x2_up = rdm_kagome.rdm2x2_kagome((0,0), state, env, sites_to_keep_00=(),\
                sites_to_keep_10=('B'),sites_to_keep_01=('A'), sites_to_keep_11=('C'))
            c2_list["C2_up"] = torch.einsum('ijlabd,abdijl', rdm2x2_up, c2)
        return c2_list

    def eval_C1(self, state, env, force_cpu=False):
        pd = self.phys_dim
        idp = torch.eye(pd, dtype=self.dtype, device=self.device)
        idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
        irrep_su3_def = su3.SU3_DEFINING(dtype=self.dtype, device=self.device)
        c1 = irrep_su3_def.C1()
        c1_dict = dict({"C1_AB_dn": 0., "C1_BC_dn": 0., "C1_AC_dn": 0.,\
            "C1_AB_up": 0., "C1_BC_up": 0., "C1_AC_up": 0.})
        with torch.no_grad():
            norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
            c1_dict["C1_AB_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                torch.einsum('ijab,kc->ijkabc', c1, idp)) / norm
            c1_dict["C1_BC_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                torch.einsum('jkbc,ia->ijkabc', c1, idp)) / norm
            c1_dict["C1_AC_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                torch.einsum('ikac,jb->ijkabc', c1, idp)) / norm

            rdm2x2_ab = rdm_kagome.rdm2x2_kagome((0,0), state, env, sites_to_keep_00=(),\
                sites_to_keep_10=('B'),sites_to_keep_01=('A'), sites_to_keep_11=())
            c1_dict["C1_AB_up"] = torch.einsum('ilad,ilad', rdm2x2_ab, c1)
            rdm1x2_bc = rdm_kagome.rdm1x2_kagome((0,0), state, env, sites_to_keep_00=('B'),\
                sites_to_keep_01=('C'))
            c1_dict["C1_BC_up"] = torch.einsum('ijab,ijab', rdm1x2_bc, c1)
            rdm2x1_ac = rdm_kagome.rdm2x1_kagome((0,0), state, env, sites_to_keep_00=('A'),\
                sites_to_keep_10=('C'))
            c1_dict["C1_AC_up"] = torch.einsum('ijab,ijab', rdm2x1_ac, c1)

 
            c1_dict["total_C1_dn"] = c1_dict["C1_AB_dn"] + c1_dict["C1_BC_dn"] + c1_dict["C1_AC_dn"]
            c1_dict["total_C1_up"] = c1_dict["C1_AB_up"] + c1_dict["C1_BC_up"] + c1_dict["C1_AC_up"]

        return c1_dict
