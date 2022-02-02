from math import sqrt
import itertools
import numpy as np

import torch
import yast.yast as yast
import config as cfg
import groups.su3_abelian as su3

import ctm.pess_kagome_abelian.rdm_kagome as rdm_kagome

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
        self.dtype= settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.device
        self.phys_dim = 3
        self.j = j  # 
        self.k = k  # 
        self.h = h  # 3-site ring exchange coupling constant (imaginary part)

        # For now, only one-site
        self.obs_ops= self.get_obs_ops()
        self.id2, self.id3, self.perm2, self.perm2xI, \
            self.perm2_tri, self.perm3_l, self.perm3_r,\
            self.h2_tri, self.h3_tri, self.h_tri = self.get_h()

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
        obs_ops["CW"]= irrep.Cartan_Weyl() 

        return obs_ops

    def get_h(self):
        irrep= su3.SU3_DEFINING_U1xU1(self.engine)
        id2= yast.tensordot(irrep.I(),irrep.I(),([],[])).transpose(axes=(0,2,1,3))
        id3= yast.tensordot(id2, irrep.I(),([],[])).transpose(axes=(0,1,4,2,3,5))

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

        return id2, id3, perm2, perm2xI, perm2_tri, perm3_l, perm3_r, h2_tri, h3_tri, h_tri

    def eval_obs(self, state, env, force_cpu=False):
        chirality = 1j * (self.perm3_l - self.perm3_r)
        obs = dict()
        with torch.no_grad():
            norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.id3).to_number()
            obs["chirality_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                chirality).to_number() / norm
            obs["chirality_dn"] = _cast_to_real(obs["chirality_dn"])
            # obs["avg_bonds_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
            #     self.perm2_tri).to_number() / norm
            # obs["avg_bonds_dn"] = _cast_to_real(obs["avg_bonds_dn"]) / 3.0
            obs["P01_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                self.perm2xI).to_number() / norm
            obs["P01_dn"] = _cast_to_real(obs["P01_dn"])
            obs["P12_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                self.perm2xI.transpose(axes=(2,0,1,5,3,4))).to_number() / norm
            obs["P12_dn"] = _cast_to_real(obs["P12_dn"])
            obs["P20_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
                self.perm2xI.transpose(axes=(0,2,1,3,5,4))).to_number() / norm
            obs["P20_dn"] = _cast_to_real(obs["P20_dn"])

            rdm2x2_ring = rdm_kagome.rdm2x2_up_triangle_open((0,0), state, env, force_cpu=force_cpu)
            obs["chirality_up"] = yast.tensordot(rdm2x2_ring, chirality,\
                ([0,1,2,3,4,5], [3,4,5,0,1,2])).to_number()
            obs["chirality_up"] = _cast_to_real(obs["chirality_up"])
            # obs["avg_bonds_up"] = yast.tensordot(rdm2x2_ring, self.perm2_tri,\
            #     ([0,1,2,3,4,5], [3,4,5,0,1,2])).to_number()
            # obs["avg_bonds_up"] = _cast_to_real(obs["avg_bonds_up"]) / 3.0
            obs["P01_up"] = yast.tensordot(rdm2x2_ring, self.perm2xI,\
                ([0,1,2,3,4,5], [3,4,5,0,1,2])).to_number()
            obs["P01_up"] = _cast_to_real(obs["P01_up"])
            obs["P12_up"] = yast.tensordot(rdm2x2_ring, self.perm2xI.transpose(axes=(2,0,1,5,3,4)),\
                ([0,1,2,3,4,5], [3,4,5,0,1,2])).to_number()
            obs["P12_up"] = _cast_to_real(obs["P12_up"])
            obs["P20_up"] = yast.tensordot(rdm2x2_ring, self.perm2xI.transpose(axes=(0,2,1,3,5,4)),\
                ([0,1,2,3,4,5], [3,4,5,0,1,2])).to_number()
            obs["P20_up"] = _cast_to_real(obs["P20_up"])

            obs.update(self.eval_generators(state, env, force_cpu=force_cpu))

        # prepare list with labels and values
        obs_labels = ["m2_A", "m2_B", "m2_C", "chirality_dn", "chirality_up"]\
            + ["P01_dn", "P12_dn", "P20_dn", "P01_up", "P12_up", "P20_up" ]
    
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_obs_2x2subsystem(self, state, env, force_cpu=False):
        chirality = 1j * (self.perm3_l - self.perm3_r)
        obs = dict()
        with torch.no_grad():
            obs["chirality_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0),\
                state, env, chirality,force_cpu=force_cpu).to_number()
            obs["chirality_dn"] = _cast_to_real(obs["chirality_dn"])
            obs["e_t_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0), state,\
                env, self.h_tri,force_cpu=force_cpu).to_number()
            obs["e_t_dn"] = _cast_to_real(obs["e_t_dn"])
            obs["avg_bonds_dn"] = rdm_kagome.rdm2x2_dn_triangle_with_operator((0,0),\
                state, env, self.perm2_tri, force_cpu=force_cpu).to_number()
            obs["avg_bonds_dn"] = _cast_to_real(obs["avg_bonds_dn"])/3

            rdm2x2_ring = rdm_kagome.rdm2x2_up_triangle_open((0,0), state, env, force_cpu=force_cpu)
            obs["chirality_up"] = yast.tensordot(rdm2x2_ring, chirality,\
                ([0,1,2,3,4,5],[3,4,5,0,1,2])).to_number()
            obs["chirality_up"] = _cast_to_real(obs["chirality_up"])
            obs["e_t_up"] = yast.tensordot(rdm2x2_ring, self.h_tri,\
                ([0,1,2,3,4,5],[3,4,5,0,1,2])).to_number()
            obs["e_t_up"] = _cast_to_real(obs["e_t_up"])
            obs["avg_bonds_up"] = yast.tensordot(rdm2x2_ring, self.perm2_tri, \
                ([0,1,2,3,4,5],[3,4,5,0,1,2])).to_number()
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
        norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.id3).to_number()
        energy_dn = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, \
            self.h_tri).to_number() / norm
        energy_dn = _cast_to_real(energy_dn)
        return energy_dn

    def energy_triangles_2x2subsystem(self, state, env, force_cpu=False):
        r"""
        Evaluate energy per site by computing contributions from down
        and up triangles, both defined on 2x2 subsystem embedded in the environment. 
        """
        # intra-cell (down)
        energy_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_tri, force_cpu=force_cpu)
        energy_dn= energy_dn.to_number()
        energy_dn = _cast_to_real(energy_dn)
        # inter-cell (up)
        # rdm2x2_up = rdm_kagome.rdm2x2_up_triangle_open(\
        #     (0,0), state, env, force_cpu=force_cpu, sym_pos_def=False)
        rdm2x2_up = rdm_kagome.rdm2x2_kagome(\
            (0,0), state, env, sites_to_keep_00=(), sites_to_keep_10=('B'),\
            sites_to_keep_01=('A'), sites_to_keep_11=('C'),
            force_cpu=force_cpu, sym_pos_def=False)
        energy_up= yast.tensordot(rdm2x2_up, self.h_tri,([0,1,2,3,4,5],[3,4,5,0,1,2]))
        energy_up= energy_up.to_number()
        energy_up = _cast_to_real(energy_up)
        energy_per_site= (energy_dn + energy_up)/3
        return energy_dn, energy_up

    def energy_per_site_2x2subsystem(self,state,env,force_cpu=False):
        e_down, e_up= self.energy_triangles_2x2subsystem(state, env, force_cpu=force_cpu)
        e_per_site= (e_down+e_up)/3
        return e_per_site

    def eval_generators(self, state, env, force_cpu=False):
        """
        Evaluate generators in Cartan-Weyl basis (in fundamental irrep) of su(3). 
        The expectation values of generators are ordered as [T^+, T^-, T^z, V^+, V^-, U^+, U^-, Y].
        Compute the length of the vector of spontaneous magnetization, playing the role of 
        :math:`m^2 = |<\\vec{S}>|` in case of SU(2) models.
        
        Returns
        -------
        gens : dict
            keys "A", "B", and "C" hold length-8 vectors with expectation values
            of Cartan-Weyl generators on each of sites "A", "B", and "C".
            keys "m2_A", "m2_B", and "m2_C" hold expectation values of the lengths 
            of vectors :math:`|<\\vec{T}>|` where Ts are generators in Gell-Mann basis.  
        """
        gen_lr= [(0,"tp"), (1,"tm"), (3,"vp"), (4,"vm"), (5,"up"), (6,"um")]
        gen_center= [(2,"tz"), (7,"y")]
        irrep= su3.SU3_DEFINING_U1xU1(self.engine)
        gens = {"A": np.zeros(8, dtype=self.dtype), 
            "B": np.zeros(8, dtype=self.dtype),
            "C": np.zeros(8, dtype=self.dtype)}
        
        # vector of su(3) generators in Cartan-Weyl basis
        CW= self.obs_ops["CW"]
        with torch.no_grad():
            for site in ["A","B","C"]:
                rdm_1site= rdm_kagome.rdm1x1_kagome((0,0), state, env, sites_to_keep=(site),\
                    force_cpu=force_cpu)
                obs_CW= yast.tensordot( rdm_1site, CW, ([0,1],[2,1]) )

                # retrieve expectation values of generators by global charge of individual
                # generators
                for i,g_id in gen_lr:
                    # identify lowering and raising operators
                    if self.obs_ops[g_id].get_tensor_charge() in obs_CW.get_blocks_charge():
                        gens[site][i]= obs_CW[self.obs_ops[g_id].get_tensor_charge()].item()
                # identify center
                center_charge= self.obs_ops[gen_center[0][1]].get_tensor_charge()
                if center_charge in obs_CW.get_blocks_charge():
                    # i indexes expectation value inside block of center charge. The expectation
                    # value is then mapped back to its position in Cartan-Weyl basis
                    for i, g_index_g_id in enumerate(gen_center):
                        gens[site][g_index_g_id[0]]= obs_CW[center_charge][i]

                m2= yast.tensordot(irrep.G(), obs_CW, ([1],[0]))
                m2= yast.tensordot(obs_CW, m2, ([0],[0])).to_number()

                gens[f"m2_{site}"]= m2
        return gens

    # def eval_C2(self, state, env, force_cpu=False):
    #     pd = self.phys_dim
    #     idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
    #     irrep_su3_def = su3.SU3_DEFINING(dtype=self.dtype, device=self.device)
    #     c2 = irrep_su3_def.C2()
    #     c2_list = dict({"C2_dn": 0., "C2_up": 0.})
    #     with torch.no_grad():
    #         norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
    #         c2_list["C2_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, c2) / norm

    #         rdm2x2_up = rdm_kagome.rdm2x2_kagome((0,0), state, env, sites_to_keep_00=(),\
    #             sites_to_keep_10=('B'),sites_to_keep_01=('A'), sites_to_keep_11=('C'))
    #         c2_list["C2_up"] = torch.einsum('ijlabd,abdijl', rdm2x2_up, c2)
    #     return c2_list

    # def eval_C1(self, state, env, force_cpu=False):
    #     pd = self.phys_dim
    #     idp = torch.eye(pd, dtype=self.dtype, device=self.device)
    #     idp3 = torch.eye(pd**3, dtype=self.dtype, device=self.device)
    #     irrep_su3_def = su3.SU3_DEFINING(dtype=self.dtype, device=self.device)
    #     c1 = irrep_su3_def.C1()
    #     c1_dict = dict({"C1_AB_dn": 0., "C1_BC_dn": 0., "C1_AC_dn": 0.,\
    #         "C1_AB_up": 0., "C1_BC_up": 0., "C1_AC_up": 0.})
    #     with torch.no_grad():
    #         norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, idp3)
    #         c1_dict["C1_AB_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
    #             torch.einsum('ijab,kc->ijkabc', c1, idp)) / norm
    #         c1_dict["C1_BC_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
    #             torch.einsum('jkbc,ia->ijkabc', c1, idp)) / norm
    #         c1_dict["C1_AC_dn"] = rdm_kagome.trace1x1_dn_kagome((0,0), state, env,\
    #             torch.einsum('ikac,jb->ijkabc', c1, idp)) / norm

    #         rdm2x2_ab = rdm_kagome.rdm2x2_kagome((0,0), state, env, sites_to_keep_00=(),\
    #             sites_to_keep_10=('B'),sites_to_keep_01=('A'), sites_to_keep_11=())
    #         c1_dict["C1_AB_up"] = torch.einsum('ilad,ilad', rdm2x2_ab, c1)
    #         rdm1x2_bc = rdm_kagome.rdm1x2_kagome((0,0), state, env, sites_to_keep_00=('B'),\
    #             sites_to_keep_01=('C'))
    #         c1_dict["C1_BC_up"] = torch.einsum('ijab,ijab', rdm1x2_bc, c1)
    #         rdm2x1_ac = rdm_kagome.rdm2x1_kagome((0,0), state, env, sites_to_keep_00=('A'),\
    #             sites_to_keep_10=('C'))
    #         c1_dict["C1_AC_up"] = torch.einsum('ijab,ijab', rdm2x1_ac, c1)

 
    #         c1_dict["total_C1_dn"] = c1_dict["C1_AB_dn"] + c1_dict["C1_BC_dn"] + c1_dict["C1_AC_dn"]
    #         c1_dict["total_C1_up"] = c1_dict["C1_AB_up"] + c1_dict["C1_BC_up"] + c1_dict["C1_AC_up"]

    #     return c1_dict