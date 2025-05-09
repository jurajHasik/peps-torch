import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.pess_kagome import rdm_kagome
from ctm.generic import corrf
import groups.su2 as su2
from math import sqrt
from numpy import exp
import itertools

def _cast_to_real(t, check=True, imag_eps=1.0e-12):
    if t.is_complex():
        assert abs(t.imag) < imag_eps,"unexpected imaginary part "+str(t.imag)
        return t.real
    return t

class S1_KAGOME():

    def __init__(self, j1=0., j1sq=0., j2=0., j2sq=0., jtrip=0., jperm=0., global_args=cfg.global_args):
        r"""
        H = J_1 \sum_{<ij>} S_i.S_j + J_{1sq} \sum_{<ij>} (S_i.S_j)^2
            + J_2 \sum_{<<ij>>} S_i.S_j + J_{2sq} \sum_{<<ij>>} (S_i.S_j)^2
            - J_{trip} \sum_t (S_{t_1} \times S_{t_2}).S_{t_3}
            + J_{perm} \sum_t P_t + J*_{perm} \sum_t P^{-1}_t
        """
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = 3
        self.j1 = j1
        self.j1sq = j1sq
        self.j2 = j2
        self.j2sq = j2sq
        self.jtrip = jtrip
        self.jperm = jperm
        
        irrep = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)

        Id1= irrep.I()
        self.Id3_t= torch.eye(self.phys_dim**3, dtype=self.dtype, device=self.device)
        SS= irrep.SS()
        self.SSnnId= torch.einsum('ijkl,ab->ijaklb',SS,Id1)
        SSnn_t= self.SSnnId + self.SSnnId.permute(1,2,0, 4,5,3) + self.SSnnId.permute(2,0,1, 5,3,4)
        SS2= torch.einsum('ijab,abkl->ijkl',SS,SS)
        SS2nnId= torch.einsum('ijkl,ab->ijaklb',SS2,Id1)
        SS2nn_t= SS2nnId + SS2nnId.permute(1,2,0, 4,5,3) + SS2nnId.permute(2,0,1, 5,3,4)

        if jtrip != 0:
            assert self.dtype==torch.complex128 or self.dtype==torch.complex64,"jtrip requires complex dtype"  
        Svec= irrep.S()
        levicivit3= torch.zeros(3,3,3, dtype=self.dtype, device=self.device)
        levicivit3[0,1,2]=levicivit3[1,2,0]=levicivit3[2,0,1]=1.
        levicivit3[0,2,1]=levicivit3[2,1,0]=levicivit3[1,0,2]=-1.
        SxSS_t= torch.einsum('abc,bij,ckl,amn->ikmjln',levicivit3,Svec,Svec,Svec)

        permute_triangle = torch.zeros([self.phys_dim]*6, dtype=self.dtype, device=self.device)
        permute_triangle_inv = torch.zeros([self.phys_dim]*6, dtype=self.dtype, device=self.device)
        for i in range(self.phys_dim):
            for j in range(self.phys_dim):
                for k in range(self.phys_dim):
                    # anticlockwise (direct)
                    #
                    # 2---1 <- 0---2
                    #  \ /      \ /
                    #   0        1
                    permute_triangle[i, j, k, j, k, i] = 1.
                    # clockwise (inverse)
                    permute_triangle_inv[i, j, k, k, i, j] = 1.

        self.h_triangle= self.j1*SSnn_t + self.j1sq*SS2nn_t + self.jtrip*SxSS_t \
            + self.jperm * permute_triangle + (self.jperm * permute_triangle_inv).conj()
        szId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SZ(),Id1,Id1).contiguous()
        spId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SP(),Id1,Id1).contiguous()
        smId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SM(),Id1,Id1).contiguous()
        self.obs_ops= {
            "sz_0": szId2, "sp_0": spId2, "sm_0": smId2,\
            "sz_1": szId2.permute(1,2,0, 4,5,3).contiguous(),\
            "sp_1": spId2.permute(1,2,0, 4,5,3).contiguous(),\
            "sm_1": smId2.permute(1,2,0, 4,5,3).contiguous(),\
            "sz_2": szId2.permute(2,0,1, 5,3,4).contiguous(),\
            "sp_2": spId2.permute(2,0,1, 5,3,4).contiguous(),\
            "sm_2": smId2.permute(2,0,1, 5,3,4).contiguous(),
            }


    # Energy terms
    def energy_triangle_dn(self, state, env, force_cpu=False):
        e_dn, e_norm= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
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

    def eval_obs(self,state,env,force_cpu=True):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]
        """
        norm_wf = rdm.rdm1x1((0, 0), state, env, operator=self.Id3_t)
        obs= {"e_t_dn": 0, "e_t_up": 0, "m_0": 0, "m_1": 0, "m_2": 0}
        with torch.no_grad():
            e_t_dn= self.energy_triangle_dn(state, env, force_cpu=force_cpu)
            e_t_up= self.energy_triangle_up(state, env, force_cpu=force_cpu)
            obs["e_t_dn"]= e_t_dn
            obs["e_t_up"]= e_t_up

            for label in self.obs_ops.keys():
                op= self.obs_ops[label].view(self.phys_dim**3, self.phys_dim**3)
                obs_val= rdm.rdm1x1((0, 0), state, env, operator=op) / norm_wf
                obs[f"{label}"]= obs_val #_cast_to_real(obs_val)

            for i in range(3):
                obs[f"m_{i}"]= sqrt(_cast_to_real(
                    obs[f"sz_{i}"]**2\
                    +0.25*(obs[f"sp_{i}"] + obs[f"sm_{i}"])**2
                    +0.25*( -1.0j*(obs[f"sp_{i}"] - obs[f"sm_{i}"]) )**2
                    )
                )
 
            # nn S.S pattern
            SS_dn_01, _= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId, force_cpu=force_cpu)
            SS_dn_12, _= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId.permute(2,1,0, 5,4,3).contiguous(),\
                force_cpu=force_cpu)
            SS_dn_02, _= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId.permute(0,2,1, 3,5,4).contiguous(),\
                force_cpu=force_cpu)
            rdm_up= rdm_kagome.rdm2x2_up_triangle_open(\
                (0, 0), state, env, force_cpu=force_cpu)
            SS_up_01= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId )
            SS_up_12= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId.permute(2,1,0, 5,4,3) )
            SS_up_02= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId.permute(0,2,1, 3,5,4) )

            obs.update({"SS_dn_01": SS_dn_01, "SS_dn_12": SS_dn_12, "SS_dn_02": SS_dn_02,\
                "SS_up_01": SS_up_01, "SS_up_12": SS_up_12, "SS_up_02": SS_up_02 })

        # prepare list with labels and values
        return list(obs.values()), list(obs.keys())