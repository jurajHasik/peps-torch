from math import sqrt
import itertools
import numpy as np
import torch
import yastn.yastn as yastn
import config as cfg
import groups.su2_abelian as su2
import ctm.pess_kagome_abelian.rdm_kagome as rdm_kagome
from ctm.generic_abelian.rdm import _cast_to_real
from ctm.generic import transferops


class KAGOME_U1():
    def __init__(self, settings, j1=1., JD=0, j1sq=0., j2=0., j2sq=0., jtrip=0,\
        jperm=0+0j, h=0, phys_dim=2, global_args=cfg.global_args):
        r"""
        :param settings: YAST configuration
        :type settings: NamedTuple or SimpleNamespace (TODO link to definition)
        :param j1: nearest-neighbour spin-spin interaction
        :type j1: float
        :param JD: Dzyaloshinskii-Moriya interaction
        :type JD: float
        :param jtrip: scalar chirality
        :type jtrip: float
        :param jperm: triangle exchange
        :type jperm: complex
        :param global_args: global configuration
        :type global_args: GLOBALARGS

        Build spin-1/2 Hamiltonian on Kagome lattice

        .. math::

            H &= J_1 \sum_{<ij>} S_i.S_j
                + J_2 \sum_{<<ij>>} S_i.S_j
                - J_{trip} \sum_t (S_{t_1} \times S_{t_2}).S_{t_3} \\
                &+ J_{perm} \sum_t P_t + J^*_{perm} \sum_t P^{-1}_t

        where the first sum runs over the pairs of sites `i,j` 
        which are nearest-neighbours (denoted as `<.,.>`), the second sum runs over 
        pairs of sites `i,j` which are next nearest-neighbours (denoted as `<<.,.>>`).
        The :math:`J_{trip}` and :math:`J_{perm}` terms represent scalar chirality 
        and triangle exchange respectively. The :math:`\sum_t` runs over all triangles.
        The sites :math:`t_1,t_2,t_3` on the triangles are always ordered anti-clockwise.  
        """
        # H = J_1 \sum_{<ij>} S_i.S_j + J_{1sq} \sum_{<ij>} (S_i.S_j)^2
        #     + J_2 \sum_{<<ij>>} S_i.S_j + J_{2sq} \sum_{<<ij>>} (S_i.S_j)^2
        #     - J_{trip} \sum_t (S_{t_1} \times S_{t_2}).S_{t_3}
        #     + J_{perm} \sum_t P_t + J*_{perm} \sum_t P^{-1}_t
        assert settings.sym.NSYM==1, "U(1) abelian symmetry is assumed"
        self.engine= settings
        self.dtype= settings.default_dtype
        self.device='cpu' if not hasattr(settings, 'device') else settings.default_device
        self.phys_dim = phys_dim
        self.j1 = j1
        self.JD = JD
        self.j1sq = j1sq
        self.jtrip = jtrip
        self.h = h
        # j2, j2sq, jperm not implemented
        self.j2 = j2
        self.j2sq = j2sq
        self.jperm = jperm

        irrep = su2.SU2_U1(self.engine, self.phys_dim)

        Id1= irrep.I()
        Id2= yastn.tensordot(Id1,Id1,([],[])).transpose(axes=(0,2,1,3))
        self.Id3_t= yastn.tensordot(Id2,Id1,([],[])).transpose(axes=(0,1,4,2,3,5))\
            .fuse_legs(axes=((0,1,2), (3,4,5)))
        self.Id2_t= Id2.fuse_legs(axes=((0,1), (2,3)))

        SS= irrep.SS(zpm=(1., 1., 1.))
        SS_JD= self.j1*SS if abs(self.JD) else irrep.SS(zpm=(j1, j1+1j*JD, j1-1j*JD)) 
        self.SSnnId= yastn.tensordot(SS_JD,Id1,([],[])).transpose(axes=(0,1,4,2,3,5))
        SSnn_t= self.SSnnId + self.SSnnId.transpose(axes=(1,2,0, 4,5,3)) \
            + self.SSnnId.transpose(axes=(2,0,1, 5,3,4))

        mag_field= irrep.SZ()
        mag_field= yastn.tensordot(mag_field,Id1,([],[])).transpose(axes=(0,2,1,3))
        mag_field= yastn.tensordot(mag_field,Id1,([],[])).transpose(axes=(0,1,4,2,3,5))
        mag_field=mag_field + mag_field.transpose(axes=(1,2,0, 4,5,3)) \
            + mag_field.transpose(axes=(2,0,1, 5,3,4))

        if self.jtrip != 0:
            assert self.dtype=="complex128" or self.dtype=="complex64","jtrip requires complex dtype"
            smsp= yastn.tensordot(irrep.SM(),irrep.SP(),([],[])).transpose(axes=(0,2,1,3))
            spsm= yastn.tensordot(irrep.SP(),irrep.SM(),([],[])).transpose(axes=(0,2,1,3))
            SxSS_t= yastn.tensordot(smsp-spsm,irrep.SZ()/(2j),([],[]))\
                .transpose(axes=(0,1,4,2,3,5))
            SxSS_t= SxSS_t+ SxSS_t.transpose(axes=(1,2,0, 4,5,3)) \
                + SxSS_t.transpose(axes=(2,0,1, 5,3,4))
        else:
            SxSS_t= 0*self.Id3_t.unfuse_legs(axes=(0,1))

        perm2= 2*SS + (1./2) * Id2
        perm2 = perm2.remove_zero_blocks(rtol=1e-14, atol=0)
        
        # 0    1
        # |-P2-|                        0  1   2
        # 2    3          =>  2<->3  => |--P3--|
        #      0    1->3                3  4   5
        #      |-P2-|
        #      2->4 3->5
        self.P_triangle = yastn.tensordot(perm2, perm2, ([3],[0])).transpose(axes=(0,1,3,2,4,5))
        self.P_triangle_inv = yastn.tensordot(perm2, perm2, ([3],[1])).transpose(axes=(0,3,1,2,4,5))

        self.h_triangle= SSnn_t + self.h*mag_field + self.jtrip*SxSS_t
        if self.jperm!=0+0j:
            assert self.dtype=="complex128" or self.dtype=="complex64","jperm requires complex dtype"
            self.h_triangle = self.h_triangle + self.jperm * self.P_triangle\
                + self.jperm.conjugate() * self.P_triangle_inv

        szId2= yastn.tensordot(irrep.SZ(),Id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        spId2= yastn.tensordot(irrep.SP(),Id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        smId2= yastn.tensordot(irrep.SM(),Id2,([],[])).transpose(axes=(0,2,3,1,4,5))
        self.obs_ops= {
            "sz_0": szId2, "sp_0": spId2, "sm_0": smId2,\
            "sz_1": szId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sp_1": spId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sm_1": smId2.transpose(axes=(1,2,0, 4,5,3)),\
            "sz_2": szId2.transpose(axes=(2,0,1, 5,3,4)),\
            "sp_2": spId2.transpose(axes=(2,0,1, 5,3,4)),\
            "sm_2": smId2.transpose(axes=(2,0,1, 5,3,4)),
            }
        self.SS01= self.SSnnId
        self.SS12= self.SSnnId.transpose(axes=(1,2,0, 4,5,3))
        self.SS02= self.SSnnId.transpose(axes=(2,0,1, 5,3,4))

    def energy_down_t_1x1subsystem(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME_ABELIAN
        :type env: ENV_ABELIAN
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from down triangle within 1x1 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome_abelian.rdm_kagome.trace1x1_dn_kagome`.
        """
        norm = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, self.id3)
        norm = _cast_to_real(norm,  **kwargs).to_number()
        e_dn = rdm_kagome.trace1x1_dn_kagome((0,0), state, env, \
            self.h_triangle) / norm
        return _cast_to_real(e_dn,  **kwargs).to_number()

    def energy_down_t_2x2subsystem(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME_ABELIAN
        :type env: ENV_ABELIAN
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from down triangle within 2x2 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome_abelian.rdm_kagome.rdm2x2_dn_triangle_with_operator`.
        """
        e_dn,_ = rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, \
            self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))), force_cpu=force_cpu, **kwargs)
        return _cast_to_real(e_dn, **kwargs).to_number()

    def energy_up_t_2x2subsystem(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME_ABELIAN
        :type env: ENV_ABELIAN
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from up triangle within 2x2 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome_abelian.rdm_kagome.rdm2x2_up_triangle_open`.
        """
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu,\
            **kwargs)
        e_up=yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),\
            self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0]))
        return _cast_to_real(e_up,  **kwargs).to_number()

    def energy_triangle_dn_NoCheck(self, state, env, force_cpu=False):
        e_dn,_ = rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
            self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))), force_cpu=force_cpu)
        e_dn= e_dn.to_number()
        return e_dn

    def energy_triangle_up_NoCheck(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
        e_up=yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),\
            self.h_triangle.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
        return e_up

    #define operators for correlation functions
    def sz_0_op(self):
        op=self.obs_ops["sz_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sz_1_op(self):
        op=self.obs_ops["sz_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sz_2_op(self):
        op=self.obs_ops["sz_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_0_op(self):
        op=self.obs_ops["sp_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_1_op(self):
        op=self.obs_ops["sp_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sp_2_op(self):
        op=self.obs_ops["sp_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_0_op(self):
        op=self.obs_ops["sm_0"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_1_op(self):
        op=self.obs_ops["sm_1"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def sm_2_op(self):
        op=self.obs_ops["sm_2"].fuse_legs(axes=((0,1,2),(3,4,5)))
        return op

    def SS01_op(self):
        op=self.SS01.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def SS12_op(self):
        op=self.SS12.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op
    def SS02_op(self):
        op=self.SS02.fuse_legs(axes=((0,1,2),(3,4,5)))
        return op


    # Observables
    def eval_obs(self,state,env,force_cpu=True, cast_real=False, disp_corre_len=False):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME_ABELIAN
        :type env: ENV_ABELIAN
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :param cast_real: if ``False`` keep imaginary part of energy contributions 
                          from up and down triangles
        :type cast_real: bool
        :param disp_corre_len: compute correlation lengths from transfer matrices
        :type disp_corre_len: bool
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Evaluate observables for IPESS_KAGOME wavefunction. In particular

            * energy contributions from up and down triangles
            * vector of spontaneous magnetization :math:`\langle \vec{S} \rangle` 
              for each site and its length :math:`m=|\langle \vec{S} \rangle|` 
            * nearest-neighbour spin-spin correlations for all bonds in the unit cell
            * (optionally) correlation lengths

        """
        obs= {"e_t_dn": 0, "e_t_up": 0, "m2_0": 0, "m2_1": 0, "m2_2": 0}
        with torch.no_grad():
            if cast_real:
                e_t_dn= self.energy_triangle_dn(state, env, force_cpu=force_cpu)
                e_t_up= self.energy_triangle_up(state, env, force_cpu=force_cpu)
            else:
                e_t_dn = self.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
                e_t_up = self.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
            obs["e_t_dn"]= e_t_dn
            obs["e_t_up"]= e_t_up

            for label in self.obs_ops.keys():
                op= self.obs_ops[label]
                obs_val,_ =rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, op,\
                    force_cpu=force_cpu)
                obs_val= obs_val.to_number()
                obs[f"{label}"]= obs_val

            for i in range(3):
                #obs[f"m_{i}"]= sqrt(_cast_to_real(obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]))
                obs[f"m2_{i}"]= obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]
 
            # nn S.S pattern
            SS_dn_01,_= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
                self.SS01, force_cpu=force_cpu)
            SS_dn_01= SS_dn_01.to_number()
            SS_dn_12,_= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
                self.SS12, force_cpu=force_cpu)
            SS_dn_12= SS_dn_12.to_number()
            SS_dn_02,_= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
                self.SS02, force_cpu=force_cpu)
            SS_dn_02= SS_dn_02.to_number()
            
            rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
            #bb=yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),self.Id3_t,([0,1],[1,0])).to_number()
            #print(bb)
            SS_up_01= yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),\
                self.SS01.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
            SS_up_12= yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),\
                self.SS12.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()
            SS_up_02= yastn.tensordot(rdm_up.fuse_legs(axes=((0,1,2),(3,4,5))),\
                self.SS02.fuse_legs(axes=((0,1,2),(3,4,5))),([0,1],[1,0])).to_number()

            obs.update({"SS_dn_01": SS_dn_01, "SS_dn_12": SS_dn_12, "SS_dn_02": SS_dn_02,\
                "SS_up_01": SS_up_01, "SS_up_12": SS_up_12, "SS_up_02": SS_up_02 })
            
            if disp_corre_len: 
                obs= eval_corr_lengths(state, env, obs=obs)    
                
        # prepare list with labels and values
        return list(obs.values()), list(obs.keys())

    # TODO transfer operator with explicit U(1) symm
    def eval_corr_lengths(state, env, coord=(0,0), obs=None):
        #convert to dense env and compute transfer operator spectrum
        env_dense= env.to_dense(state)
        state_dense= state.to_dense()
        Ns=3
        direction=(1,0)
        Lx= transferops.get_Top_spec(Ns, coord, direction, state_dense, env_dense)
        direction=(0,1)
        Ly= transferops.get_Top_spec(Ns, coord, direction, state_dense, env_dense)
        lambdax_0=torch.abs(Lx[0,0]+1j*Lx[0,1])
        lambdax_1=torch.abs(Lx[1,0]+1j*Lx[1,1])
        lambdax_2=torch.abs(Lx[2,0]+1j*Lx[2,1])
        lambday_0=torch.abs(Ly[0,0]+1j*Ly[0,1])
        lambday_1=torch.abs(Ly[1,0]+1j*Ly[1,1])
        lambday_2=torch.abs(Ly[2,0]+1j*Ly[2,1])
        correlen_x=-1/(torch.log(lambdax_1/lambdax_0))
        correlen_y=-1/(torch.log(lambday_1/lambday_0))
        corr_len_obs= {"lambdax_0": lambdax_0, "lambdax_1": lambdax_1,\
            "lambdax_2": lambdax_2,"lambday_0": lambday_0, "lambday_1": lambday_1,\
            "lambday_2": lambday_2,"correlen_x": correlen_x, "correlen_y": correlen_y}
        if obs:
            obs.update()
            return obs
        else:
            return corr_len_obs