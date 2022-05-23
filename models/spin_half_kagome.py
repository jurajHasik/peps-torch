import warnings
import torch
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.pess_kagome import rdm_kagome
from ctm.generic import corrf
from ctm.generic import transferops
import groups.su2 as su2
from math import sqrt
from numpy import exp
import itertools

_cast_to_real= rdm._cast_to_real

class S_HALF_KAGOME():

    def __init__(self, j1=1., JD=0, j1sq=0, j2=0, j2sq=0, jtrip=0.,\
        jperm=0+0j, h=0, phys_dim=2, global_args=cfg.global_args):
        r"""
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
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = phys_dim
        self.j1 = j1
        self.j1sq = j1sq
        self.JD = JD
        # j2, j2sq not implemented
        self.j2 = j2
        self.j2sq = j2sq
        self.jtrip = jtrip
        self.jperm = jperm
        self.h = h
        
        irrep = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)

        Id1= irrep.I()
        self.Id3_t= torch.eye(self.phys_dim**3, dtype=self.dtype, device=self.device)
        
        # nearest-neighbour terms with DMI
        SS= irrep.SS(xyz=(1., 1., 1.))
        SS_JD= self.j1*SS if abs(self.JD)==0 else irrep.SS(xyz=(j1, j1+1j*JD, j1-1j*JD))
        self.SSnnId= torch.einsum('ijkl,ab->ijaklb',SS_JD,Id1)
        SSnn_t= self.SSnnId + self.SSnnId.permute(1,2,0, 4,5,3) + self.SSnnId.permute(2,0,1, 5,3,4)
        
        # nearest-neighbour S.S^2 term. For S>1/2 it is equivalent to a constant
        SS2= torch.einsum('ijab,abkl->ijkl',SS,SS)
        SS2nnId= torch.einsum('ijkl,ab->ijaklb',SS2,Id1)
        SS2nn_t= SS2nnId + SS2nnId.permute(1,2,0, 4,5,3) + SS2nnId.permute(2,0,1, 5,3,4)

        # on-site magnetic field
        mag_field=torch.einsum('ij,kl,ab->ikajlb',irrep.SZ(),Id1,Id1)
        mag_field=mag_field + mag_field.permute(1,2,0, 4,5,3) + mag_field.permute(2,0,1, 5,3,4)

        # three-spin chiral interaction S.(SxS)
        if jtrip != 0:
            assert self.dtype==torch.complex128 or self.dtype==torch.complex64,"jtrip requires complex dtype"
        Svec= irrep.S()
        levicivit3= torch.zeros(3,3,3, dtype=self.dtype, device=self.device)
        levicivit3[0,1,2]=levicivit3[1,2,0]=levicivit3[2,0,1]=1.
        levicivit3[0,2,1]=levicivit3[2,1,0]=levicivit3[1,0,2]=-1.
        SxSS_t= torch.einsum('abc,bij,ckl,amn->ikmjln',levicivit3,Svec,Svec,Svec)

        self.P_triangle = torch.zeros([self.phys_dim]*6, dtype=self.dtype, device=self.device)
        self.P_triangle_inv = torch.zeros([self.phys_dim]*6, dtype=self.dtype, device=self.device)
        for i in range(self.phys_dim):
            for j in range(self.phys_dim):
                for k in range(self.phys_dim):
                    # anticlockwise (direct)
                    #
                    # 2---1 <- 0---2
                    #  \ /      \ /
                    #   0        1
                    self.P_triangle[i, j, k, j, k, i] = 1.
                    # clockwise (inverse)
                    self.P_triangle_inv[i, j, k, k, i, j] = 1.

        self.h_triangle= SSnn_t + self.j1sq*SS2nn_t + self.jtrip*SxSS_t \
            + self.jperm * self.P_triangle + self.jperm.conjugate() * self.P_triangle_inv \
            + self.h*mag_field

        szId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SZ(),Id1,Id1).contiguous()
        spId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SP(),Id1,Id1).contiguous()
        smId2= torch.einsum('ij,kl,ab->ikajlb',irrep.SM(),Id1,Id1).contiguous()
        self.obs_ops= {
            "sz_0": szId2, "sp_0": spId2, "sm_0": smId2,\
            "sz_1": szId2.permute(2,0,1, 5,3,4).contiguous(),\
            "sp_1": spId2.permute(2,0,1, 5,3,4).contiguous(),\
            "sm_1": smId2.permute(2,0,1, 5,3,4).contiguous(),\
            "sz_2": szId2.permute(1,2,0, 4,5,3).contiguous(),\
            "sp_2": spId2.permute(1,2,0, 4,5,3).contiguous(),\
            "sm_2": smId2.permute(1,2,0, 4,5,3).contiguous(),
        }

    # :param fail_on_check: raise Exception if the imaginary part of the expectation 
    #                           value is larger than ``imag_eps``
    #     :type fail_on_check: bool
    #     :param warn_on_check: raise Warning if the imaginary part of the expectation 
    #                           value is larger ``imag_eps``
    #     :type warn_on_check: bool
    #     :param imag_eps: tolerance for imaginary part in expectation value
    #     :type imag_eps: float

    # Energy terms
    def energy_triangle_dn(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME
        :type env: ENV
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from down triangle within 2x2 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome.rdm_kagome.rdm2x2_dn_triangle_with_operator`.
        """
        e_dn, norm_2x2_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_triangle, force_cpu=force_cpu)
        return _cast_to_real(e_dn, **kwargs)

    def energy_triangle_dn_1x1(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME
        :type env: ENV
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from down triangle within 1x1 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome.rdm_kagome.rdm1x1_kagome`.
        """
        rdm1x1_dn= rdm_kagome.rdm1x1_kagome((0, 0), state, env, force_cpu=force_cpu, **kwargs)
        e_dn= torch.einsum('ijkmno,mnoijk', rdm1x1_dn, self.h_triangle )
        return _cast_to_real(e_dn, **kwargs)

    def energy_triangle_up(self, state, env, force_cpu=False, **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME
        :type env: ENV
        :param force_cpu: perform computation on CPU
        :type force_cpu: bool
        :return: energy per site
        :rtype: float
        
        Evaluate energy contribution from up triangle within 2x2 subsystem embedded in environment, 
        see :meth:`ctm.pess_kagome.rdm_kagome.rdm2x2_up_triangle_open`.
        """
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open(\
            (0, 0), state, env, force_cpu=force_cpu, **kwargs)
        e_up= torch.einsum('ijkmno,mnoijk', rdm_up, self.h_triangle )
        return _cast_to_real(e_up, **kwargs)

    # These functions do not cast observable into Real, thus returning
    # complex numbers
    def energy_triangle_up_NoCheck(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open(\
            (0, 0), state, env, force_cpu=force_cpu)
        e_up= torch.einsum('ijkmno,mnoijk', rdm_up, self.h_triangle )
        return e_up

    def energy_triangle_dn_NoCheck(self, state, env, force_cpu=False):
        e_dn, norm_2x2_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
            (0, 0), state, env, self.h_triangle, force_cpu=force_cpu)
        return e_dn

    # def energy_nnn(self, state, env, force_cpu=False):
    #     if self.j2 == 0:
    #         return 0.
    #     else:
    #         vNNN = self.P_bonds_nnn(state, env, force_cpu=force_cpu)
    #         return(self.j2*(vNNN[0]+vNNN[1]+vNNN[2]+vNNN[3]+vNNN[4]+vNNN[5]))

    # Observables

    def P_dn(self, state, env, force_cpu=False):
        vP_dn,norm_2x2_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env,\
            operator=self.P_triangle, force_cpu=force_cpu)
        return vP_dn

    def P_up(self, state, env, force_cpu=False):
        rdm_up= rdm_kagome.rdm2x2_up_triangle_open((0, 0), state, env, force_cpu=force_cpu)
        vP_up= torch.einsum('ijkmno,mnoijk', rdm_up, self.P_triangle)
        return vP_up

    # def P_bonds_nnn(self, state, env, force_cpu=False):
    #     norm_wf = rdm_kagome.rdm2x2_dn_triangle_with_operator((0, 0), state, env, \
    #         self.id_downT, force_cpu=force_cpu)
    #     vNNN1_12, vNNN1_31 = rdm_kagome.rdm2x2_nnn_1((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
    #     vNNN2_32, vNNN2_21 = rdm_kagome.rdm2x2_nnn_2((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
    #     vNNN3_31, vNNN3_23 = rdm_kagome.rdm2x2_nnn_3((0, 0), state, env, operator=exchange_bond, force_cpu=force_cpu)
    #     return _cast_to_real(vNNN1_12 / norm_wf), _cast_to_real(vNNN2_21 / norm_wf), \
    #         _cast_to_real(vNNN1_31 / norm_wf), _cast_to_real(vNNN3_31 / norm_wf), \
    #         _cast_to_real(vNNN2_32 / norm_wf), _cast_to_real(vNNN3_23 / norm_wf)

    # def P_bonds_nn(self, state, env):
    #     id_matrix = torch.eye(27, dtype=torch.complex128, device=cfg.global_args.device)
    #     norm_wf = rdm.rdm1x1((0, 0), state, env, operator=id_matrix)
    #     # bond 2--3
    #     bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 bond_op[fmap(i,j,k),fmap(i,k,j)] = 1.
    #     vP_23 = rdm.rdm1x1((0,0), state, env, operator=bond_op) / norm_wf
    #     # bond 1--3
    #     bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 bond_op[fmap(i,j,k),fmap(k,j,i)] = 1.
    #     vP_13 = rdm.rdm1x1((0, 0), state, env, operator=bond_op) / norm_wf
    #     # bond 1--2
    #     bond_op = torch.zeros((27, 27), dtype=torch.complex128, device=cfg.global_args.device)
    #     for i in range(3):
    #         for j in range(3):
    #             for k in range(3):
    #                 bond_op[fmap(i,j,k),fmap(j,i,k)] = 1.
    #     vP_12 = rdm.rdm1x1((0, 0), state, env, operator=bond_op) / norm_wf
    #     return(torch.real(vP_23), torch.real(vP_13), torch.real(vP_12))

    def eval_obs(self, state, env, force_cpu=True, cast_real=False, disp_corre_len=False,\
        **kwargs):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME
        :type env: ENV
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
                e_t_dn= self.energy_triangle_dn(state, env, force_cpu=force_cpu, **kwargs)
                e_t_up= self.energy_triangle_up(state, env, force_cpu=force_cpu, **kwargs)
            else:
                e_t_dn= self.energy_triangle_dn_NoCheck(state, env, force_cpu=force_cpu)
                e_t_up= self.energy_triangle_up_NoCheck(state, env, force_cpu=force_cpu)
            obs["e_t_dn"]= e_t_dn
            obs["e_t_up"]= e_t_up

            # compute on-site observables i.e. magnetizations
            norm_wf_1x1 = rdm_kagome.trace1x1_dn_kagome((0, 0), state, env, self.Id3_t)
            for label in self.obs_ops.keys():
                op= self.obs_ops[label].view(self.phys_dim**3, self.phys_dim**3)
                obs_val= rdm_kagome.trace1x1_dn_kagome((0, 0), state, env, op) / norm_wf_1x1
                obs[f"{label}"]= obs_val

            # compute magnitude of magnetization, m^2, on-site
            for i in range(3):
                #obs[f"m_{i}"]= sqrt(_cast_to_real(obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]))
                obs[f"m2_{i}"]= obs[f"sz_{i}"]*obs[f"sz_{i}"]+ obs[f"sp_{i}"]*obs[f"sm_{i}"]
 
            # nn S.S pattern. In self.SSnnId, the identity is placed on s2 of three sites
            # in the unitcell i.e. \vec{S}_0 \cdot \vec{S}_1 \otimes Id_2
            SS_dn_01,norm_2x2_dn= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId, force_cpu=force_cpu, **kwargs)
            # move identity from site 2 to site 0
            SS_dn_12,_= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId.permute(2,1,0, 5,4,3).contiguous(),\
                force_cpu=force_cpu, **kwargs)
            # move identity from site 2 to site 1
            SS_dn_02,_= rdm_kagome.rdm2x2_dn_triangle_with_operator(\
                (0, 0), state, env, self.SSnnId.permute(0,2,1, 3,5,4).contiguous(),\
                force_cpu=force_cpu, **kwargs)
            rdm_up= rdm_kagome.rdm2x2_up_triangle_open(\
                (0, 0), state, env, force_cpu=force_cpu, **kwargs)
            SS_up_01= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId )
            SS_up_12= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId.permute(2,1,0, 5,4,3) )
            SS_up_02= torch.einsum('ijkmno,mnoijk', rdm_up, self.SSnnId.permute(0,2,1, 3,5,4) )

            obs.update({"SS_dn_01": SS_dn_01, "SS_dn_12": SS_dn_12, "SS_dn_02": SS_dn_02,\
                "SS_up_01": SS_up_01, "SS_up_12": SS_up_12, "SS_up_02": SS_up_02 })
            
            if disp_corre_len: 
                obs= eval_corr_lengths(state, env, obs=obs)
                
        # prepare list with labels and values
        return list(obs.values()), list(obs.keys())

        def eval_corr_lengths(state, env, coord=(0,0), obs=None):
            Ns=3
            direction=(1,0)
            Lx= transferops.get_Top_spec(Ns, coord, direction, state, env)
            direction=(0,1)
            Ly= transferops.get_Top_spec(Ns, coord, direction, state, env)
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

    def eval_corrf_SS(self,coord,direction,state,env,dist,site=0):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS_KAGOME
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :param site: selects one of the non-equivalent physical degrees of freedom
                     within the unit cell
        :type site: int
        :return: dictionary with full and spin-resolved spin-spin correlation functions
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        
        The on-site tensor of the underlying IPEPS_KAGOME contains all three DoFs 
        of the down triangle. Choosing ``site`` selects one of them to be used for
        evaluating correlation function::

               a
               |
            b--\                     
                \
                s0--s2--d
                 | / 
                 |/   <- down triangle
                s1
                 |
                 c
        """
        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            # rot_op= ...
            op_0= op
            # op_rot= torch.einsum('ki,kl,lj->ij',rot_op,op_0,rot_op)
            def _gen_op(r):
                #return op_rot if r%2==0 else op_0
                return op_0
            return _gen_op

        assert site in [0,1,2],"site has to be 0,1, or 2"
        op_sz= self.obs_ops[f"sz_{site}"].view([self.phys_dim**3]*2)
        op_sx= 0.5*(self.obs_ops[f"sp_{site}"] + self.obs_ops[f"sm_{site}"])\
            .view([self.phys_dim**3]*2)
        op_isy= -0.5*(self.obs_ops[f"sp_{site}"] - self.obs_ops[f"sm_{site}"])\
            .view([self.phys_dim**3]*2)

        Sz0szR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sz, conjugate_op(op_sz), dist)
        Sx0sxR= corrf.corrf_1sO1sO(coord,direction,state,env, op_sx, conjugate_op(op_sx), dist)
        nSy0SyR= corrf.corrf_1sO1sO(coord,direction,state,env, op_isy, conjugate_op(op_isy), dist)

        res= dict({"ss": Sz0szR+Sx0sxR-nSy0SyR, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR})
        return res  