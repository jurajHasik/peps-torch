import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import corrf
from ctm.generic import rdm
from math import sqrt
import itertools
import numpy as np


class COUPLEDCHAINS():
    def __init__(self, phys_dim=3, j1_x=1.0, j1_y=1.0, k1_x=0.0, k1_y=0.0, 
            global_args=cfg.global_args):
        r"""
        :param phys_dim: dimension of physical spin, i.e., 3 for spin-1
        :param j1_x: strength of nearest-neighbour spin-spin interaction along x-axis
        :param j1_y: strength of nearest-neighbour spin-spin interaction along y-axis
        :param k1_x: strength of nearest-neighbour biquadratic interaction along x-axis
        :param k1_y: strength of nearest-neighbour biquadratic interaction along x-axis
        :param global_args: global configuration
        :type phys_dim: int
        :type j1_x: float
        :type j1_y: float
        :type k1_x: float
        :type k1_y: float
        :type global_args: GLOBALARGS

        Build Spin-S bilinear-biquadratic :math:`J-K` Hamiltonian on square lattice

        .. math:: H =  &\sum_{i,j} \bigg[J_x\boldsymbol{S}_{i,j} \cdot \boldsymbol{S}_{i+1,j} 
            + K_x (\boldsymbol{S}_{i,j} \cdot \boldsymbol{S}_{i+1,j})^2\bigg] \notag\\ & 
            + \sum_{i, j} \bigg[J_y\boldsymbol{S}_{i, j} \cdot \boldsymbol{S}_{i, j+1} 
            + K_y (\boldsymbol{S}_{i, j} \cdot \boldsymbol{S}_{i,j+1})^2\bigg].

        """
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = phys_dim
        self.j1_x = j1_x
        self.j1_y = j1_y
        self.k1_x = k1_x
        self.k1_y = k1_y
        self.obs_ops = self.get_obs_ops()
        self.h2_x, self.h2_y, self.hp_h, self.hp_v, self.hp = self.get_h()
        self.Q = self.get_Q()
        self.flip = torch.tensor([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]], \
            dtype=self.dtype, device=self.device)

    def get_obs_ops(self):
        obs_ops = dict()
        irrep = su2.SU2(self.phys_dim, dtype=self.dtype, device=self.device)
        obs_ops["sz"] = irrep.SZ()
        obs_ops["sp"] = irrep.SP()
        obs_ops["sm"] = irrep.SM()
        obs_ops["I"] = irrep.I()
        obs_ops["sx"] = 0.5 * (obs_ops["sp"] + obs_ops["sm"])
        obs_ops["isy"] = 0.5 * (obs_ops["sp"] - obs_ops["sm"])
        obs_ops["SS"] = irrep.SS()
        obs_ops["SS_square"] = torch.einsum('ijab,abkl->ijkl', obs_ops["SS"], obs_ops["SS"])
        return obs_ops

    def get_h(self):
        pd = self.phys_dim
        # identity operator on two spin-S spins
        idp = torch.eye(pd ** 2, dtype=self.dtype, device=self.device)
        idp = idp.view(pd, pd, pd, pd).contiguous()
        SS = self.obs_ops["SS"]
        SS = SS.view(pd ** 2, pd ** 2)
        SS_square = SS @ SS
        h2_x = self.j1_x * SS + self.k1_x * SS_square
        h2_y = self.j1_y * SS + self.k1_y * SS_square

        # Reshape back into rank-4 tensor for later use with reduced density matrices
        h2_x = h2_x.view(pd, pd, pd, pd).contiguous()
        h2_y = h2_y.view(pd, pd, pd, pd).contiguous()

        h2x2_h2_x = torch.einsum('ijab,klcd->ijklabcd', h2_x, idp)
        h2x2_h2_y = torch.einsum('ijab,klcd->ijklabcd', h2_y, idp)
        # Create operators acting on four spins-S on plaquette. These are useful
        # for computing energy by different rearragnement of Hamiltonian terms
        #
        # NN-terms along horizontal bonds of plaquette
        hp_h = h2x2_h2_x + h2x2_h2_x.permute(2, 3, 0, 1, 6, 7, 4, 5)
        # NN-terms along vertical bonds of plaquette
        hp_v = h2x2_h2_y.permute(0, 2, 1, 3, 4, 6, 5, 7) + h2x2_h2_y.permute(2, 0, 3, 1, 6, 4, 7, 5)
        # All NN-terms within plaquette
        hp = hp_h + hp_v

        return h2_x, h2_y, hp_h, hp_v, hp

    def energy_2x1_1x2(self, state, env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 2x1 unit cell containing two tensors A, B. We can
        tile the square lattice in two ways::

            BIPARTITE           STRIPE   

            A B A B             A B A B
            B A B A             A B A B
            A B A B             A B A B
            B A B A             A B A B

        Taking reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster with indexing 
        of sites as follows :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of individual tensors following
        set of terms has to be evaluated in order to compute energy-per-site::
                
               0           
            1--A--3
               2
            
            Ex.1 unit cell A B, with BIPARTITE tiling

                A3--1B, B3--1A, A, B, A3  , B3  ,   1A,   1B
                                2  0   \     \      /     / 
                                0  2    \     \    /     /  
                                B  A    1A    1B  A3    B3  
            
            Ex.2 unit cell A B, with STRIPE tiling

                A3--1A, B3--1B, A, B, A3  , B3  ,   1A,   1B
                                2  0   \     \      /     / 
                                0  2    \     \    /     /  
                                A  B    1B    1A  B3    A3  
        """
        energy = 0.
        for coord, site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord, state, env)
            rdm1x2 = rdm.rdm1x2(coord, state, env)
            energy += torch.einsum('ijab,ijab', rdm2x1, self.h2_x)
            energy += torch.einsum('ijab,ijab', rdm1x2, self.h2_y)

        energy_per_site = energy / len(state.sites.items())
        return energy_per_site

    def energy_2x2_4site(self, state, env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return: energy per site
        :rtype: float

        We assume iPEPS with 2x2 unit cell containing four tensors A, B, C, and D with
        simple PBC tiling::

            A B A B
            C D C D
            A B A B
            C D C D
    
        Taking the reduced density matrix :math:`\rho_{2x2}` of 2x2 cluster given by 
        :py:func:`ctm.generic.rdm.rdm2x2` with indexing of sites as follows 
        :math:`\rho_{2x2}(s_0,s_1,s_2,s_3;s'_0,s'_1,s'_2,s'_3)`::
        
            s0--s1
            |   |
            s2--s3

        and without assuming any symmetry on the indices of the individual tensors a set
        of four :math:`\rho_{2x2}`'s are needed over which :math:`h2` operators 
        for the nearest and next-neaerest neighbour pairs are evaluated::  

            A3--1B   B3--1A   C3--1D   D3--1C
            2    2   2    2   2    2   2    2
            0    0   0    0   0    0   0    0
            C3--1D & D3--1C & A3--1B & B3--1A
        """
        rdm2x2_00 = rdm.rdm2x2((0, 0), state, env)
        rdm2x2_10 = rdm.rdm2x2((1, 0), state, env)
        rdm2x2_01 = rdm.rdm2x2((0, 1), state, env)
        rdm2x2_11 = rdm.rdm2x2((1, 1), state, env)
        energy = torch.einsum('ijklabcd,ijklabcd', rdm2x2_00, self.hp_h)
        energy += torch.einsum('ijklabcd,ijklabcd', rdm2x2_10, self.hp_v)
        energy += torch.einsum('ijklabcd,ijklabcd', rdm2x2_01, self.hp_v)
        energy += torch.einsum('ijklabcd,ijklabcd', rdm2x2_11, self.hp_h)

        energy_per_site = energy / 8.0  # definition of hp_h and hp_v
        return energy_per_site

    def eval_obs(self, state, env):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :return:  expectation values of observables, labels of observables
        :rtype: list[float], list[str]

        Computes the following observables in order

            1. average on-site magnetization over the unit cell
            2. average 2nd and 3rd moment of quadrupole matrix (see :meth:`get_Q`)
            3. nearest-neighbour spin-spin and biquadratic correlations
            4. dimer order parameter (see :meth:`eval_dimer_operator`)

        where the on-site magnetization is defined as

        .. math::
            m = \sqrt{ \langle S^z \rangle^2+\langle S^x \rangle^2+\langle S^y \rangle^2 }.
        """
        obs = dict({"avg_m": 0., "avg_II_Q": 0., "avg_III_Q": 0.})
        with torch.no_grad():
            # one-site observables
            for coord, site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                for label in ["sz", "sp", "sm"]:
                    op = self.obs_ops[label]
                    obs[f"{label}{coord}"] = torch.trace(rdm1x1 @ op)
                obs[f"m{coord}"] = sqrt(abs(obs[f"sz{coord}"] ** 2 + obs[f"sp{coord}"] * obs[f"sm{coord}"]))
                # question?
                obs["avg_m"] += obs[f"m{coord}"]
                obs[f"Q{coord}"] = torch.einsum("ab,ijba->ij", rdm1x1, self.Q)
                obs[f"avg_II_Q{coord}"] = - 1 / 2 * torch.trace((obs[f"Q{coord}"] * self.flip) @ obs[f"Q{coord}"])
                obs[f"avg_III_Q{coord}"] = - torch.det(obs[f"Q{coord}"])
                obs["avg_II_Q"] += obs[f"avg_II_Q{coord}"]
                obs["avg_III_Q"] += obs[f"avg_III_Q{coord}"]

            obs["avg_m"] = obs["avg_m"] / len(state.sites.keys())
            obs["avg_II_Q"] = obs["avg_II_Q"] / len(state.sites.keys())
            obs["avg_III_Q"] = obs["avg_III_Q"] / len(state.sites.keys())

            # two-site observables
            _ss_and_ss2_labels=[]
            for coord, site in state.sites.items():
                rdm2x1 = rdm.rdm2x1(coord, state, env)
                rdm1x2 = rdm.rdm1x2(coord, state, env)
                obs[f"SS_2x1{coord}"]= torch.einsum('ijab,abij',rdm2x1,self.obs_ops["SS"])
                obs[f"SS_1x2{coord}"]= torch.einsum('ijab,abij',rdm1x2,self.obs_ops["SS"])
                obs[f"SS2_2x1{coord}"]= torch.einsum('ijab,abij',rdm2x1,self.obs_ops["SS_square"])
                obs[f"SS2_1x2{coord}"]= torch.einsum('ijab,abij',rdm1x2,self.obs_ops["SS_square"])
                _ss_and_ss2_labels+=[f"SS_2x1{coord}",f"SS2_2x1{coord}",f"SS_1x2{coord}",f"SS2_1x2{coord}"]
            obs["dimer_op"] = self.eval_dimer_operator(state, env)

        # prepare list with labels and values
        obs_labels = ["avg_m", "avg_II_Q", "avg_III_Q", "anti_fm", "dimer_op"] + _ss_and_ss2_labels
        obs_values = [obs[label] for label in obs_labels]
        return obs_values, obs_labels

    def eval_onsite_obs(self, state, env):

        obs = dict({"avg_m": 0., "avg_II_Q": 0., "avg_III_Q": 0.})
        with torch.no_grad():
            # one-site observables
            for coord, site in state.sites.items():
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                for label in ["sz", "sp", "sm"]:
                    op = self.obs_ops[label]
                    obs[f"{label}{coord}"] = torch.trace(rdm1x1 @ op)
                obs[f"m{coord}"] = sqrt(abs(obs[f"sz{coord}"] ** 2 + obs[f"sp{coord}"] * obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]

                obs[f"Q{coord}"] = torch.einsum("ab,ijba->ij", rdm1x1, self.Q)
                obs[f"avg_II_Q{coord}"] = - 1 / 2 * torch.einsum("ij,ij,ji", obs[f"Q{coord}"], self.flip, obs[f"Q{coord}"])
                obs[f"avg_III_Q{coord}"] = - torch.det(obs[f"Q{coord}"])

                obs["avg_II_Q"] += obs[f"avg_II_Q{coord}"]
                obs["avg_III_Q"] += obs[f"avg_III_Q{coord}"]

            obs["avg_m"] = obs["avg_m"] / len(state.sites.keys())
            obs["avg_II_Q"] = obs["avg_II_Q"] / len(state.sites.keys())
            obs["avg_III_Q"] = obs["avg_III_Q"] / len(state.sites.keys())
            obs["dimer_op"] = self.eval_dimer_operator(state, env)
        return obs

    def eval_dimer_operator(self, state, env, direction=(1,0)):
        r"""
        :param state: wavefunction
        :param env: CTM environment
        :param direction: unit vector specifying direction as either 
            horizontal ``(1,0)`` or vertical ``(0,1)``
        :type state: IPEPS
        :type env: ENV
        :type direction: tuple(int)
        :return:  expectation value of dimer order parameter in ``direction``
        :rtype: float

        The dimer order parameter in horizontal direction is defined as

        .. math::
            D = |\vec{S}_{i,j}\cdot\vec{S}_{i+1,j} - \vec{S}_{i+1,j}\cdot\vec{S}_{i+2,j}|.

        For vertical direction the definition is analogous.
        """
        assert direction==(1,0) or direction==(0,1),"Invalid direction"
        ss = []
        with torch.no_grad():
            for coord, site in state.sites.items():
                if direction==(1,0):
                    _tmp_rdm= rdm.rdm2x1(coord, state, env)
                elif direction==(0,1):
                    _tmp_rdm= rdm.rdm1x2(coord, state, env)

                ss.append( torch.einsum('ijab,abij',_tmp_rdm,self.obs_ops["SS"]) )

        dimer_op = torch.abs(ss[0] - ss[1])
        return dimer_op

    def get_Q(self):
        r"""
        :return: quadrupole matrix
        :rtype: torch.Tensor

        Quadrupole matrix is 3x3 matrix of spin operators defined as

        .. math:: Q^{\alpha\beta} = S^{\alpha} S^{\beta} + S^{\beta} S^{\alpha}
            − \frac{2}{3}S(S + 1)\delta^{\alpha\beta}
        
        To work with real-valued entries only, we introduce extra imaginary units
        as follows::

            Q  iQ  Q
            iQ -Q iQ
            Q  iQ  Q
        """
        Q = []
        spin_s = (self.phys_dim - 1) / 2
        for i in ["sx", "isy", "sz"]:
            row = []
            for j in ["sx", "isy", "sz"]:
                op = self.obs_ops[i] @ self.obs_ops[j] + self.obs_ops[j] @ self.obs_ops[i]
                if i == j:
                    if i == "isy":
                        op = op + 2 / 3 * spin_s * (spin_s + 1) * self.obs_ops["I"]
                    else:
                        op = op - 2 / 3 * spin_s * (spin_s + 1) * self.obs_ops["I"]
                row.append(op)
            row = torch.stack(row)
            Q.append(row)
        Q = torch.stack(Q)
        return Q

    def eval_corrf(self, coord, direction, state, env, dist):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with full and spin-resolved spin-spin correlation functions,
            biquadratic correlation function, and quadrupole-quadrupole correlations
        :rtype: dict(str: torch.Tensor)
        
        Evaluate spin-spin correlation functions :math:`\langle\mathbf{S}(r).\mathbf{S}(0)\rangle` 
        up to r = ``dist`` in given direction. Similarly for correlations 
        :math:`\langle\mathbf{S}^2(r).\mathbf{S}^2(0)\rangle` 
        and :math:`\langle\mathbf{Q}(r).\mathbf{Q}(0)\rangle`.

        The quadrupole vector is defined as 

        .. math:: \mathbf{Q}=\bigg[ (S^x)^2 − (S^y)^2, 
            \frac{1}{\sqrt{3}}[2(S^z)^2 − S(S + 1))], S^xS^y + S^yS^x, \\
            S^yS^z + S^zS^y, S^zS^x + S^xS^z \bigg].

        See :meth:`ctm.generic.corrf.corrf_1sO1sO`.
        """

        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            op_0 = op

            def _gen_op(r):
                return op_0

            return _gen_op

        op_sz = self.obs_ops["sz"]
        op_sx = self.obs_ops["sx"]
        op_isy = self.obs_ops["isy"]

        # spin-spin correlation functions
        Sz0szR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sz,
                                    conjugate_op(op_sz), dist)
        Sx0sxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sx,
                                    conjugate_op(op_sx), dist)
        nSy0SyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isy,
                                     conjugate_op(op_isy), dist)
        ss = Sz0szR + Sx0sxR - nSy0SyR

        op_szsz = op_sz @ op_sz
        op_sxsx = op_sx @ op_sx
        op_nsysy = op_isy @ op_isy
        op_szsx = op_sz @ op_sx
        op_sxsz = op_sx @ op_sz
        op_iszsy = op_sz @ op_isy
        op_isysz = op_isy @ op_sz
        op_isxsy = op_sx @ op_isy
        op_isysx = op_isy @ op_sx

        # S^{\alpha}_0 S^{\beta}_0 S^{\gamma}_r S^{\delta}_r 
        Szz0SzzR = corrf.corrf_1sO1sO(coord, direction, state, env, op_szsz, conjugate_op(op_szsz), dist)
        Sxx0SxxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sxsx, conjugate_op(op_sxsx), dist)
        Syy0SyyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_nsysy, conjugate_op(op_nsysy), dist)
        Szx0SzxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_szsx, conjugate_op(op_szsx), dist)
        Sxz0SxzR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sxsz, conjugate_op(op_sxsz), dist)
        nSzy0SzyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_iszsy, conjugate_op(op_iszsy), dist)
        nSyz0SyzR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isysz, conjugate_op(op_isysz), dist)
        nSxy0SxyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isxsy, conjugate_op(op_isxsy), dist)
        nSyx0SyxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isysx, conjugate_op(op_isysx), dist)

        ss_square = Szz0SzzR + Sxx0SxxR + Syy0SyyR + Szx0SzxR + Sxz0SxzR - nSzy0SzyR - nSyz0SyzR \
                    - nSxy0SxyR - nSyx0SyxR
        spin_s = (self.phys_dim - 1) / 2
        qq = 2 * ss_square + ss - 2 / 3 * (spin_s ** 2) * ((spin_s + 1) ** 2)
        res = dict({"ss": ss, "szsz": Sz0szR, "sxsx": Sx0sxR, "sysy": -nSy0SyR, \
                    "ss_square": ss_square, "qq": qq})
        return res

    def eval_corrf_DD_H(self, coord, direction, state, env, dist):
        r"""
        :param coord: reference site
        :type coord: tuple(int,int)
        :param direction: 
        :type direction: tuple(int,int)
        :param state: wavefunction
        :param env: CTM environment
        :type state: IPEPS
        :type env: ENV
        :param dist: maximal distance of correlator
        :type dist: int
        :return: dictionary with horizontal dimer-dimer correlation function
        :rtype: dict(str: torch.Tensor)
        
        Evaluate horizontal dimer-dimer correlation functions 

        .. math::
            \langle(\mathbf{S}(r+3).\mathbf{S}(r+2))(\mathbf{S}(1).\mathbf{S}(0))\rangle 

        up to r = ``dist`` .
        """
        
        # build dimer operator
        op_sz = self.obs_ops["sz"]
        op_sx = self.obs_ops["sx"]
        op_isy = self.obs_ops["isy"]
        op_SS= torch.einsum('ij,ab->iajb', op_sz, op_sz) + \
            torch.einsum('ij,ab->iajb', op_sx, op_sx) + \
            (-1)*torch.einsum('ij,ab->iajb', op_isy, op_isy)

        def _gen_op(r):
            return op_SS

        D0Dr= corrf.corrf_2sOH2sOH_E1((0,0), (1,0), state, env, \
            op_SS, _gen_op, dist, verbosity=0)

        res = dict({"DD_H": D0Dr})
        return res