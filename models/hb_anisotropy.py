import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import corrf
from ctm.generic import rdm
from math import sqrt
import itertools
import numpy as np


class HB():
    def __init__(self, phys_dim=3, j1_x=1.0, j1_y=1.0, k1_x=0.0, k1_y=0.0, global_args=cfg.global_args):
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
        self.flip = torch.tensor([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]], dtype=self.dtype, device=self.device)

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
        energy = 0.
        print("-----------")
        for coord, site in state.sites.items():
            rdm2x1 = rdm.rdm2x1(coord, state, env)
            rdm1x2 = rdm.rdm1x2(coord, state, env)
            energy += torch.einsum('ijab,ijab', rdm2x1, self.h2_x)
            energy += torch.einsum('ijab,ijab', rdm1x2, self.h2_y)

        energy_per_site = energy / len(state.sites.items())
        return energy_per_site

    def energy_2x2_4site(self, state, env):

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

        obs = dict({"avg_m": 0., "avg_II_Q": 0., "avg_III_Q": 0.,
                    "avg_anti_s_vector": np.zeros(3),
                    "avg_s_vector": np.zeros(3)})
        with torch.no_grad():
            # one-site observables
            sign = 1
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

            s_vector = obs["avg_s_vector"] / len(state.sites.keys())
            obs["avg_m"] = sqrt(abs(s_vector[0] ** 2 + s_vector[1] * s_vector[2]))
            s_vector = obs["avg_anti_s_vector"] / len(state.sites.keys())
            obs["anti_fm"] = sqrt(abs(s_vector[0] ** 2 + s_vector[1] * s_vector[2]))

            obs["avg_II_Q"] = obs["avg_II_Q"] / len(state.sites.keys())
            obs["avg_III_Q"] = obs["avg_III_Q"] / len(state.sites.keys())
            obs["dimer_op"] = self.eval_dimer_operator(state, env)

        # prepare list with labels and values
        obs_labels = ["avg_m", "avg_II_Q", "avg_III_Q", "anti_fm", "dimer_op"]
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
        def conjugate_op(op):
            # rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op = torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0 = op
            op_rot = torch.einsum('ki,kl,lj->ij', rot_op, op_0, rot_op)

            def _gen_op(r):
                # return op_rot if r%2==0 else op_0
                return op_0

            return _gen_op

        op_sz = self.obs_ops["sz"]
        op_sx = self.obs_ops["sx"]
        op_isy = self.obs_ops["isy"]
        ss = []

        print("op_sz = ", op_sz)
        print("conj_sz = ", conjugate_op(op_sz)(0))

        with torch.no_grad():
            # one-site observables
            for coord, site in state.sites.items():
                Sz0szR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sz,
                                            conjugate_op(op_sz), 0)
                Sx0sxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sx,
                                            conjugate_op(op_sx), 0)
                nSy0SyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isy,
                                             conjugate_op(op_isy), 0)

                print([Sx0sxR, -nSy0SyR, Sz0szR])
                ss.append(Sz0szR + Sx0sxR - nSy0SyR)

        dimer_op = torch.abs(ss[0] - ss[1])
        print(ss)
        return dimer_op.numpy()

    def get_Q(self):
        """
        Q Matrix is following
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
        # function allowing for additional site-dependent conjugation of op
        def conjugate_op(op):
            # rot_op= su2.get_rot_op(self.phys_dim, dtype=self.dtype, device=self.device)
            rot_op = torch.eye(self.phys_dim, dtype=self.dtype, device=self.device)
            op_0 = op
            op_rot = torch.einsum('ki,kl,lj->ij', rot_op, op_0, rot_op)

            def _gen_op(r):
                # return op_rot if r%2==0 else op_0
                return op_0

            return _gen_op

        op_sz = self.obs_ops["sz"]
        op_sx = self.obs_ops["sx"]
        op_isy = self.obs_ops["isy"]
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
