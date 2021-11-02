import torch
import groups.su2 as su2
import config as cfg
from ctm.generic.env import ENV
from ctm.generic import rdm
from ctm.generic import corrf
from math import sqrt
import itertools
import numpy as np
from tn_interface import mm

class BBH_J1J2():
    def __init__(self, spin_s=3, j1=1.0, j2=0.0, k1=0.0, k2=0.0, ratio=1.0, base_h=None, global_args=cfg.global_args):
        self.dtype = global_args.torch_dtype
        self.device = global_args.device
        self.phys_dim = spin_s
        self.j1 = j1
        self.j2 = j2
        self.k1 = k1
        self.k2 = k2
        self.ratio = ratio
        self.base_h_type = base_h

        self.bonds_nn_p, self.bonds_nnn_p, self.h2, self.h = self.get_h()
        self.obs_ops = self.get_obs_ops()
        self.Q = self.get_Q()
        self.flip = torch.tensor([[1., -1., 1.], [-1., 1., -1.], [1., -1., 1.]], dtype=self.dtype, device=self.device)
    # build spin-S bilinear-biquadratic Hamiltonian up to NNN coupling
    # H = j1*\sum_{<i,j>}S_i.S_j + k1*\sum_{<i,j>}(S_i.S_j)^2 + j2*\sum_{<<i,j>>}S_i.S_j + k2*\sum_{<<i,j>>}(S_i.S_j)^2
    #  
    # y\x
    #    _:__:__:__:_
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    # ..._|__|__|__|_...
    #     :  :  :  : 
    # 
    # where h_ij = S_i.S_j, indices of h correspond to s_i,s_j;s_i',s_j'
    def get_h(self):
        pd = self.phys_dim
        irrep = su2.SU2(pd, dtype=self.dtype, device=self.device)
        # identity operator on two spin-S spins
        idp = torch.eye(pd**2, dtype=self.dtype, device=self.device)
        idp = idp.view(pd, pd, pd, pd).contiguous()
        SS = irrep.SS()
        ss_2x2 = torch.einsum('ijab,klcd->ijklabcd', SS, idp)
        bonds_nn_p = ss_2x2 + ss_2x2.permute(2,3,0,1,6,7,4,5) + ss_2x2.permute(0,2,1,3,4,6,5,7) + ss_2x2.permute(2,0,3,1,6,4,7,5)
        bonds_nnn_p = ss_2x2.permute(0,2,1,3,4,6,5,7) + ss_2x2.permute(2,0,3,1,4,5,7,6)
        SS = SS.view(pd**2, pd**2)
        h2_nn = self.j1*SS + self.k1*SS@SS
        h2_nnn = self.j2*SS + self.k2*SS@SS
        # Reshape back into rank-4 tensor for later use with reduced density matrices
        h2_nn = h2_nn.view(pd, pd, pd, pd).contiguous()
        h2_nnn = h2_nnn.view(pd, pd, pd, pd).contiguous()
        h2 = h2_nn + h2_nnn

        h2x2_h2_nn = torch.einsum('ijab,klcd->ijklabcd', h2_nn, idp)
        h2x2_h2_nnn = torch.einsum('ilad,jkbc->ijklabcd', h2_nnn, idp)
        h_10_01 = h2x2_h2_nn + h2x2_h2_nn.permute(0,2,1,3,4,6,5,7)
        h_cross = h2x2_h2_nnn + h2x2_h2_nnn.permute(1,0,3,2,5,4,7,6)
        h = h_10_01 + h_cross

        return bonds_nn_p, bonds_nnn_p, h2, h

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

    # evaluation of energy depends on the nature of underlying
    # ipeps state
    #
    # Ex.1 for 1-site c4v invariant iPEPS there is just a single 2site
    # operator which gives the energy-per-site
    #
    # Ex.2 for 1-site invariant iPEPS there are two two-site terms
    # which give the energy-per-site
    #    0       0
    # 1--A--3 1--A--3 
    #    2       2                          A
    #    0       0                          2
    # 1--A--3 1--A--3                       0
    #    2       2    , terms A--3 1--A and A have to be evaluated
    #
    # Ex.3 for 2x2 cluster iPEPS there are eight two-site terms
    #    0       0       0
    # 1--A--3 1--B--3 1--A--3
    #    2       2       2
    #    0       0       0
    # 1--C--3 1--D--3 1--C--3
    #    2       2       2             A--3 1--B      A B C D
    #    0       0                     B--3 1--A      2 2 2 2
    # 1--A--3 1--B--3                  C--3 1--D      0 0 0 0
    #    2       2             , terms D--3 1--C and  C D A B
    def energy_2x2(self, state, env):
        energy = 0.
        for coord, site in state.sites.items():
            rdm2x2 = rdm.rdm2x2(coord, state, env)
            energy += torch.einsum('ijklabcd,ijklabcd', rdm2x2, self.h)
        energy_per_site = energy/len(state.sites.items())
        return energy_per_site

    def energy_2x2_var(self, state, env, type="plaquette"):
        energy_base = 0.
        energy_target = 0.
        pd = self.phys_dim
        idp = torch.eye(pd, dtype=self.dtype, device=self.device)
        for coord, site in state.sites.items():
            rdm2x2 = rdm.rdm2x2(coord, state, env)
            energy_target += torch.einsum('ijklabcd,ijklabcd', rdm2x2, self.h)
        if self.base_h_type == "plaquette":
            for coord, site in state.sites.items():
                if coord[0]%2 == 0 and coord[1]%2 == 0:
                    rdm2x2 = rdm.rdm2x2(coord, state, env)
                    ss = torch.einsum('ijab,kc,ld->ijklabcd', self.obs_ops["SS"], idp, idp)
                    baseh = ss + ss.permute(0,2,1,3,4,6,5,7) + ss.permute(2,0,3,1,6,4,7,5)+ss.permute(2,3,0,1,6,7,4,5)
                    # baseh = ss + ss.permute(0,2,1,3,4,6,5,7) + 0.5 * (ss.permute(0,2,3,1,4,6,7,5)+ss.permute(2,0,1,3,6,4,5,7))
                    energy_base += torch.einsum('ijklabcd,ijklabcd', rdm2x2, baseh)
        elif self.base_h_type == "dimer":
            for coord, site in state.sites.items():
                if coord[0] % 2 == 0:
                    rdm2x1 = rdm.rdm2x1(coord, state, env)
                    energy_base += torch.einsum('ijab,ijab', rdm2x1, self.obs_ops["SS"])
        elif self.base_h_type == "staggered_dimer":
            for coord, site in state.sites.items():
                if (coord[0] + coord[1]) % 2 == 0:
                    rdm2x1 = rdm.rdm2x1(coord, state, env)
                    energy_base += torch.einsum('ijab,ijab', rdm2x1, self.obs_ops["SS"])
        elif self.base_h_type == "neel_afm":
            for coord, site in state.sites.items():
                sign = (-1)**(coord[0] + coord[1])
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                energy_base += sign*torch.einsum('ia,ia', rdm1x1, self.obs_ops["sz"])
        elif self.base_h_type == "stripe_afm":
            for coord, site in state.sites.items():
                sign = (-1)**coord[0]
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                energy_base += sign*torch.einsum('ia,ia', rdm1x1, self.obs_ops["sz"])
        else:
            raise ValueError("Invalid base state type: " + str(type) + " Supported options: " \
                             + "plaquette, dimer, staggered_dimer, neel_afm, stripe_afm")
        energy = (1 - self.ratio) * energy_base + self.ratio * energy_target
        energy_per_site = energy/len(state.sites.items())
        return energy_per_site

    def eval_obs(self, state, env):
        obs = dict({"avg_II_Q": 0., "avg_III_Q": 0.,
                    "avg_s_vector": torch.zeros(3, dtype=self.dtype, device=self.device),
                    "avg_s_vector_pi_0": torch.zeros(3, dtype=self.dtype, device=self.device),
                    "avg_s_vector_pi_pi": torch.zeros(3, dtype=self.dtype, device=self.device)})
        with torch.no_grad():
            # one-site observables
            for coord, site in state.sites.items():
                sign_pi_0 = (-1)**(coord[0])
                sign_pi_pi = (-1) ** (coord[0] + coord[1])
                rdm1x1 = rdm.rdm1x1(coord, state, env)
                for label in ["sz", "sp", "sm"]:
                    op = self.obs_ops[label]
                    obs[f"{label}{coord}"] = torch.einsum('ij,ji',rdm1x1,op)
                # obs[f"m{coord}"] = sqrt(abs(obs[f"sz{coord}"] ** 2 + obs[f"sp{coord}"] * obs[f"sm{coord}"]))
                # # question?
                # obs["avg_m"] += obs[f"m{coord}"]
                s_vector = np.array([obs[f"sz{coord}"], obs[f"sp{coord}"], obs[f"sm{coord}"]])
                obs["avg_s_vector"] += s_vector
                obs["avg_s_vector_pi_0"] += sign_pi_0 * s_vector
                obs["avg_s_vector_pi_pi"] += sign_pi_pi * s_vector

                obs[f"Q{coord}"] = torch.einsum("ab,ijba->ij", rdm1x1, self.Q)
                obs[f"avg_II_Q{coord}"] = -1/2 * torch.einsum('ij,ji',(obs[f"Q{coord}"] * self.flip),obs[f"Q{coord}"])
                # TODO det currently not supported for complex tensors
                obs[f"avg_III_Q{coord}"] = - torch.det(obs[f"Q{coord}"])
                obs["avg_II_Q"] += obs[f"avg_II_Q{coord}"]
                obs["avg_III_Q"] += obs[f"avg_III_Q{coord}"]

            s_vector = obs["avg_s_vector"] / len(state.sites.keys())
            obs["avg_m"] = sqrt(abs(s_vector[0] ** 2 + s_vector[1] * s_vector[2]))
            s_vector = obs["avg_s_vector_pi_0"] / len(state.sites.keys())
            obs["anti_fm_pi_0"] = sqrt(abs(s_vector[0] ** 2 + s_vector[1] * s_vector[2]))
            s_vector = obs["avg_s_vector_pi_pi"] / len(state.sites.keys())
            obs["anti_fm_pi_pi"] = sqrt(abs(s_vector[0] ** 2 + s_vector[1] * s_vector[2]))

            obs["avg_II_Q"] = obs["avg_II_Q"] / len(state.sites.keys())
            obs["avg_III_Q"] = obs["avg_III_Q"] / len(state.sites.keys())
            obs["dimer_op"] = self.eval_dimer_operator(state, env)

            # # two-site(nn) observables
            # s_vector = obs["avg_s_vector"] / len(state.sites.keys())
            # anti_s_vector = obs["avg_s_vector_pi_0"] / len(state.sites.keys())
            # s_vector_a = s_vector + anti_s_vector
            # s_vector_b = s_vector - anti_s_vector
            # sz_a = s_vector_a[0]
            # sz_b = s_vector_b[0]
            # sx_a = (s_vector_a[1] + s_vector_a[2]) / 2
            # sx_b = (s_vector_b[1] + s_vector_b[2]) / 2
            # isy_a = (s_vector_a[1] - s_vector_a[2]) / 2
            # isy_b = (s_vector_b[1] - s_vector_b[2]) / 2
            # obs["ss_aa"] = sz_a * sz_a + sx_a * sx_a - isy_a * isy_a
            # obs["ss_ab"] = sz_a * sz_b + sx_a * sx_b - isy_a * isy_b

            obs["pVBS"] = self.eval_plaquette_order(state, env)
        # prepare list with labels and values
        obs_labels = ["avg_m", "avg_II_Q", "avg_III_Q", "anti_fm_pi_0", "anti_fm_pi_pi", "dimer_op", "pVBS"]
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
                    obs[f"{label}{coord}"] = torch.einsum('ij,ji',rdm1x1,op)
                obs[f"m{coord}"] = sqrt(abs(obs[f"sz{coord}"] ** 2 + obs[f"sp{coord}"] * obs[f"sm{coord}"]))
                obs["avg_m"] += obs[f"m{coord}"]

                obs[f"Q{coord}"] = torch.einsum("ab,ijba->ij", rdm1x1, self.Q)
                obs[f"avg_II_Q{coord}"] = -1/2 * torch.einsum("ij,ji", (obs[f"Q{coord}"]*self.flip), obs[f"Q{coord}"])
                # TODO det currently not supported for complex tensors
                obs[f"avg_III_Q{coord}"] = -torch.det(obs[f"Q{coord}"])

                obs["avg_II_Q"] += obs[f"avg_II_Q{coord}"]
                obs["avg_III_Q"] += obs[f"avg_III_Q{coord}"]

            obs["avg_m"] = obs["avg_m"] / len(state.sites.keys())
            obs["avg_II_Q"] = obs["avg_II_Q"] / len(state.sites.keys())
            obs["avg_III_Q"] = obs["avg_III_Q"] / len(state.sites.keys())
            obs["dimer_op"] = self.eval_dimer_operator(state, env)
        return obs

    def eval_dimer_operator(self, state, env, direction=(1, 0)):
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
        with torch.no_grad():
            # one-site observables
            for coord, site in state.sites.items():
                Sz0szR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sz,
                                            conjugate_op(op_sz), 0)
                Sx0sxR = corrf.corrf_1sO1sO(coord, direction, state, env, op_sx,
                                            conjugate_op(op_sx), 0)
                nSy0SyR = corrf.corrf_1sO1sO(coord, direction, state, env, op_isy,
                                             conjugate_op(op_isy), 0)
                ss.append(Sz0szR + Sx0sxR - nSy0SyR)

        dimer_op = torch.abs(ss[0] - ss[1])
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
                op = mm(self.obs_ops[i], self.obs_ops[j]) + mm(self.obs_ops[j], self.obs_ops[i])
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

    def eval_plaquette_order(self, state, env):
        with torch.no_grad():
            # one-site observables
            rdm2x2_00 = rdm.rdm2x2((0, 0), state, env)
            pVBS = torch.einsum('ijklabcd,ijklabcd', rdm2x2_00, self.bonds_nn_p)
            rdm2x2_11 = rdm.rdm2x2((1, 1), state, env)
            pVBS -= torch.einsum('ijklabcd,ijklabcd', rdm2x2_11, self.bonds_nn_p)
        return torch.abs(pVBS)
