import argparse
import copy
import json
import logging
import os
import time
import unittest
import pickle
from typing import Sequence, Union

import context
import config as cfg
import numpy as np
import torch

import yastn.yastn as yastn
from yastn.yastn import Tensor, tensordot, load_from_dict
from yastn.yastn.tensor import YastnError
from yastn.yastn.tn.fpeps import Bond, EnvCTM, Site, RectangularUnitcell
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.sym import sym_Z2
from yastn.yastn.tn.fpeps.envs.rdm import *
from ipeps.integration_yastn import PepsAD, load_PepsAD
from optim.ad_optim_lbfgs_mod import optimize_state


class tV_model:
    def __init__(self, config, pd):
        # config: YASTN.config
        # pd: parameter dict
        # V1: n.n. interaction; V2: 2nd n.n. interaction; V3: 3rd n.n. interaction
        # t1: n.n. hopping, t2: amplitude of the 2nd n.n. hopping
        # phi: phase of the 2nd n.n. hopping along the positive direction
        # mu: chemical potential

        self.config = config
        self.dtype = config.default_dtype
        self.device = config.default_device

        self.V1 = pd["V1"]
        self.V2 = pd["V2"]
        self.V3 = pd["V3"]
        self.mu = pd["mu"]

        self.t1 = pd["t1"]
        self.t2 = pd["t2"]
        self.phi = pd["phi"]

    def get_parameters(self):
        if self.mu.requires_grad:
            return [self.mu,]
        return []

    def energy_per_site(self, psi, env):
        r"""
        :param psi: Peps
        :param env: CTM environment
        :return: energy
        :rtype: float
        """

        # x\y
        #     _:__:__:__:_
        #  ..._|__|__|__|_...
        #  ..._|__|__|__|_...
        #  ..._|__|__|__|_...
        #  ..._|__|__|__|_...
        #  ..._|__|__|__|_...
        #      :  :  :  :

        energy_onsite, energy_horz, energy_vert, energy_diag, energy_anti_diag= 0,0,0,0,0

        # sf = yastn.operators.SpinfulFermions(sym=str(self.config.sym), fermionic=True)
        _tmp_config = {x: y for x, y in self.config._asdict().items() if x != "sym"}
        sf = yastn.operators.SpinfulFermions(sym=str(self.config.sym), **_tmp_config)
        n_A = sf.n(spin="u")  # parity-even operator, no swap gate needed
        n_B = sf.n(spin="d")
        c_A = sf.c(spin="u")
        cp_A = sf.cp(spin="u")
        c_B = sf.c(spin="d")
        cp_B = sf.cp(spin="d")
        I = sf.I()
        N = len(psi.sites())

        ncon_order1 = ((1, 2), (3, 4), (2, 1, 4, 3))
        ncon_order2 = ((1, 2, 5), (3, 4, 5), (2, 1, 4, 3))
        for site in psi.sites():
            # onsite
            onsite_rdm, onsite_norm = rdm1x1(site, psi, env)  # s s'

            op = (
                self.V1 * (n_A @ n_B)
                - self.mu * (n_A + n_B)
                - self.t1 * (cp_A @ c_B + cp_B @ c_A).remove_zero_blocks()
            )
            e_1x1_loc = (
                yastn.ncon([op, onsite_rdm], ((1, 2), (2, 1)))
            )
            energy_onsite += e_1x1_loc.to_number()

            # horizontal bond
            horz_rdm, horz_rdm_norm = rdm1x2(site, psi, env)  # s0 s0' s1 s1'
            # horz_norm = yastn.trace(horz_rdm, ((0, 2), (1, 3))).to_number()
            e_1x2_loc = self.V1 * yastn.ncon([n_B, n_A, horz_rdm], ncon_order1)
            e_1x2_loc += self.V2 * yastn.ncon([n_B, n_B, horz_rdm], ncon_order1)
            e_1x2_loc += self.V2 * yastn.ncon([n_A, n_A, horz_rdm], ncon_order1)

            site_r = psi.nn_site(site, "r")
            ordered = psi.geometry.f_ordered(site, site_r)
            ci_B, cjp_A = op_order(c_B, cp_A, ordered, fermionic=True)
            cip_B, cj_A = op_order(cp_B, c_A, ordered, fermionic=True)
            e_1x2_loc += -self.t1 * (
                -yastn.ncon([ci_B, cjp_A, horz_rdm], ncon_order2)
                + yastn.ncon([cip_B, cj_A, horz_rdm], ncon_order2)
            )

            ci_A, cjp_A = op_order(c_A, cp_A, ordered, fermionic=True)
            cip_B, cj_B = op_order(cp_B, c_B, ordered, fermionic=True)
            e_1x2_loc += (
                -self.t2
                * np.exp(1j * self.phi)
                * (
                    -yastn.ncon([ci_A, cjp_A, horz_rdm], ncon_order2)
                    + yastn.ncon([cip_B, cj_B, horz_rdm], ncon_order2)
                )
            )

            cip_A, cj_A = op_order(cp_A, c_A, ordered, fermionic=True)
            ci_B, cjp_B = op_order(c_B, cp_B, ordered, fermionic=True)
            e_1x2_loc += (
                -self.t2
                * np.exp(-1j * self.phi)
                * (
                    yastn.ncon([cip_A, cj_A, horz_rdm], ncon_order2)
                    + -yastn.ncon([ci_B, cjp_B, horz_rdm], ncon_order2)
                )
            )
            energy_horz += e_1x2_loc.to_number()

            # vertical bond
            vert_rdm, vert_rdm_norm = rdm2x1(site, psi, env)  # s0 s0' s1 s1'
            # vert_norm = yastn.trace(vert_rdm, axes=((0, 2), (1, 3))).to_number()
            e_2x1_loc = self.V1 * yastn.ncon([n_A, n_B, vert_rdm], ncon_order1)
            e_2x1_loc += self.V2 * yastn.ncon([n_B, n_B, vert_rdm], ncon_order1)
            e_2x1_loc += self.V2 * yastn.ncon([n_A, n_A, vert_rdm], ncon_order1)

            site_b = psi.nn_site(site, "b")
            ordered = psi.geometry.f_ordered(site, site_b)

            cip_A, cj_B = op_order(cp_A, c_B, ordered, fermionic=True)
            ci_A, cjp_B = op_order(c_A, cp_B, ordered, fermionic=True)
            e_2x1_loc += -self.t1 * (
                yastn.ncon([cip_A, cj_B, vert_rdm], ncon_order2)
                + -yastn.ncon([ci_A, cjp_B, vert_rdm], ncon_order2)
            )
            ci_A, cjp_A = op_order(c_A, cp_A, ordered, fermionic=True)
            cip_B, cj_B = op_order(cp_B, c_B, ordered, fermionic=True)
            e_2x1_loc += (
                -self.t2
                * np.exp(1j * self.phi)
                * (
                    -yastn.ncon([ci_A, cjp_A, vert_rdm], ncon_order2)
                    + yastn.ncon([cip_B, cj_B, vert_rdm], ncon_order2)
                )
            )
            cip_A, cj_A = op_order(cp_A, c_A, ordered, fermionic=True)
            ci_B, cjp_B = op_order(c_B, cp_B, ordered, fermionic=True)
            e_2x1_loc += (
                -self.t2
                * np.exp(-1j * self.phi)
                * (
                    yastn.ncon([cip_A, cj_A, vert_rdm], ncon_order2)
                    + -yastn.ncon([ci_B, cjp_B, vert_rdm], ncon_order2)
                )
            )
            energy_vert += e_2x1_loc.to_number()

            if self.V2 != 0 or self.V3 != 0 or self.t2 != 0:

                plaq_rdm, plaq_rdm_norm  = rdm2x2(site, psi, env)  # s0 s0' s1 s1' s2 s2' s3 s3'
                # plaq_norm = yastn.trace(
                #     plaq_rdm, axes=((0, 2, 4, 6), (1, 3, 5, 7))
                # ).to_number()

                # diagonal bond
                diag_rdm = yastn.trace(plaq_rdm, ((2, 4), (3, 5)))  # s0 s0' s3 s3'
                e_2x2_diag_loc = self.V2 * (
                    yastn.ncon([n_A, n_A, diag_rdm], ncon_order1)
                    + yastn.ncon([n_B, n_B, diag_rdm], ncon_order1)
                )
                e_2x2_diag_loc += self.V3 * (
                    yastn.ncon([n_A, n_B, diag_rdm], ncon_order1)
                    + yastn.ncon([n_B, n_A, diag_rdm], ncon_order1)
                )

                site_br = psi.nn_site(site, "br")
                ordered = psi.geometry.f_ordered(site, site_br)

                cip_A, cj_A = op_order(cp_A, c_A, ordered, fermionic=True)
                ci_B, cjp_B = op_order(c_B, cp_B, ordered, fermionic=True)
                e_2x2_diag_loc += (
                    -self.t2
                    * np.exp(1j * self.phi)
                    * (
                        yastn.ncon([cip_A, cj_A, diag_rdm], ncon_order2)
                        + -yastn.ncon([ci_B, cjp_B, diag_rdm], ncon_order2)
                    )
                )
                ci_A, cjp_A = op_order(c_A, cp_A, ordered, fermionic=True)
                cip_B, cj_B = op_order(cp_B, c_B, ordered, fermionic=True)
                e_2x2_diag_loc += (
                    -self.t2
                    * np.exp(-1j * self.phi)
                    * (
                        -yastn.ncon([ci_A, cjp_A, diag_rdm], ncon_order2)
                        + yastn.ncon([cip_B, cj_B, diag_rdm], ncon_order2)
                    )
                )
                energy_diag += e_2x2_diag_loc.to_number()

                # anti-diagonal bond
                anti_diag_rdm = yastn.trace(plaq_rdm, ((0, 6), (1, 7)))  # s1 s1' s2 s2'
                energy_anti_diag += self.V3 * yastn.ncon(
                    [n_A, n_B, anti_diag_rdm], ncon_order1
                ).to_number()

        energy_per_site = (
            energy_onsite + energy_horz + energy_vert + energy_diag + energy_anti_diag
        ) / N

        return energy_per_site.real

    def eval_obs(self, psi, env):
        _tmp_config = {x: y for x, y in self.config._asdict().items() if x != "sym"}
        sf = yastn.operators.SpinfulFermions(sym=str(self.config.sym), **_tmp_config)
        n_A = sf.n(spin="u")  # parity-even operator, no swap gate needed
        n_B = sf.n(spin="d")
        c_A = sf.c(spin="u")
        cp_A = sf.cp(spin="u")
        c_B = sf.c(spin="d")
        cp_B = sf.cp(spin="d")
        I = sf.I()
        obs = {}
        op_n_list = {}
        op_c_list = {}
        op_cp_list = {}
        for s0 in psi.sites():
            obs_nA = measure_rdm_1site(s0, psi, env, n_A)
            obs_nB = measure_rdm_1site(s0, psi, env, n_B)
            m = abs(obs_nA - obs_nB)
            obs[s0] = [obs_nA, obs_nB, m]
            print(f"nA: {obs_nA:.4f}, nB: {obs_nB:.4f}, m: {m:.4f}")

            # obs_cA = measure_rdm_1site(s0, psi, env, c_A).item()
            # obs_cpA = measure_rdm_1site(s0, psi, env, cp_A).item()
            # print(f"cA: {obs_cA:.4f}, cp_A: {obs_cpA:.4f}")
            # op_n_list[psi.site2index(s0)] = n_A - obs_nA*I
            # op_c_list[psi.site2index(s0)] = c_A
            # op_cp_list[psi.site2index(s0)] = cp_A

        # corr_n = measure_rdm_corr_1x2(s0, 30, psi, env, op_list[psi.site2index(s0)], op_list)
        # corr_cc = measure_rdm_corr_1x2(s0, 30, psi, env, op_c_list[psi.site2index(s0)], op_cp_list)
        # print(corr_cc)
        return obs


def test_state(config=None, noise=0):
    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
        }
    )

    if config is None:  # use default
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="Z2")

    # t l b r s a
    # |11>
    t0 = yastn.Tensor(config=config, n=0, s=(-1, 1, 1, -1, 1, 1))
    t0.set_block(ts=(0, 0, 0, 0, 0, 0), Ds=(1, 1, 1, 1, 2, 1), val=torch.tensor([0, 1]))
    # |01>
    t0.set_block(ts=(0, 0, 0, 1, 1, 0), Ds=(1, 1, 1, 1, 2, 1), val=torch.tensor([0, 1]))
    t0 = t0.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))
    # |10>
    t1 = yastn.Tensor(config=config, n=0, s=(-1, 1, 1, -1, 1, 1))
    t1.set_block(ts=(0, 1, 0, 0, 1, 0), Ds=(1, 1, 1, 1, 2, 1), val=torch.tensor([1, 0]))
    # |00>
    t1.set_block(ts=(0, 0, 0, 0, 0, 0), Ds=(1, 1, 1, 1, 2, 1), val=torch.tensor([1, 0]))
    t1 = t1.fuse_legs(axes=(0, 1, 2, 3, (4, 5)))

    # |0110> + |1100> per iPEPS unit-cell
    psi = PepsAD(geometry, tensors={(0, 0): t0, (0, 1): t1})

    return psi


def random_3x3_state(config=None, bond_dim=(1, 1)):
    # 3x3 unit-cell pattern in the square lattice:
    # A B C
    # B C A
    # C A B
    # with one tensor in {A B C} having one electron occupied

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 1,
            (1, 1): 2,
            (1, 2): 0,
            (2, 0): 2,
            (2, 1): 0,
            (2, 2): 1,
        }
    )
    if config is None:  # use default
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="Z2")
    vectors = {}

    D0, D1 = bond_dim
    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(2, 2)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_flag=None):
        if dummy_leg_flag is not None:
            if dummy_leg_flag == 'even':
                dummy_leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
            elif dummy_leg_flag == 'odd':
                dummy_leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
            elif dummy_leg_flag == 'even_odd':
                dummy_leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            legs = legs + [dummy_leg]
            A = yastn.rand(config=config, n=n, legs=legs)
            l = len(legs)
            axes = [i for i in range(l - 2)] + [(l - 2, l - 1)]
            A = A.fuse_legs(axes=axes)  # Fuse the physical leg with the dummy leg
        else:
            A = yastn.rand(config=config, n=n, legs=legs)
        return A

    psi = PepsAD(
        geometry,
        tensors={
            (0, 0): rand_tensor_norm(0, legs, dummy_leg_flag='odd'),
            (0, 1): rand_tensor_norm(0, legs, dummy_leg_flag='even'),
            (0, 2): rand_tensor_norm(0, legs, dummy_leg_flag='even'),
        },
    )
    return psi

def random_1x1_state(config=None, bond_dim=(1, 1)):
    # 1x1 unit-cell in the square lattice

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
        }
    )

    if config is None:  # use default
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="Z2")
    vectors = {}

    D0, D1 = bond_dim
    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(2, 2)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_flag=None):
        if dummy_leg_flag is not None:
            if dummy_leg_flag == 'even':
                dummy_leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
            elif dummy_leg_flag == 'odd':
                dummy_leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
            elif dummy_leg_flag == 'even_odd':
                dummy_leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            legs = legs + [dummy_leg]
            A = yastn.rand(config=config, n=n, legs=legs)
            l = len(legs)
            axes = [i for i in range(l - 2)] + [(l - 2, l - 1)]
            A = A.fuse_legs(axes=axes)  # Fuse the physical leg with the dummy leg
        else:
            A = yastn.rand(config=config, n=n, legs=legs)
        return A

    psi = PepsAD(
        geometry,
        tensors={
            (0, 0): rand_tensor_norm(0, legs, dummy_leg_flag='odd'),
        },
    )
    return psi


def random_hc_state(config=None, bond_dim=(1, 1)):
    # Random state using the honeycomb lattice geometry
    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
        }
    )

    if config is None:  # use default
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="Z2")
    vectors = {}

    # tensors = [tensorA, tensorB] for A, B sublattice
    #   0       2   1       t(0)  r(3)
    #   |        \ /         \   /
    #   A--3      B--3  =>     B--
    #  / \        |            |   -> (4)
    # 1   2       0            A--
    #                        /   \
    #                       l(1)  b(2)

    D0, D1 = bond_dim
    A_legs = [
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(1, 1)),
    ]
    B_legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(1, 1)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_flag=None):
        if dummy_leg_flag is not None:
            if dummy_leg_flag == 'even':
                dummy_leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
            elif dummy_leg_flag == 'odd':
                dummy_leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
            elif dummy_leg_flag == 'even_odd':
                dummy_leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
            legs = legs + [dummy_leg]
            A = yastn.rand(config=config, n=n, legs=legs)
            l = len(legs)
            axes = [i for i in range(l - 2)] + [(l - 2, l - 1)]
            A = A.fuse_legs(axes=axes)  # Fuse the physical leg with the dummy leg
        else:
            A = yastn.rand(config=config, n=n, legs=legs)
        return A

    psi = PepsExtended(
        geometry,
        tensors={
            (0, 0): [
                rand_tensor_norm(0, A_legs, dummy_leg_flag="odd"),
                rand_tensor_norm(0, B_legs, dummy_leg_flag="even"),
            ],
        },
    )
    return psi


log = logging.getLogger(__name__)
# parse command line args and build necessary configuration objects
parser = cfg.get_args_parser()
parser.add_argument(
    "--V1", type=float, default=1.0, help="Nearest-neighbor interaction"
)
parser.add_argument(
    "--V2", type=float, default=0.0, help="2nd. nearest-neighbor interaction"
)
parser.add_argument(
    "--V3", type=float, default=0.0, help="3rd. nearest-neighbor interaction"
)
parser.add_argument(
    "--t1", type=float, default=1.0, help="Nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--t2", type=float, default=0.0, help="2nd. nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--phi", type=float, default=0.0, help="phase of the 2nd. nearest-neighbor hopping"
)
parser.add_argument("--mu", type=float, default=0.0, help="chemical potential")


parser.add_argument(
    "--yast_backend",
    type=str,
    default="torch",
    help="YAST backend",
    choices=["torch", "torch_cpp"],
)
args = parser.parse_args()  # process command line arguments
num_cores = os.cpu_count() // 2
args, unknown_args = parser.parse_known_args(
    [
        "--bond_dim",
        "1",
        "--chi",
        "20",
        "--opt_max_iter",
        "100",
        "--omp_cores",
        "8",
        # "--opt_resume",
        # "output_checkpoint.p",
        # "tV_1x1_D_2_chi_20_V_1.5_checkpoint.p",
        # "--opt_resume_override_params",
        "--seed",
        "120",
        "--CTMARGS_ctm_max_iter",
        "100",
        "--CTMARGS_ctm_env_init_type",
        "eye",
        "--OPTARGS_fd_eps",
        "1e-8",
        "--OPTARGS_opt_log_grad",
        # "--CTMARGS_fwd_checkpoint_move",
        "--OPTARGS_line_search",
        "backtracking",
        # "strong_wolfe",
    ],
    namespace=args,
)


def main():
    global args
    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend
    # elif args.yast_backend=='torch_cpp':
    #     from yastn.yastn.backend import backend_torch_cpp as backend
    # settings_full= yastn.make_config(backend=backend, \
    #     default_device= cfg.global_args.device, default_dtype=cfg.global_args.dtype)

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_Z2,
        fermionic=True,
        default_device=cfg.global_args.device,
        default_dtype=cfg.global_args.dtype,
    )

    yastn_config.backend.set_num_threads(args.omp_cores)
    yastn_config.backend.random_seed(args.seed)

    pd = {}
    pd["V1"], pd["V2"], pd["V3"] = args.V1, args.V2, args.V3
    pd["t1"], pd["t2"] = args.t1, args.t2
    pd["phi"] = args.phi
    pd["mu"] = args.mu

    model = tV_model(yastn_config, pd)

    @torch.no_grad()
    def ctm_conv_check(env,history,corner_tol):
        converged,max_dsv,history= ctm_conv_corner_spec(env,history,corner_tol)
        log.log(logging.INFO,f"CTM iter {len(history)} |delta_C| {max_dsv}")
        return converged, history

    def get_converged_env(env, method='2site', max_sweeps=100, iterator_step=1, opts_svd=None, corner_tol=1e-8):
        t_ctm, t_check = 0.0, 0.0
        t_ctm_prev = time.perf_counter()
        converged,conv_history=False,[]

        for sweep in range(max_sweeps):
            env.update_(opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=True)
            t_ctm_after = time.perf_counter()
            t_ctm += t_ctm_after - t_ctm_prev
            t_ctm_prev = t_ctm_after
            
            converged, conv_history= ctm_conv_check(env, conv_history, corner_tol)
            if converged: 
                break

        return env, converged, conv_history, t_ctm, t_check


    def loss_fn(state, ctm_env_in, opt_context):
        state.sync_()
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            # print("Reinit")
            chi = cfg.main_args.chi // 2
            env_leg = yastn.Leg(yastn_config, s=1, t=(0, 1), D=(chi, chi))
            ctm_env_in = EnvCTM(state, init=ctm_args.ctm_env_init_type, leg=env_leg)

        # 1) compute environment by CTMRG
        ctm_env_out, converged, *ctm_log, t_ctm, t_check = get_converged_env(
            ctm_env_in,
            max_sweeps=ctm_args.ctm_max_iter,
            iterator_step=1,
            opts_svd=opts_svd,
            corner_tol=1e-8,
        )

        # 2) evaluate loss with converged environment
        loss = model.energy_per_site(state, ctm_env_out) # H= H_0 + mu * (nA + nB)

        return (loss, ctm_env_out, *ctm_log, t_ctm, t_check)

    @torch.no_grad()
    def post_proc(state, env, opt_context):
        pass

    @torch.no_grad()
    def obs_fn(state, ctm_env, opt_context):
        state.sync_()
        if opt_context["line_search"]:
            epoch = len(opt_context["loss_history"]["loss_ls"])
            loss = opt_context["loss_history"]["loss_ls"][-1]
            print("LS " + ", ".join([f"{epoch}", f"{loss}"]+[f"{x.item()}" for x in model.get_parameters()]))
            # print("LS " + ", ".join([f"{epoch}", f"{(loss+0.75*pd['V1'])/2}"]))
        else:
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
            print(", ".join([f"{epoch}", f"{loss}"]+[f"{x.item()}" for x in model.get_parameters()]))
            # print(", ".join([f"{epoch}", f"{(loss+0.75*pd['V1'])/2}"]))

        model.eval_obs(state, ctm_env)

    V1 = 1.4
    pd["V1"] = V1
    pd["mu"] = torch.tensor([0.,],dtype=torch.float64,device=yastn_config.default_device,requires_grad=False)
    pd["t1"] = 1.
    D = args.bond_dim
    D1, D2 = 2, 2
    args, unknown_args = parser.parse_known_args(
        [
            "--out_prefix",
            f"tV_1x1_D_{D1}+{D2}_chi{args.chi}_V_{pd['V1']:.2f}",
        ],
        namespace=args,
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    cfg.configure(args)
    # cfg.print_config()
    model = tV_model(yastn_config, pd)

    opts_svd = {
        "D_total": cfg.main_args.chi,
        "tol": cfg.ctm_args.projector_svd_reltol,
        "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
    }

    # state= load_PepsAD(yastn_config, "tV_1x1_D_2+3_chi20_V_1.40_0r1_state.json")
    # state= load_PepsAD(yastn_config,"tV_1x1_D_2+2_chi20_V_1.40_state.json")
    state = random_1x1_state(config=yastn_config, bond_dim=(D1, D2))
    # state = random_3x3_state(config=yastn_config, bond_dim=(D1, D2))
    # state = add_noise(state, noise=1.0)
    conv_env = None
    optimize_state(state, conv_env, loss_fn, obs_fn=obs_fn, post_proc=None)


if __name__ == "__main__":
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()

