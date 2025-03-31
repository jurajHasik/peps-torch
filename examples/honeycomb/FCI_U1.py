import json
import logging
import os
import time
import ast, argparse


os.environ["OMP_NUM_THREADS"] = "8"
import context
import config as cfg
import numpy as np
import torch

import yastn.yastn as yastn
from yastn.yastn.tn.fpeps import EnvCTM, RectangularUnitcell, Bond
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps.envs.rdm import *
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint, env_raw_data, refill_env
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
        self.t3 = pd["t3"]
        self.phi = pd["phi"]
        self.m = pd["m"]

    def get_parameters(self):
        return []

    # from memory_profiler import profile
    # @profile
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

        energy_onsite, energy_horz, energy_vert, energy_diag, energy_anti_diag = (
            0,
            0,
            0,
            0,
            0,
        )

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


        # measure_function-based computation
        for site in psi.sites():
            op = (
                self.V1 * (n_A @ n_B)
                - self.mu * (n_A + n_B)
                - self.t1 * (cp_A @ c_B + cp_B @ c_A).remove_zero_blocks()
                + self.m * (n_A - n_B)
            )
            energy_onsite += env.measure_1site(op, site=site)

            # horizontal bond
            h_bond = Bond(site, psi.nn_site(site, "r"))
            e_1x2_loc = self.V1 * env.measure_nn(n_B, n_A, bond=h_bond)+ self.V2 * env.measure_nn(n_B, n_B, bond=h_bond)+ self.V2 * env.measure_nn(n_A, n_A, bond=h_bond)
            site_r = psi.nn_site(site, "r")
            res = self.t1 * env.measure_nn(c_B, cp_A, bond=h_bond)
            e_1x2_loc += res + res.conj()
            res = self.t2*np.exp(1j*self.phi)*env.measure_nn(c_A, cp_A, bond=h_bond)
            e_1x2_loc += (res + res.conj()).real
            res = -self.t2*np.exp(1j*self.phi)*env.measure_nn(cp_B, c_B, bond=h_bond)
            e_1x2_loc += (res + res.conj()).real
            energy_horz += e_1x2_loc


            # vertical bond
            v_bond = Bond(site, psi.nn_site(site, "b"))
            e_2x1_loc = self.V1*env.measure_nn(n_A, n_B, bond=v_bond) + self.V2*env.measure_nn(n_B, n_B, bond=v_bond) + self.V2*env.measure_nn(n_A, n_A, bond=v_bond)
            site_b = psi.nn_site(site, "b")
            res = -self.t1*env.measure_nn(cp_A, c_B, bond=v_bond)
            e_2x1_loc += (res + res.conj()).real
            res = self.t2*np.exp(1j*self.phi)*env.measure_nn(c_A, cp_A, bond=v_bond)
            e_2x1_loc += (res + res.conj()).real
            res = -self.t2*np.exp(1j*self.phi)*env.measure_nn(cp_B, c_B, bond=v_bond)
            e_2x1_loc += (res + res.conj()).real
            energy_vert += e_2x1_loc

            if self.V2 != 0 or self.V3 != 0 or self.t2 != 0 or self.t3 != 0:
                site_br = psi.nn_site(site, "br")
                e_2x2_diag_loc = self.V2*env.measure_2x2(n_A, n_A, sites=[site, site_br]) + self.V2*env.measure_2x2(n_B, n_B, sites=[site, site_br])
                e_2x2_diag_loc += self.V3*env.measure_2x2(n_A, n_B, sites=[site, site_br]) + self.V3*env.measure_2x2(n_B, n_A, sites=[site, site_br])

                res = -self.t2*np.exp(1j*self.phi) * env.measure_2x2(cp_A, c_A, sites=[site, site_br])
                e_2x2_diag_loc += (res + res.conj()).real
                res = self.t2*np.exp(1j*self.phi) * env.measure_2x2(c_B, cp_B, sites=[site, site_br])
                e_2x2_diag_loc += (res + res.conj()).real
                res = self.t3*env.measure_2x2(c_B, cp_A, sites=[site, site_br])
                e_2x2_diag_loc += (res + res.conj()).real
                res = self.t3*env.measure_2x2(c_A, cp_B, sites=[site, site_br])
                e_2x2_diag_loc += (res + res.conj()).real

                energy_diag += e_2x2_diag_loc

                site_b = psi.nn_site(site, "b")
                site_r = psi.nn_site(site, "r")
                e_2x2_anti_diag_loc = self.V3*env.measure_2x2(n_B, n_A, sites=[site_b, site_r])

                res = self.t3*env.measure_2x2(c_B, cp_A, sites=[site_b, site_r])
                e_2x2_anti_diag_loc += (res + res.conj()).real

                energy_anti_diag += e_2x2_anti_diag_loc

        energy_per_site = (
            energy_onsite + energy_horz + energy_vert + energy_diag + energy_anti_diag
        ) / N
        return energy_per_site.real

    # @profile
    def energy_per_site_rdm(self, psi, env):
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

        energy_onsite, energy_horz, energy_vert, energy_diag, energy_anti_diag = (
            0,
            0,
            0,
            0,
            0,
        )

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

        energy_onsite, energy_horz, energy_vert, energy_diag, energy_anti_diag = (
            0,
            0,
            0,
            0,
            0,
        )
        ncon_order1 = ((1, 2), (3, 4), (2, 1, 4, 3))
        ncon_order2 = ((1, 2, 5), (3, 4, 5), (2, 1, 4, 3))
        for site in psi.sites():
            # onsite
            onsite_rdm, onsite_norm = rdm1x1(site, psi, env)  # s s'

            op = (
                self.V1 * (n_A @ n_B)
                - self.mu * (n_A + n_B)
                - self.t1 * (cp_A @ c_B + cp_B @ c_A).remove_zero_blocks()
                + self.m * (n_A - n_B)
            )
            e_1x1_loc = yastn.ncon([op, onsite_rdm], ((1, 2), (2, 1)))
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
            v_bond = Bond(site, psi.nn_site(site, "b"))
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

            if self.V2 != 0 or self.V3 != 0 or self.t2 != 0 or self.t3 != 0:
                # plaq_rdm, plaq_rdm_norm  = rdm2x2(site, psi, env)  # s0 s0' s1 s1' s2 s2' s3 s3'
                diag_rdm, diag_rdm_norm = rdm2x2_diagonal(
                    site, psi, env
                )  # s0 s0' s3 s3'
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
                e_2x2_diag_loc += -self.t3 * (
                    -yastn.ncon([ci_B, cjp_A, diag_rdm], ncon_order2)
                    + yastn.ncon([cip_B, cj_A, diag_rdm], ncon_order2)
                    - yastn.ncon([ci_A, cjp_B, diag_rdm], ncon_order2)
                    + yastn.ncon([cip_A, cj_B, diag_rdm], ncon_order2)
                )
                energy_diag += e_2x2_diag_loc.to_number()

                # anti-diagonal bond
                anti_diag_rdm, anti_diag_rdm_norm = rdm2x2_anti_diagonal(site, psi, env)
                # anti_diag_rdm = yastn.trace(plaq_rdm, ((0, 6), (1, 7)))  # s1 s1' s2 s2'
                energy_anti_diag += (
                    self.V3
                    * yastn.ncon([n_B, n_A, anti_diag_rdm], ncon_order1).to_number()
                )

                site_b = psi.nn_site(site, "b")
                site_r = psi.nn_site(site, "r")
                ordered = psi.geometry.f_ordered(site_b, site_r)

                cip_B, cj_A = op_order(cp_B, c_A, ordered, fermionic=True)
                ci_B, cjp_A = op_order(c_B, cp_A, ordered, fermionic=True)
                energy_anti_diag += (
                    -self.t3
                    * (
                        -yastn.ncon([ci_B, cjp_A, anti_diag_rdm], ncon_order2)
                        + yastn.ncon([cip_B, cj_A, anti_diag_rdm], ncon_order2)
                    ).to_number()
                )


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
            n = obs_nA + obs_nB
            obs[psi.site2index(s0)] = [obs_nA.item().real, obs_nB.item().real, n.item().real, m.item().real]
            print(f"nA: {obs_nA:.4f}, nB: {obs_nB:.4f}, m: {m:.4f}, n: {n:.4f}")
            log.log(
                logging.INFO,
                f"nA: {obs_nA:.4f}, nB: {obs_nB:.4f}, m: {m:.4f}, n: {n:.4f}",
            )

            # obs_cA = measure_rdm_1site(s0, psi, env, c_A).item()
            # obs_cpA = measure_rdm_1site(s0, psi, env, cp_A).item()
            # print(f"cA: {obs_cA:.4f}, cp_A: {obs_cpA:.4f}")
            # op_n_list[psi.site2index(s0)] = n_A - obs_nA*I
            # op_c_list[psi.site2index(s0)] = c_A
            # op_cp_list[psi.site2index(s0)] = cp_A

        # corr_n = measure_rdm_corr_1x2(s0, 30, psi, env, op_list[psi.site2index(s0)], op_list)
        # corr_cc = measure_rdm_corr_1x2(s0, 30, psi, env, op_c_list[psi.site2index(s0)], op_cp_list)
        # print(corr_cc)
        # D = 2*args.bond_dim
        # D = sum([args.bond_dims[t] for t in args.bond_dims.keys()])
        # obs_file = f"data/U1/obs_FCI_fp_1x1_D_{D:d}_U1_chi_{args.chi:d}_V_{self.V1:.2f}_t1_{self.t1:.2f}_t2_{self.t2:.2f}_t3_{self.t3:.2f}_mu_{self.mu:.3f}.json"
        # with open(obs_file, "w") as f:
        #     json.dump([], f)

        # # Dynamic appending in a loop
        # new_data = {"obs": obs}

        # # Read the current data
        # with open(obs_file, "r") as f:
        #     data = json.load(f)

        # # Append the new entry
        # data.append(new_data)

        # # Write the updated data back
        # with open(obs_file, "w") as f:
        #     json.dump(data, f, indent=4)

        return obs


def test_state(config=None, noise=0):
    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
        }
    )

    if config is None:  # use default
        from yastn.yastn.backend import backend_np
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


def random_3x3_state_U1(bond_dims, config=None):
    # 3x3 unit-cell pattern in the square lattice:
    # A B C
    # B C A
    # C A B
    # with one tensor in {A B C} having a extra charge

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
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_charge=0):
        if dummy_leg_charge != 0:
            dummy_leg = yastn.Leg(config, s=1, t=(dummy_leg_charge,), D=(1,))
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
            (0, 0): rand_tensor_norm(0, legs, dummy_leg_charge=-1),
            (0, 1): rand_tensor_norm(0, legs, dummy_leg_charge=0),
            (0, 2): rand_tensor_norm(0, legs, dummy_leg_charge=0),
        },
    )
    return psi


def random_1x3_state_U1(bond_dims, config=None):
    # 1x3 unit-cell pattern in the square lattice:
    # A B C

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
        }
    )
    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_charge=0):
        if dummy_leg_charge != 0:
            dummy_leg = yastn.Leg(config, s=1, t=(dummy_leg_charge,), D=(1,))
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
            (0, 0): rand_tensor_norm(0, legs, dummy_leg_charge=-1),
            (0, 1): rand_tensor_norm(0, legs, dummy_leg_charge=0),
            (0, 2): rand_tensor_norm(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_3x1_state_U1(bond_dims, config=None):
    # 1x3 unit-cell pattern in the square lattice:
    # A B C

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (1, 0): 1,
            (2, 0): 2,
        }
    )
    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_charge=0):
        if dummy_leg_charge != 0:
            dummy_leg = yastn.Leg(config, s=1, t=(dummy_leg_charge,), D=(1,))
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
            (0, 0): rand_tensor_norm(0, legs, dummy_leg_charge=-1),
            (1, 0): rand_tensor_norm(0, legs, dummy_leg_charge=0),
            (2, 0): rand_tensor_norm(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_1x1_state_U1(bond_dims, config=None):
    # 1x1 unit-cell in the square lattice
    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
        }
    )

    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def rand_tensor_norm(n, legs, dummy_leg_charge=0):
        if dummy_leg_charge != 0:
            dummy_leg = yastn.Leg(config, s=1, t=(dummy_leg_charge,), D=(1,))
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
            (0, 0): rand_tensor_norm(
                0, legs, dummy_leg_charge=-1
            ),  # 1 electron per unit-cell
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
        from yastn.yastn.backend import backend_np
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
            if dummy_leg_flag == "even":
                dummy_leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
            elif dummy_leg_flag == "odd":
                dummy_leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
            elif dummy_leg_flag == "even_odd":
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
    "--t3", type=float, default=0.0, help="2nd. nearest-neighbor hopping amplitude"
)
parser.add_argument(
    "--phi", type=float, default=0.0, help="phase of the 2nd. nearest-neighbor hopping"
)
parser.add_argument("--mu", type=float, default=0.0, help="chemical potential")
parser.add_argument("--m", type=float, default=0.0, help="Semenoff mass")
parser.add_argument("--pattern", default="1x1", help="unit-cell of iPEPS: choice={1x1, 3x3}")
parser.add_argument("--init_state_file", default=None, help="initial state file")

def parse_dict(input_string):
    try:
        # Use `ast.literal_eval` to safely evaluate the string
        parsed = ast.literal_eval(input_string)
        if isinstance(parsed, dict):
            return parsed
        else:
            raise argparse.ArgumentTypeError("Input is not a valid dictionary.")
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid dictionary format: {e}")

parser.add_argument("--bond_dims", type=parse_dict, help="dict of bond dimensions keyed on charge sectors  (e.g., \"{'charge1': 'D1', 'charge2': D2}\")")

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
        "--opt_max_iter",
        "1000",
        "--omp_cores",
        "16",
        "--CTMARGS_ctm_env_init_type",
        "eye",
        "--OPTARGS_fd_eps",
        "1e-8",
        "--OPTARGS_no_opt_ctm_reinit",
        "--OPTARGS_no_line_search_ctm_reinit",
        "--GLOBALARGS_dtype",
        "complex128",
        # "--OPTARGS_opt_log_grad",
        # "--CTMARGS_fwd_checkpoint_move",
        "--OPTARGS_line_search",
        # "backtracking",
        "strong_wolfe",
    ],
    namespace=args,
)
cfg.configure(args)



def main():
    global args
    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=True,
        default_device=cfg.global_args.device,
        default_dtype=cfg.global_args.dtype,
        tensordot_policy="no_fusion",
    )

    torch.set_num_threads(args.omp_cores)
    yastn_config.backend.random_seed(args.seed)

    pd = {}
    pd["V1"], pd["V2"], pd["V3"] = args.V1, args.V2, args.V3
    pd["t1"], pd["t2"], pd["t3"] = args.t1, args.t2, args.t3
    # pd["phi"] = args.phi
    pd["t2"], pd["t3"] = 0.7*pd["t1"], -0.9*pd["t1"]
    pd["phi"] = 0.35 * np.pi
    pd["m"], pd["mu"] = args.m, args.mu

    @torch.no_grad()
    def ctm_conv_check(env, history, corner_tol):
        converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
        print("max_dsv:", max_dsv)
        log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
        return converged, history

    def get_converged_env(
        env,
        method="2site",
        max_sweeps=100,
        iterator_step=1,
        opts_svd=None,
        corner_tol=1e-8,
    ):
        t_ctm, t_check = 0.0, 0.0
        t_ctm_prev = time.perf_counter()
        converged, conv_history = False, []

        for sweep in range(max_sweeps):
            env.update_(
                opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=True
            )
            t_ctm_after = time.perf_counter()
            t_ctm += t_ctm_after - t_ctm_prev
            t_ctm_prev = t_ctm_after

            converged, conv_history = ctm_conv_check(env, conv_history, corner_tol)
            if converged:
                break
        env.update_(
            opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=True
        )

        return env, converged, conv_history, t_ctm, t_check


    def loss_fn(state, ctm_env_in, opt_context):
        state.sync_()
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            print("Reinit")
            chi = cfg.main_args.chi//2
            env_leg = yastn.Leg(yastn_config, s=1, t=(0, 1), D=(chi, chi))
            # env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
            ctm_env_in = EnvCTM(state, init=ctm_args.ctm_env_init_type, leg=env_leg)

        state_params = state.get_parameters()
        env_params, slices = env_raw_data(ctm_env_in)
        env_out_data = FixedPoint.apply(env_params, slices, yastn_config, ctm_env_in, opts_svd, cfg.main_args.chi, 1e-10, ctm_args, *state_params)
        ctm_env_out, ctm_log, t_ctm, t_check = FixedPoint.ctm_env_out, FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check
        refill_env(ctm_env_out, env_out_data, FixedPoint.slices)
        # 2) evaluate loss with converged environment
        loss = model.energy_per_site(state, ctm_env_out)  # H= H_0 - mu * (nA + nB)
        # assert abs(loss - model.energy_per_site_rdm(state, ctm_env_out)) < 1e-9
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
            print(
                "LS "
                + ", ".join(
                    [f"{epoch}", f"{loss}"]
                    + [f"{x.item()}" for x in model.get_parameters()]
                )
            )
            # print("LS " + ", ".join([f"{epoch}", f"{(loss+0.75*pd['V1'])/2}"]))
        else:
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
            print(
                ", ".join(
                    [f"{epoch}", f"{loss}"]
                    + [f"{x.item()}" for x in model.get_parameters()]
                )
            )
            # print(", ".join([f"{epoch}", f"{(loss+0.75*pd['V1'])/2}"]))

        model.eval_obs(state, ctm_env)

    bond_dims = args.bond_dims
    D = sum([bond_dims[t] for t in bond_dims.keys()])
    seed_dir = f"FCI_U1_data_seed_{args.seed:d}"
    if not os.path.exists(seed_dir):
        os.mkdir(seed_dir)
    state_file = f"FCI_U1_data_seed_{args.seed}/FCI_fp_{args.pattern}_D_{D:d}_U1_chi_{args.chi:d}_V_{pd['V1']:.2f}_t1_{pd['t1']:.2f}_t2_{pd['t2']:.2f}_t3_{pd['t3']:.2f}_phi_{pd['phi']/np.pi:.2f}_mu_{pd['mu']:.3f}"

    args, unknown_args = parser.parse_known_args(
        [
            "--out_prefix",
            state_file,
        ],
        namespace=args,
    )
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    cfg.configure(args)
    log.log(logging.INFO, "device: "+cfg.global_args.device)
    log.log(logging.INFO, f"bond_dims:{bond_dims}")
    model = tV_model(yastn_config, pd)

    opts_svd = {
        "D_total": cfg.main_args.chi,
        "tol": cfg.ctm_args.projector_svd_reltol,
        "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
        "truncate_multiplets": True,
    }

    if args.init_state_file is None or not os.path.exists(args.init_state_file):
        if args.pattern == '1x1':
            state = random_1x1_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3':
            state = random_3x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '1x3':
            state = random_1x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x1':
            state = random_3x1_state_U1(bond_dims=bond_dims, config=yastn_config)
    else:
        state = load_PepsAD(yastn_config, args.init_state_file)
        log.log(logging.INFO, "loaded " + args.init_state_file)
        print("loaded ", args.init_state_file)
        state.add_noise_(noise=0.5)

    chi = cfg.main_args.chi
    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
    ctm_env_in = EnvCTM(state, init=cfg.ctm_args.ctm_env_init_type, leg=env_leg)
    optimize_state(state, ctm_env_in, loss_fn, obs_fn=obs_fn, post_proc=None)

if __name__ == "__main__":
    if len(unknown_args) > 0:
        print("args not recognized: " + str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()
