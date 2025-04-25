from typing import Union
import torch
import numpy as np
import yastn.yastn as yastn
from yastn.yastn.tn.fpeps.envs.rdm import *
from yastn.yastn.tn.fpeps import RectangularUnitcell, Bond
from yastn.yastn.backend import backend_np
from ipeps.integration_yastn import PepsAD, load_PepsAD
from yastn.yastn.sym import sym_U1, sym_Z2

class tV_model:
    def __init__(self, config, V1 : float = 0, V2:float = 0, V3:float = 0,
                 t1 : float= 1., t2:float= 0, t3:float=0, phi:float=0, mu: Union[float, torch.Tensor] =0, m:float=0, **kwargs):
        """
        Parameters
        ----------
            config : module | _config(NamedTuple)
                :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
            V1 : n.n. interaction
            V2 : 2nd n.n. interaction
            V3 : 3rd n.n. interaction
            t1: amplitude of n.n. hopping,
            t2: amplitude of the 2nd n.n. hopping
            t3: amplitude of the 3rd n.n. hopping
            phi: phase of the 2nd n.n. hopping along the positive direction
            mu: chemical potential
            m: semenoff mass
        """
        self.config = config
        self.dtype = config.default_dtype
        self.device = config.default_device

        self.V1 = V1
        self.V2 = V2
        self.V3 = V3
        self.mu = mu
        self.m = m
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.phi = phi

        self.sf = yastn.operators.SpinfulFermions(**self.config._asdict())
        # build once, then reuse
        self.ops = { "n_A": self.sf.n(spin="u"),  # parity-even operator, no swap gate needed
            "n_B": self.sf.n(spin="d"),
            "c_A": self.sf.c(spin="u"),
            "cp_A": self.sf.cp(spin="u"),
            "c_B": self.sf.c(spin="d"),
            "cp_B": self.sf.cp(spin="d"),
            "I": self.sf.I()
        }


    def get_parameters(self):
        if hasattr(self.mu,'requires_grad'):
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
        N = psi.Nx * psi.Ny


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
        N = psi.Nx * psi.Ny

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


    def eval_obs(self, psi : yastn.tn.fpeps.Peps, env : yastn.tn.fpeps.EnvCTM) -> dict[str,float]:
        n_A, n_B, c_A, cp_A, c_B, cp_B, I = tuple( self.ops[k] for k in ("n_A", "n_B", "c_A", "cp_A", "c_B", "cp_B", "I") )

        obs = {}
        op_n_list = {}
        op_c_list = {}
        op_cp_list = {}
        for s0 in psi.sites():
            obs_nA = measure_rdm_1site(s0, psi, env, n_A)
            obs_nB = measure_rdm_1site(s0, psi, env, n_B)
            m = abs(obs_nA - obs_nB)
            obs.update({f"nA_{s0}": obs_nA.item().real, f"nB_{s0}": obs_nB.item().real})

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


# Define a set of initial states for the tV model
#

# utility functions
#
def _rand_tensor(config, n:int, legs:Sequence[yastn.tensor.Leg], dummy_leg_flag:Union[str,None]=None)->yastn.tensor.Tensor:
    """
    Create random tensor with given legs. Optionally, create charged tensor by appending extra dummy leg (effective dimension 1 with single charge sector)
    carrying the charge. This keeps the tensor invariant under the symmetry (i.e. in trivial rep).

    By convention, dummy leg is the last leg of the tensor and it is fused with the physical leg.

    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        dummy_leg_flag : str | None
            * 'even' : even charge sector
            * 'odd' : odd charge sector'
            * 'even_odd' : both charge sectors
    """
    if dummy_leg_flag is not None:
        if dummy_leg_flag == 'even':
            dummy_leg = yastn.Leg(config, s=1, t=(0,), D=(1,))
        elif dummy_leg_flag == 'odd':
            dummy_leg = yastn.Leg(config, s=1, t=(1,), D=(1,))
        elif dummy_leg_flag == 'even_odd':
            dummy_leg = yastn.Leg(config, s=1, t=(0, 1), D=(1, 1))
        legs = legs + [dummy_leg]
        A = yastn.rand(config=config, n=n, legs=legs)
        axes = [i for i in range(len(legs) - 2)] + [(len(legs) - 2, len(legs) - 1)]
        A = A.fuse_legs(axes=axes)  # Fuse the physical leg with the dummy leg
    else:
        A = yastn.rand(config=config, n=n, legs=legs)
    return A


def state_2x1(config= yastn.make_config(backend=backend_np, fermionic=True, sym="Z2"), noise=0):
    """
    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        noise : float
    """
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
    psi = PepsAD(RectangularUnitcell([[0, 1],]), parameters={(0, 0): t0, (0, 1): t1})

    return psi


def random_1x1_state_Z2(config=yastn.make_config(backend=backend_np, fermionic=True), bond_dim=(1, 1)):
    """
    Coarse-grain honeycomb lattice to square lattice by combining two sites within each honeycomb unit cell.
    Resulting iPEPS has a 1x1 unit cell on the effective square lattice. The filling is fixed at one electron per unit-cell.

    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        noise : float
    """
    if config is None:  # use default
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="Z2")
    assert config.sym == sym_Z2, "Expecting Z2 symmetry"

    D0, D1 = bond_dim
    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(2, 2)),
    ]

    psi = PepsAD(
        RectangularUnitcell([[0],]),
        parameters={
            (0, 0): _rand_tensor(config, 0, legs, dummy_leg_flag='odd'),
        },
    )
    return psi

def random_1x3_state_Z2(config= yastn.make_config(backend=backend_np, fermionic=True), bond_dim=(1, 1)):
    """
    Coarse-grain honeycomb lattice to square lattice by combining two sites within each honeycomb unit cell.
    Resulting iPEPS has a 1x3 unit cell on the effective square lattice with 3 unique tensors arranged in following pattern::

        1x3 unit-cell pattern on the square lattice:
        A B C

    The filling is fixed at 1/3 electron per unit-cell.

    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        noise : float
    """
    assert config.sym == sym_Z2, "Expecting Z2 symmetry"

    # define legs of on-site tensor
    D0, D1 = bond_dim
    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(2, 2)),
    ]

    psi = PepsAD(RectangularUnitcell([[0,1,2],]),
        parameters={
            (0, 0): _rand_tensor(config, 0, legs, dummy_leg_flag='odd'),
            (0, 1): _rand_tensor(config, 0, legs, dummy_leg_flag='even'),
            (0, 2): _rand_tensor(config, 0, legs, dummy_leg_flag='even'),
        },
    )
    return psi

def random_3x3_state_Z2(config= yastn.make_config(backend=backend_np, fermionic=True), bond_dim=(1, 1)):
    """
    Coarse-grain honeycomb lattice to square lattice by combining two sites within each honeycomb unit cell.
    Resulting iPEPS has a 3x3 unit cell on the effective square lattice with 3 unique tensors arranged in following pattern::

        3x3 unit-cell pattern on the square lattice:
        A B C
        B C A
        C A B

    The filling is fixed at 1/3 electron per unit-cell.

    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        noise : float
    """
    assert config.sym == sym_Z2, "Expecting Z2 symmetry"

    # define legs of on-site tensor
    D0, D1 = bond_dim
    legs = [
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=-1, t=(0, 1), D=(D0, D1)),
        yastn.Leg(config, s=1, t=(0, 1), D=(2, 2)),
    ]

    psi = PepsAD(RectangularUnitcell([[0,1,2],[1,2,0],[2,0,1]]),
        parameters={
            (0, 0): _rand_tensor(config, 0, legs, dummy_leg_flag='odd'),
            (0, 1): _rand_tensor(config, 0, legs, dummy_leg_flag='even'),
            (0, 2): _rand_tensor(config, 0, legs, dummy_leg_flag='even'),
        },
    )
    return psi


def random_3x3_state_U1(bond_dims, config=None):
    # 3x3 unit-cell pattern on the square lattice:
    # A B C
    # B C A
    # C A B
    # with one tensor in {A B C} having an extra charge

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

    assert config.sym == sym_U1, "Expecting U1 symmetry"

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (0, 1): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 2): _rand_tensor(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_3x3_2_state_U1(bond_dims, config=None):
    # 3x3 unit-cell pattern on the square lattice:
    # A B C
    # C A B
    # B C A
    # with one tensor in {A B C} having an extra charge

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (1, 0): 2,
            (1, 1): 0,
            (1, 2): 1,
            (2, 0): 1,
            (2, 1): 2,
            (2, 2): 0,
        }
    )
    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    assert config.sym == sym_U1, "Expecting U1 symmetry"

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (0, 1): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 2): _rand_tensor(0, legs, dummy_leg_charge=0),
        },
    )
    return psi


def random_1x3_state_U1(bond_dims, config=None):
    # 1x3 unit-cell pattern on the square lattice:
    # A B C
    # with one tensor in {A B C} having an extra charge

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

    assert config.sym == sym_U1, "Expecting U1 symmetry"

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (0, 1): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 2): _rand_tensor(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_1x6_state_U1(bond_dims, config=None):
    # 1x6 unit-cell pattern on the square lattice:
    # A B C D E F
    # with two tensors in {A B C D E F} having an extra charge

    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 3,
            (0, 4): 4,
            (0, 5): 5,
        }
    )
    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    assert config.sym == sym_U1, "Expecting U1 symmetry"

    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (0, 1): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 2): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 3): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (0, 4): _rand_tensor(0, legs, dummy_leg_charge=0),
            (0, 5): _rand_tensor(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_3x1_state_U1(bond_dims, config=None):
    # 1x3 unit-cell pattern on the square lattice:
    # A B C
    # with one tensor in {A B C} having an extra charge

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

    assert config.sym == sym_U1, "Expecting U1 symmetry"
    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(0, legs, dummy_leg_charge=-1),
            (1, 0): _rand_tensor(0, legs, dummy_leg_charge=0),
            (2, 0): _rand_tensor(0, legs, dummy_leg_charge=0),
        },
    )
    return psi

def random_1x1_state_U1(bond_dims, config=None):
    # 1x1 unit-cell on the square lattice with the tensor carrying one external charge.
    geometry = RectangularUnitcell(
        pattern={
            (0, 0): 0,
        }
    )

    if config is None:  # use default
        from yastn.yastn.backend import backend_np
        config = yastn.make_config(backend=backend_np, fermionic=True, sym="U1")

    assert config.sym == sym_U1, "Expecting U1 symmetry"
    charges = tuple(bond_dims.keys())
    Ds = tuple([bond_dims[t] for t in charges])

    legs = [
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=-1, t=charges, D=Ds),
        yastn.Leg(config, s=1, t=(0, 1, 2), D=(1, 2, 1)),
    ]

    def _rand_tensor(n, legs, dummy_leg_charge=0):
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
        parameters={
            (0, 0): _rand_tensor(
                0, legs, dummy_leg_charge=-1
            ),  # 1 electron per unit-cell
        },
    )
    return psi




def random_ipess_state(config= yastn.make_config(backend=backend_np, fermionic=True, sym="Z2"), bond_dim=(1, 1)):
    """
    Coarse-grain honeycomb lattice to square lattice by combining two sites within each honeycomb unit cell.
    The on-site tensor is given further structure (iPESS) by expressing it as a contraction of two rank-4 tensors::

          0       2   1       t(0)  r(3)
          |        \ /         \   /
          A--3  x   B--3   =>    B--
         / \        |            |   -> physical_a, physical_b (4) [in this order]
        1   2       0            A--
                               /   \
                             l(1)  b(2)

    Parameters
    ----------
        config : module | _config(NamedTuple)
            :ref:`YASTN configuration <tensor/configuration:yastn configuration>`
        noise : float
    """
    assert config.sym == "Z2", "Expecting Z2 symmetry"

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

    def get_tensors(parameters):
        return {
            (0, 0): yastn.einsum('xlbp,xrts->tlbrps',parameters[(0, 0)]["A"], parameters[(0, 0)]["B"]).fuse_legs(axes=(0,1,2,3,(4, 5))),
        }

    psi = PepsAD(
        RectangularUnitcell([[0],]),
        parameters={
            (0, 0): {
                "A": _rand_tensor(0, A_legs, dummy_leg_flag="odd"),
                "B": _rand_tensor(0, B_legs, dummy_leg_flag="even"),
            },
        },
        get_tensors= get_tensors,
    )
    return psi
