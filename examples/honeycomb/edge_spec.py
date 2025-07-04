import json
import os
import pickle
from itertools import cycle

import context
import config as cfg
import numpy as np
import torch

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs
from functools import partial
import matplotlib.pyplot as plt

import yastn.yastn as yastn
from yastn.yastn.sym import sym_Z2, sym_U1
from ipeps.integration_yastn import PepsAD, load_PepsAD
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tn.fpeps.envs.rdm import op_order


@torch.no_grad()
def ctm_conv_check(env, history, corner_tol):
    converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
    print(len(history), max_dsv)
    return converged, history


@torch.no_grad()
def get_converged_env(
    env,
    method="2site",
    max_sweeps=100,
    iterator_step=1,
    opts_svd=None,
    corner_tol=1e-8,
):
    converged, conv_history = False, []

    for sweep in range(max_sweeps):
        env.update_(
            opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move="reentrant",
        )
        converged, conv_history = ctm_conv_check(env, conv_history, corner_tol)
        if converged:
            break
    return env, converged, conv_history

def ent_spec_cylinder_abelian(yastn_config, state, env, Lx, charge_sector=0, num_evals=8):
    # - Perform CTMRG to obtain the environment tensors
    # - Approximate the reduced density matrices on the virtual space by the environment tensors

    # Assume single-site
    assert len(state.sites()) == 1
    site = list(state.sites())[0]
    TL, TR = env[site].l, env[site].r
    TL = TL.transpose(axes=(2, 1, 0))
    TL = TL.unfuse_legs(axes=(1,))
    TR = TR.unfuse_legs(axes=(1,))

    #  0   __ 1 (top)    (top) 1__    0
    #  | /                        \  |
    #  TL----2 (bot)     (bot) 2----TR
    #  |                            |
    #  3                            3

    def mv(v0, meta):
        # V: YASTN Tensor
        # TR action
        v0 = torch.as_tensor(v0, dtype=TL.dtype, device=TL.device)
        V = yastn.decompress_from_1d(v0, meta)
        TR_start = TR.swap_gate(axes=(2, (0, 3)))
        V = TR_start.tensordot(V, ([2], [0]))

        #
        #       | \
        #   1--TR |  |-----|
        #    /-|--|--|     |
        #      2  0  |  V  |
        #    3 ------|     |
        #   ...      |     |
        #  Lx+1 -----|-----|

        for i in range(1, Lx - 1):
            #
            #       | \                                  | \
            #   1--TR | |-----|                      3--TR |   |-----|
            #    /-|--|-|     |                       /-|--|---|     |
            #	   |  | |	  |							   |   |	 |
            #   2--TR | |     |                      4--TR |   |     |
            #    /-|--|-|     |                       /-|--|---|     |
            #     ... | |     |        --->          ...   |   |     |
            #   i-TR  | |  V  |                    0 --TR  |   |  V  |
            #   /-|---|-|     |                       /-|--|---|     |
            #    i+1  0 |     |                         1  2   |     |
            #    i+2 ---|     |                       i+3 -----|     |
            #             ...                               	 ...
            #   Lx+1 ---|-----|                      Lx+1 -----|-----|
            TR_middle = TR.swap_gate(axes=(2,3))
            V = V.swap_gate(axes=(0, i+2))
            V = TR_middle.tensordot(V, ([0, 2], [i + 1, i + 2]))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, Lx + 2))
            V = V.transpose(order)

        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        V = V.tensordot(TR, ([Lx, Lx + 1, 0], [0, 2, 3]))


        TL_start = TL.swap_gate(axes=(2,(0,3)))
        V = TL_start.tensordot(V, ([1], [0]))
        for i in range(1, Lx - 1):
            # TL action (contract TL top with V)
            #
            #    / \                            / \
            #    | TL---|-----|                 | TL---|-----|
            #  1-|-|-\  |     |               3-|-|-\  |     |
            #	 | |    |	  |				    | |    |	 |
            #    | TL---|     |                 | TL---|     |
            #  2-|-|-\	|     |               4-|-|-\  |     |
            #    |...   |     |        --->     |...   |     |
            #    | TL---|  V  |                 | TL---|  V  |
            #  i-|-|-\  |     |               0-|-|-\  |     |
            #    0 i+1  |     |                 2 1    |     |
            #    i+2 ---|     |                 i+3 ---|     |
            #             ...                            ...
            #   Lx+1 ---|-----|                Lx+1 ---|-----|
            TL_middle = TL.swap_gate(axes=(2,3))
            V = TL_middle.tensordot(V, ([0, 1], [i + 1, i + 2]))
            V = V.swap_gate(axes=(0,2))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, Lx + 2))
            V = V.transpose(order)

        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        V = V.tensordot(TL, ([Lx, Lx + 1, 0], [0, 1, 3]))
        v1, _ = yastn.Tensor.compress_to_1d(V, meta)
        return v1


    leg = TR.get_legs(axes=1)
    legs = [
        leg for _ in range(Lx)
    ]
    v0 = yastn.rand(config=yastn_config, n=charge_sector, legs=legs)
    data, meta = yastn.Tensor.compress_to_1d(v0)

    sigma_L_sigma_R = LinearOperator(
        (len(data), len(data)),
        matvec=partial(mv, meta=meta) ,
        dtype="complex128" if TL.is_complex() else "float64",
    )

    vals, evecs = eigs(
        sigma_L_sigma_R, k=num_evals, v0=data, which="LM", return_eigenvectors=True
    )

    return vals, evecs, meta


def ent_spec_cylinder_Lx(yastn_config, env, Lx, start_site, charge_sector=0, num_evals=8):
    # Compute the entanglement spectrum of PEPS ansatz for a cut specified by the start_site
    # Lx: width of the cylinder, measure in unit-cells (finite width along x-direction)
    # start_site: the site whose left-bond is crossed by the entanglement cut

    TLs, TRs = [], []
    site = start_site
    for _ in range(env.psi.dims[0]): # iterate in x-direction
        r_site = env.nn_site(site, "l") # env[r_site].r matches with env[site].l
        TL, TR = env[site].l, env[r_site].r
        TL = TL.transpose(axes=(2, 1, 0))
        TL = TL.unfuse_legs(axes=(1,))
        TR = TR.unfuse_legs(axes=(1,))
        TLs.append(TL)
        TRs.append(TR)
        site = env.nn_site(site, "b")
    TL_gen = cycle(TLs)
    TR_gen = cycle(TRs)

    #  0   __ 1 (top)    (top) 1__    0
    #  | /                        \  |
    #  TL----2 (bot)     (bot) 2----TR
    #  |                            |
    #  3                            3

    def mv(v0, meta):
        # V: YASTN Tensor
        # TR action
        v0 = torch.as_tensor(v0, dtype=TLs[0].dtype, device=TLs[0].device)
        V = yastn.decompress_from_1d(v0, meta)
        TR = next(TR_gen)
        TR_start = TR.swap_gate(axes=(2, (0, 3)))
        V = TR_start.tensordot(V, ([2], [0]))

        #
        #       | \
        #   1--TR |  |-----|
        #    /-|--|--|     |
        #      2  0  |  V  |
        #    3 ------|     |
        #   ...      |     |
        #  Lx+1 -----|-----|

        for i in range(1, env.psi.dims[0]*Lx - 1):
            #
            #       | \                                  | \
            #   1--TR | |-----|                      3--TR |   |-----|
            #    /-|--|-|     |                       /-|--|---|     |
            #	   |  | |	  |							   |   |	 |
            #   2--TR | |     |                      4--TR |   |     |
            #    /-|--|-|     |                       /-|--|---|     |
            #     ... | |     |        --->          ...   |   |     |
            #   i-TR  | |  V  |                    0 --TR  |   |  V  |
            #   /-|---|-|     |                       /-|--|---|     |
            #    i+1  0 |     |                         1  2   |     |
            #    i+2 ---|     |                       i+3 -----|     |
            #             ...                               	 ...
            # L*Lx+1 ---|-----|                    L*Lx+1 -----|-----|
            TR = next(TR_gen)
            TR_middle = TR.swap_gate(axes=(2,3))
            V = V.swap_gate(axes=(0, i+2))
            V = TR_middle.tensordot(V, ([0, 2], [i + 1, i + 2]))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, env.psi.dims[0]*Lx + 2))
            V = V.transpose(order)

        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        TR = next(TR_gen)
        V = V.tensordot(TR, ([env.psi.dims[0]*Lx, env.psi.dims[0]*Lx + 1, 0], [0, 2, 3]))


        TL = next(TL_gen)
        TL_start = TL.swap_gate(axes=(2,(0,3)))
        V = TL_start.tensordot(V, ([1], [0]))
        for i in range(1, env.psi.dims[0]*Lx - 1):
            # TL action (contract TL top with V)
            #
            #    / \                            / \
            #    | TL---|-----|                 | TL---|-----|
            #  1-|-|-\  |     |               3-|-|-\  |     |
            #	 | |    |	  |				    | |    |	 |
            #    | TL---|     |                 | TL---|     |
            #  2-|-|-\	|     |               4-|-|-\  |     |
            #    |...   |     |        --->     |...   |     |
            #    | TL---|  V  |                 | TL---|  V  |
            #  i-|-|-\  |     |               0-|-|-\  |     |
            #    0 i+1  |     |                 2 1    |     |
            #    i+2 ---|     |                 i+3 ---|     |
            #             ...                            ...
            # L*Lx+1 ---|-----|              L*Lx+1 ---|-----|
            TL = next(TL_gen)
            TL_middle = TL.swap_gate(axes=(2,3))
            V = TL_middle.tensordot(V, ([0, 1], [i + 1, i + 2]))
            V = V.swap_gate(axes=(0,2))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, env.psi.dims[0]*Lx + 2))
            V = V.transpose(order)

        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        TL = next(TL_gen)
        V = V.tensordot(TL, ([env.psi.dims[0]*Lx, env.psi.dims[0]*Lx + 1, 0], [0, 1, 3]))
        v1, _ = yastn.Tensor.compress_to_1d(V, meta)
        return v1


    legs = [next(TR_gen).get_legs(axes=1) for _ in range(env.psi.dims[0]*Lx)]
    v0 = yastn.rand(config=yastn_config, n=charge_sector, legs=legs)
    data, meta = yastn.Tensor.compress_to_1d(v0)

    sigma_L_sigma_R = LinearOperator(
        (len(data), len(data)),
        matvec=partial(mv, meta=meta) ,
        dtype="complex128" if TL.is_complex() else "float64",
    )

    vals, evecs = eigs(
        sigma_L_sigma_R, k=num_evals, v0=data, which="LM", return_eigenvectors=True
    )

    return vals, evecs, meta


def ent_spec_cylinder_Ly(yastn_config, env, Ly, start_site, charge_sector=0, num_evals=8):
    # Compute the entanglement spectrum of PEPS ansatz for a cut specified by the start_site
    # Ly: width of the cylinder, measure in unit-cells (finite width along y-direction)
    # start_site: the site whose top-bond is crossed by the entanglement cut

    TTs, TBs = [], []
    site = start_site
    for _ in range(env.psi.dims[1]): # iterate in y-direction
        b_site = env.nn_site(site, "t")
        TT, TB = env[site].t, env[b_site].b
        TB = TB.transpose(axes=(2, 1, 0))
        TB = TB.unfuse_legs(axes=(1,))
        TT = TT.unfuse_legs(axes=(1,))
        #  0   __ 2 (bot)    (bot) 2__   0
        #  | /                        \  |
        #  TB----1 (top)     (top) 1----TT
        #  |                             |
        #  3                             3
        TT = TT.swap_gate(axes=(1,2))
        TB = TB.swap_gate(axes=(1,2))
        #  0                             0
        #  | /\                       /\ |
        #  TB-\--1 (top)     (top)1--/--TT
        #  |  \--2 (bot)     (bot)2--/   |
        #  3                             3
        TTs.append(TT)
        TBs.append(TB)
        site = env.nn_site(site, "r")
    TT_gen = cycle(TTs)
    TB_gen = cycle(TBs)

    def mv(v0, meta):
        # V: YASTN Tensor
        # TR action
        v0 = torch.as_tensor(v0, dtype=TBs[0].dtype, device=TBs[0].device)
        V = yastn.decompress_from_1d(v0, meta)
        TT = next(TT_gen)
        TT_start = TT.swap_gate(axes=(2, (0, 3)))
        V = TT_start.tensordot(V, ([2], [0]))
        #
        #       | \
        #   1--TT |  |-----|
        #    /-|--|--|     |
        #      2  0  |  V  |
        #    3 ------|     |
        #   ...      |     |
        #  Ly+1 -----|-----|

        for i in range(1, env.psi.dims[1]*Ly - 1):
            #
            #       | \                                  | \
            #   1--TT | |-----|                      3--TT |   |-----|
            #    /-|--|-|     |                       /-|--|---|     |
            #	   |  | |	  |							   |   |	 |
            #   2--TT | |     |                      4--TT |   |     |
            #    /-|--|-|     |                       /-|--|---|     |
            #     ... | |     |        --->          ...   |   |     |
            #   i-TT  | |  V  |                    0 --TT  |   |  V  |
            #   /-|---|-|     |                       /-|--|---|     |
            #    i+1  0 |     |                         1  2   |     |
            #    i+2 ---|     |                       i+3 -----|     |
            #             ...                               	 ...
            #   Ly+1 ---|-----|                      Ly+1 -----|-----|
            TT = next(TT_gen)
            TT_middle = TT.swap_gate(axes=(2,3))
            V = V.swap_gate(axes=(0, i+2))
            V = TT_middle.tensordot(V, ([0, 2], [i + 1, i + 2]))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, env.psi.dims[1]*Ly + 2))
            V = V.transpose(order)


        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        TT = next(TT_gen)
        V = V.tensordot(TT, ([env.psi.dims[1]*Ly, env.psi.dims[1]*Ly + 1, 0], [0, 2, 3]))

        TB = next(TB_gen)
        TB_start = TB.swap_gate(axes=(2,(0, 3)))
        V = TB_start.tensordot(V, ([1], [0]))
        for i in range(1, env.psi.dims[1]*Ly - 1):
            # TL action (contract TL top with V)
            #
            #    / \                            / \
            #    | TB---|-----|                 | TB---|-----|
            #  1-|-|-\  |     |               3-|-|-\  |     |
            #	 | |    |	  |				    | |    |	 |
            #    | TB---|     |                 | TB---|     |
            #  2-|-|-\	|     |               4-|-|-\  |     |
            #    |...   |     |        --->     |...   |     |
            #    | TB---|  V  |                 | TB---|  V  |
            #  i-|-|-\  |     |               0-|-|-\  |     |
            #    0 i+1  |     |                 2 1    |     |
            #    i+2 ---|     |                 i+3 ---|     |
            #             ...                            ...
            # L*Ly+1 ---|-----|              L*Ly+1 ---|-----|
            TB = next(TB_gen)
            TB_middle = TB.swap_gate(axes=(2,3))
            V = TB_middle.tensordot(V, ([0, 1], [i + 1, i + 2]))
            V = V.swap_gate(axes=(0,2))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, env.psi.dims[1]*Ly + 2))
            V = V.transpose(order)

        # add sigma_z to leg 0:
        # V = V.swap_gate(axes=(0,0))
        TB = next(TB_gen)
        V = V.tensordot(TB, ([env.psi.dims[1]*Ly,env.psi.dims[1]*Ly + 1, 0], [0, 1, 3]))
        v1, _ = yastn.Tensor.compress_to_1d(V, meta)
        return v1

    legs = [next(TT_gen).get_legs(axes=1) for _ in range(env.psi.dims[1]*Ly)]
    v0 = yastn.rand(config=yastn_config, n=charge_sector, legs=legs)
    data, meta = yastn.Tensor.compress_to_1d(v0)

    sigma_T_sigma_B = LinearOperator(
        (len(data), len(data)),
        matvec=partial(mv, meta=meta) ,
        dtype="complex128" if TTs[0].is_complex() else "float64",
    )

    vals, evecs = eigs(
        sigma_T_sigma_B, k=num_evals, v0=data, which="LM", return_eigenvectors=True
    )

    return vals, evecs, meta

def analyze_momentum_sector_abelian(vals, evecs, meta, Ly, k_shift=0):
    def translation(V):
        # With sigma_z on Ly-1
        for i in range(Ly):

        # Without sigma_z on Ly-1
        # for i in range(Ly-1):
            V = V.swap_gate(axes=(Ly-1, i))

        translateddims = [(i - 1) % Ly for i in range(Ly)]
        V = V.transpose(translateddims)
        return V

    evals = {}
    for i, val in enumerate(vals):
        cnt = 0
        v0 = torch.as_tensor(evecs[:, i])
        V = yastn.decompress_from_1d(v0, meta)
        # print((translation(V).to_dense()-np.exp()V.to_dense()))
        for k in range(Ly):
            k = k+k_shift
            if (
                yastn.linalg.norm(
                    translation(V)
                    - np.exp(1j * 2 * np.pi * (k) / Ly) * V
                )
                <= 1e-6
            ):
                if k not in evals:
                    evals[k] = [val]
                else:
                    evals[k].append(val)
                # break
                cnt += 1
        assert cnt == 1
    return evals

def analyze_momentum_sector_unit_L(vals, evecs, meta, unit_L, Lx, k_shift=0):
    # unit_L: linear length of the unit-cell; Lx: width of the cylinder
    def translation(V):
        # With sigma_z on Lx-1
        for i in range(Lx):

        # Without sigma_z on Lx-1
        # for i in range(Lx-1):
            V = V.swap_gate(axes=(Lx-1, i))

        translateddims = [(i - 1) % Lx for i in range(Lx)]
        V = V.transpose(translateddims)
        return V

    evals = {}
    for i, val in enumerate(vals):
        cnt = 0
        v0 = torch.as_tensor(evecs[:, i])
        V = yastn.decompress_from_1d(v0, meta)
        # fuse legs within a unit-cell
        fuse_order = tuple(tuple(range(i, i + unit_L)) for i in range(0, Lx*unit_L, unit_L))
        V = V.fuse_legs(axes=fuse_order)
        # print((translation(V).to_dense()-np.exp()V.to_dense()))
        for k in range(Lx):
            k = k+k_shift
            if (
                yastn.linalg.norm(
                    translation(V)
                    - np.exp(1j * 2 * np.pi * (k) / Lx) * V
                )
                <= 1e-6
            ):
                if k not in evals:
                    evals[k] = [val]
                else:
                    evals[k].append(val)
                # break
                cnt += 1
        assert cnt == 1
    return evals

def plot_es(es, color, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for k in es.keys():
        print(k, -np.log(np.abs(es[k])))
        ax.plot(k * np.ones(len(es[k])), -np.log(np.abs(es[k])), lw=0, marker="o", color=color)
    return ax

def load_env_from_dict(env, d, yastn_config):
    assert d['class'] == 'EnvCTM'
    for site in d['data']:
        for dirn in d['data'][site]:
            setattr(env[site], dirn, yastn.load_from_dict(yastn_config, d['data'][site][dirn]))

    return env



if __name__ == "__main__":
    Lx, Ly = 4, 4
    tot_D = 4
    tot_chi = 48

    state_file = (
        f"CI_grad/CI_2x2_SU_init_D_6_chi_48_state.json"
    )

    yastn_config = yastn.make_config(
        backend=backend,
        # sym=sym_Z2,
        sym=sym_U1,
        fermionic=True,
        default_device="cpu",
        default_dtype="complex128",
    )
    omp_cores = 16
    torch.set_num_threads(omp_cores)

    # Converge environment
    opts_svd = {
        "D_total": tot_chi,
        "tol": 1e-8,
        "eps_multiplet": 1e-8,
        "fix_signs": True,
        "truncate_multiplets": True,
    }

    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(tot_chi,))
    env_dict_filename = "CI_grad/env_CI_fp_2x2_D_6_U1_chi_48_V_0.00_t1_1.00_t2_1.00_t3_0.00_phi_0.35"


    _tmp_config = {x: y for x, y in yastn_config._asdict().items() if x != "sym"}
    sf = yastn.operators.SpinfulFermions(sym=str(yastn_config.sym), **_tmp_config)
    n_A = sf.n(spin="u")  # parity-even operator, no swap gate needed
    n_B = sf.n(spin="d")
    c_A = sf.c(spin="u")
    cp_A = sf.cp(spin="u")
    c_B = sf.c(spin="d")
    cp_B = sf.cp(spin="d")
    I = sf.I()

    dist = 100
    torch.set_printoptions(precision=10)

    # ci_A, cjp_A = op_order(c_A, cp_A, ordered=True, fermionic=True)
    # cip_A, cj_B = op_order(cp_A, c_B, ordered=True, fermionic=True)


    # corrf = corr(state, env, state.sites()[0], (1, 0), ci_A, cjp_A, dist)
    # print(corrf.abs())
    # plt.style.use("science")
    # plt.plot(abs(corrf), lw=0, marker = 'o', markersize=3)
    # plt.yscale('log')
    # plt.ylabel(r"$|\langle c_{00, A} c^\dagger_{r0, A}\rangle|$")
    # plt.xlabel(r"$r$")
    # plt.show()
    # plt.savefig("figs/cA_cpA_corrf.pdf")

    # vals, evecs = ent_spec_cylinder(state, env, Ly, tot_chi, n=50)
    # es = analyze_momentum_sector(vals, evecs, tot_D, Ly)
    ax = None
    rerun = False
    colors = {-1: 'k', 0: 'b', 1: 'red'}
    for i, n in enumerate([0]):
        filename = f"CI_grad/es/es_CI_0.35pi_Ly_{Ly:d}_D_{tot_D:d}_U1_chi_{tot_chi:d}_n_{n:d}_APBC"
        if os.path.exists(filename) and not rerun:
            with open(filename, "rb") as handle:
                es = pickle.load(handle)
        else:
            state = load_PepsAD(yastn_config, state_file)
            rerun = False
            if os.path.exists(env_dict_filename) and not rerun:
                with open(env_dict_filename, "rb") as handle:
                    d = pickle.load(handle)
                env = EnvCTM(state, init="eye", leg=env_leg)
                env = load_env_from_dict(env, d,  yastn_config)
                print("Loaded")
            else:
                ctm_env_in = EnvCTM(state, init="eye", leg=env_leg)
                # with open(env_dict_filename, "rb") as handle:
                # 	d = pickle.load(handle)
                # ctm_env_in = load_env_from_dict(ctm_env_in, d,  yastn_config)
                # print("Loaded")
                max_sweeps=1500
                env, converged, *ctm_log = get_converged_env(
                    ctm_env_in,
                    max_sweeps=max_sweeps,
                    iterator_step=1,
                    opts_svd=opts_svd,
                    corner_tol=1e-8,
                    # corner_tol=1e-11,
                )
                env_dict = env.save_to_dict()
                with open(env_dict_filename, "wb") as handle:
                    pickle.dump(env_dict, handle)
            # vals, evecs, meta = ent_spec_cylinder_Lx(yastn_config, env, Lx, env.sites()[0], charge_sector=n, num_evals=70)
            vals, evecs, meta = ent_spec_cylinder_Ly(yastn_config, env, Ly, env.sites()[0], charge_sector=n, num_evals=70)
            print(vals)
            # k_shift = 0 if n%2 ==0 else 0.5
            # es = analyze_momentum_sector_abelian(vals, evecs, meta, Ly, k_shift=k_shift)
            k_shift = 0
            es = analyze_momentum_sector_unit_L(vals, evecs, meta, env.psi.dims[1], Ly, k_shift=k_shift)
            with open(filename, "wb") as handle:
                pickle.dump(es, handle)
        # ax = plot_es(es, color=colors[i], ax=ax)
        ax = plot_es(es, color='blue', ax=None)
        # plt.savefig(f"es/es_Ly_{Ly:d}_D_{tot_D:d}_U1_chi_{tot_chi:d}_n_{n:d}_APBC.pdf")
        plt.savefig(f"CI_grad/es/es_CI_0.35pi_Ly_{Ly:d}_D_{tot_D:d}_U1_chi_{tot_chi:d}_n_{n:d}_APBC.pdf")

    # ks = np.linspace(0, 4.5, 100)
    # lower_line = 1.3*ks
    # ax.plot(ks, lower_line-8.8, c='k', ls="--")

    # upper_line = 1.3*ks
    # ax.plot(ks, lower_line-3.7, c='k', ls="--")
    # plt.savefig(f"es/es_Ly_{Ly:d}_D_{tot_D:d}_U1_chi_{tot_chi:d}_PBC.pdf")
    # plt.show()

    # for dirn in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
    #   # vals = transfer_spec_TAT(state, env, state.sites()[0], dirn, tot_D, tot_chi, n=30, eigenvectors=False)
    #   spec_file = "TAT_spec_({dirn[0]:d}, {dirn[1]:d})_tV_1x1_D_2+2_odd_chi_40_V_0.00_t1_0.50_t2_0.35_t3_-0.45_mu_-0.90"
    #   # with open(spec_file, "wb") as handle:
    #   #   pickle.dump(vals, handle)
    #   with open(spec_file, "rb") as handle:
    #       vals = pickle.load(handle)
    #   eigs = np.zeros(vals.size(dim=0), dtype=np.complex128)
    #   eigs = vals[:, 0] + 1j*vals[:, 1]
    #   plt.plot(np.ones(len(eigs)), np.abs(eigs), lw=0, marker='o')
    #   plt.show()
