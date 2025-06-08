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


def apply_TM_TAT(state, env, site, dirn, V, op=None):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param site: tuple (x,y) specifying vertex on a square lattice
    :param dirn: direction in which the transfer operator is applied
    :param V: tensor of dimensions :math:`\chi \times D^2 \times \chi (\times d_{aux})`,
                 potentially with 4th auxiliary index
    :param op: operator to be inserted into transfer matrix

    :type state: PepsAD
    :type env: yastn.fn.fpeps.EnvCTM
    :type site: yastn.tn.fpeps.Site
    :type dirn: tuple(int,int)
    :type edge: yastn.Tensor
    :type op: yastn.Tensor
    :return: Resulting tensor from applying the transfer matrix applied to V.
             The tensor either has an identical index structure as the original V
             or has an additional auxiliary index from op.
    :rtype: yastn.tensor
    """
    def get_dl_tensor(op, dirn):
        # Forming double tensor
        A_top, A_bot = state[site].unfuse_legs(axes=(0, 1)), state[site].unfuse_legs(axes=(0, 1))
        A_bot = A_bot.swap_gate(axes=(0, 1, 2, 3)) # t' x l', b' x r'
        if op is None: # identity operator
            dl_tensor = A_top.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r t' l' b' r'
            dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7))) # [t t'] [l l'] [b b'] [r r']
            #   \ \
            # --|--A--------
            #   |  | \
            #   |  |  \                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \
        else:
            dims_op = op.get_shape()
            if len(dims_op) == 2: # no aux index
                # check if a dummy leg is fused with the physical leg
                leg = A_top.get_legs(axes=4)
                if leg.is_fused():
                    A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux

                    dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p
                    dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3,(5,4))) # t l b r [p p_aux]
                else:
                    dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p

                dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4))
                dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
                dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7))) # [t t'] [l l'] [b b'] [r r']
                #   \ \
                # --|--A--------
                #   |  | \
                #   |  O  \                     \       \     / \
                #    \ |   \                ----Ah---= --\---Ac--\---
                # -----Ah---\---                 \        \ /     \
                #       \    \
            elif len(dims_op) == 3: #  op has an extra index to make it charge-neutrual
                if dirn in [(1, 0), (0, 1), (0, -1)]:
                    # check if a dummy leg is fused with the physical leg
                    leg = A_top.get_legs(axes=4)
                    if leg.is_fused():
                        A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                        dl_tensor = dl_tensor.swap_gate(axes=(4, 6))
                        dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6)) # t l b r [p p_aux] aux
                    else:
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux


                    dl_tensor = dl_tensor.swap_gate(axes=(5, (2, 3))) # aux x b r
                    dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
                    dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
                    dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
                    dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
                    #
                    #   \ \        ____ (aux)
                    # --|--A-----/---
                    #   |  | \ /
                    #   |  O-/\                     \       \     / \
                    #    \ |   \                ----Ah---= --\---Ac--\---
                    # -----Ah---\---                 \        \ /     \
                    #       \    \
                elif dirn in [(-1, 0)]:
                    # check if a dummy leg is fused with the physical leg
                    leg = A_top.get_legs(axes=4)
                    if leg.is_fused():
                        A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                        dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6))
                    else:
                        dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

                    dl_tensor = dl_tensor.swap_gate(axes=(5, 1)) # aux x l
                    dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
                    dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
                    dl_tensor = dl_tensor.swap_gate(axes=(4, (1, 8), 2, 7)) # t' x [l, aux], b x r'
                    dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
                    # aux
                    #  \ \ \
                    # -|-|--A--------
                    # | |  | \
                    # --|--O  \                     \       \     / \
                    #    \ |   \                ----Ah---= --\---Ac--\---
                    # -----Ah---\---                 \        \ /     \
                    #       \    \

        return dl_tensor

    dl_tensor = get_dl_tensor(op, dirn)
    dl_dim = len(dl_tensor.get_shape())
    V_dim = len(V.get_shape())
    if dirn == (1, 0):
        # right action
        #               ___aux
        # ---0  --T_t-/---
        # |       | /
        # V--1  --A --           if dl_dim == 5 and V_dim == 3
        # |       |
        # ---2 --T_b--
        # Or
        # ---(aux)
        # ---0  --T_t--
        # |       |
        # V--1  --A --           if dl_dim == 4 and V_dim == 3 or 4
        # |       |
        # ---2 --T_b--

        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     |                    |
        #     1               2---T_b---0

        res = V.tensordot(T_t, (0, 0))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        # ------------- (4)
        # --------T_t----3
        # |       |
        # V--0    2
        # |
        # ---1

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 2], [1, 0]))
            if V_dim == 4:
                res = res.transpose(axes=(0,1,3,4,2))
            # -----------(4)
            #  -------T_t---1
            # |       |
            # V-------A----- 3
            # |       |
            # ---0    2
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0, 2], [1, 0]))
            res = res.swap_gate(axes=(1, 4))
            #               -----4
            #  -------T_t-/--1
            # |       | /
            # V-------A----- 3
            # |       |
            # ---0   2


        res = res.tensordot(T_b, ([0, 2], [2, 1]))
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0,1,3,2))
        # -------(3)
        #  -------T_t---0
        # |       |
        # V-------A-----1
        # |       |
        #  -------T_b---2

    elif dirn == (-1, 0):
        # left action
        # aux --\
        #      --\-T_t--   0--
        #         \ |        |
        #      ----A---   1--V      if dl_dim == 5 and V_dim == 3
        #          |        |
        #      ---T_b--  2--
        # Or
        #       (aux)---
        # --T_t--   0---
        #   |         |
        # --A----  1--V      if dl_dim == 4 and V_dim == 3 or 4
        #   |         |
        # --T_b--  2--
        #

        T_t, T_b = env[site].t.to_dense(), env[site].b.to_dense()
        # 0 --T_t -- 2              1
        #     ||                   ||
        #     1               2---T_b---0

        res = V.tensordot(T_t, (0, 2))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        #     (4) ---
        # 2--T_t-----
        #    |      |
        #    3 0----V
        #           |
        #      1----

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 3], [3, 0]))
            if V_dim == 4:
                res = res.transpose(axes=(0, 1, 3, 4, 2))
            #    (4)---
            # 1--T_t---
            #    |    |
            # 2--A----V
            #    |    |
            #    3 0--
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0, 3], [3, 0]))
            res = res.swap_gate(axes=(1, 4))
            #  4
            # 1-\--T_t---
            #    \ |    |
            #   2--A----V
            #      |    |
            #      3 0--

        res = res.tensordot( T_b, ([0, 3], [0, 1]))
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0,1,3,2))
        # (3)--------
        # 0--T_t----
        #    |     |
        # 1--A-----V
        #    |     |
        # 2--T_b----
    elif dirn == (0, -1):
        # up action
        # \  \  \/ aux
        #  \  \/ \
        # T_l-A- T_r       if dl_dim == 5 and V_dim == 3
        #   0   1   2
        #    \   \  \
        #    ----V----
        # Or
        #  \  \  \   \ (aux)
        # T_l--A--T_r \      if dl_dim == 4 and V_dim == 3 or 4
        #   0   1  2   \
        #    \   \  \   \
        #      ----V------

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [0]))
        if V_dim == 4:
            res = res.transpose(axes=(0, 1, 3, 4, 2))
        # 3
        #  \
        # T_l-- 2
        #   \   0   1   (4)
        #    \   \   \   \
        #     ------V------

        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 2], [2, 1]))
            if V_dim == 4:
                res = res.transpose(axes=(0, 1, 3, 4, 2))
            # 1   2
            #  \   \
            # T_l---A--3
            #   \   \   0   (4)
            #    \   \   \   \
            #     ------V------
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, axes=([0,2], [2, 1]))

            # 1   2    4
            #  \   \ /
            # T_l---A--3
            #   \   \    0
            #    \   \    \
            #     ----V-----
        res = res.tensordot(T_r, ([0, 3], [2, 1]))
        if dl_dim == 5:
            res = res.swap_gate(axes=(2, 3))
            #         3   2(aux)
            # 0   1    \/
            #  \   \  / \
            # T_l---A---TR
            #   \   \    \
            #    \   \    \
            #     ----V-----
        if len(res.get_shape()) == 4:
            res = res.transpose(axes=(0, 1, 3, 2))
        # 0   1    2
        #  \   \    \
        # T_l---A---TR
        #   \   \    \ \ (aux)
        #    \   \    \ \
        #     ----V------

    elif dirn == (0, 1):
        # down action
        # ---V---
        #  \  \  \ /---
        #   \  \ /\    \
        #  T_l--A--T_r  \          if dl_dim == 5 and V_dim == 3
        #     \  \  \    \ aux
        # Or
        # ---V---------
        #  \  \  \    \
        #  T_l--A--T_r \          if dl_dim == 4 and V_dim == 3 or 4
        #    \  \  \    \ (aux)

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [2]))
        if V_dim == 4:
            res = res.transpose(axes=(0,1,3,4,2))
        # ------V---------
        #  \     \    \   \
        #  T_l-3  0    1   \
        #    \              \ (4)
        #     2
        if dl_dim == 4:
            res = res.tensordot(dl_tensor, ([0, 3], [0, 1]))
            if V_dim == 4:
                res = res.transpose(axes=(0,1,3,4,2))
            #  ------V---------
            #  \     \      \   \
            #  T_l----A-- 3  0   \
            #    \     \          \ (4)
            #     1    2
        else:
            assert dl_dim == 5 and V_dim == 3
            res = res.tensordot(dl_tensor, ([0,3], [0,1]))
            res = res.swap_gate(axes=(0, 4))
            #  ------V-------
            #  \     \ /----\---
            #  T_l----A-- 3  \  \
            #    \     \      \  \
            #     1     2      0  4

        res = res.tensordot(T_r, ([0, 3], [0, 1]))
        if (len(res.get_shape())) == 4:
            res = res.transpose(axes=(0,1,3,2))

        # ---V---------
        #  \  \  \    \
        #  T_l--A--T_r \
        #    \  \  \    \ (aux)
    return res

def get_edge(state, env, site, dirn):
    """
    Build an edge of site ``coord`` by contracting one of the following networks
    depending on the chosen ``direction``::

            up=(0,-1)   left=(-1,0)  down=(0,1)   right=(1,0)

                         C--0         0  1  2       0--C
                         |            |  |  |          |
        E =  C--T--C     T--1         C--T--C       1--T
             |  |  |     |                             |
             0  1  2     C--2                       2--C

    """
    if dirn == (0, -1):
        C1, C2 = env[site].tl, env[site].tr
        T = env[site].t
        res = C1.tensordot(T, axes=(1, 0))
        # C1--T--2
        # |   |
        # 0   1
        res = res.tensordot(C2, axes=(2, 0))
        # C1--T--C2
        # |   |  |
        # 0   1  2
    elif dirn == (-1, 0):
        C1, C2 = env[site].tl, env[site].bl
        T = env[site].l
        res = C1.tensordot(T, axes=(0, 2))
        # C1--0
        # |
        # T--2
        # |
        # 1
        res = res.tensordot(C2, axes=(1, 1))
        # C1--0
        # |
        # T--1
        # |
        # C2--2
    elif dirn == (0, 1):
        C1, C2 = env[site].bl, env[site].br
        T = env[site].b
        res = C1.tensordot(T, axes=(0, 2))
        #  0   2
        #  |   |
        # C1 --T-- 1
        res = res.tensordot(C2, axes=(1, 1))
        #  0   1    2
        #  |   |    |
        # C1 --T-- C2
    elif dirn == (1, 0):
        C1, C2 = env[site].tr, env[site].br
        T = env[site].r
        res = C1.tensordot(T, axes=(1, 0))
        # 0 --C1
        #     |
        # 1 --T
        #     |
        #     2
        res = res.tensordot(C2, axes=(2, 0))
        # 0 --C1
        #     |
        # 1 --T
        #     |
        # 2 --C2
    return res

def apply_TM_TAT_contract_aux(state, env, site, dirn, V, op):
    r"""
    :param state: underlying wavefunction
    :param env: environment corresponding to ``state``
    :param site: tuple (x,y) specifying vertex on a square lattice
    :param dirn: direction in which the transfer operator is applied
    :param V: tensor of dimensions :math:`\chi \times D^2 \times \chi \times d_{aux}`
    :param op: operator to be inserted into transfer matrix, with an additional aux. index

    :type state: PepsAD
    :type env: yastn.fn.fpeps.EnvCTM
    :type site: yastn.tn.fpeps.Site
    :type dirn: tuple(int,int)
    :type edge: yastn.Tensor
    :type op: yastn.Tensor
    :return: Resulting tensor from applying the transfer matrix applied to V.
             The aux. index of tensor is contracted with the aux. index of op.
    :rtype: yastn.tensor
    """


    def get_dl_tensor(op, dirn):
        A_top, A_bot = state[site].unfuse_legs(axes=(0, 1)), state[site].unfuse_legs(axes=(0, 1))
        A_bot = A_bot.swap_gate(axes=(0, 1, 2, 3)) # t' x l', b' x r'
        dims_op = op.get_shape()
        assert len(dims_op) == 3 # extra index to make op charge-neutrual
        if dirn in [(-1, 0), (0, -1), (0, 1)]:
            leg = A_top.get_legs(axes=4)
            if leg.is_fused():
                A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                dl_tensor = dl_tensor.swap_gate(axes=(4, 6))
                dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6)) # t l b r [p p_aux] aux
            else:
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

            dl_tensor = dl_tensor.swap_gate(axes=(5, (2, 3))) # aux x b r
            dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
            dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
            dl_tensor = dl_tensor.swap_gate(axes=(1, 4, 2, 7)) # l x t', b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
            #
            #   \ \        ____ (aux)
            # --|--A-----/---
            #   |  | \ /
            #   |  O-/\                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \

        elif dirn in [(1, 0)]:
            leg = A_top.get_legs(axes=4)
            if leg.is_fused():
                A_top = A_top.unfuse_legs(axes=4) # t l b r p p_aux
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p_aux p aux
                dl_tensor = dl_tensor.fuse_legs(axes=(0,1,2,3, (5,4), 6))
            else:
                dl_tensor = A_top.tensordot(op, axes=(4, 1)) # t l b r p aux

            dl_tensor = dl_tensor.swap_gate(axes=(5, 1)) # aux x l
            dl_tensor = dl_tensor.tensordot(A_bot.conj(), axes=(4, 4)) # t l b r aux t' l' b' r'
            dl_tensor = dl_tensor.transpose(axes=(0, 1, 2, 3, 5, 6, 7, 8, 4)) # t l b r t' l' b' r' aux
            dl_tensor = dl_tensor.swap_gate(axes=(4, (1, 8), 2, 7)) # t' x [l, aux], b x r'
            dl_tensor = dl_tensor.fuse_legs(axes=((0, 4), (1, 5), (2, 6), (3, 7), 8)) # [t t'] [l l'] [b b'] [r r'] aux
            # aux
            #  \ \ \
            # -|-|--A--------
            # | |  | \
            # --|--O  \                     \       \     / \
            #    \ |   \                ----Ah---= --\---Ac--\---
            # -----Ah---\---                 \        \ /     \
            #       \    \

        return dl_tensor

    dl_tensor = get_dl_tensor(op, dirn)
    assert len(V.get_shape()) == 4
    if dirn == (1, 0):
        # right action
        # ----
        #  ----\--T_t---0
        # |     \ |
        # V-------A-----1
        # |       |
        #  -------T_b---2
        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     |                    |
        #     1               2---T_b---0

        V = V.swap_gate(axes=(3, 0))
        # ------
        # -----\---0
        # |     (3)
        # V--1
        # |
        # ---2
        res = V.tensordot(T_t, (0, 0))
        res = res.transpose(axes=(0, 1, 3, 4, 2))
        # ----
        # ----\---T_t----3
        # |   (4) |
        # V--0    2
        # |
        # ---1

        res = res.tensordot(dl_tensor, ([0, 2, 4], [1, 0, 4]))
        # contract the auxiliary indices
        #  ----
        #  ----\-T_t---1
        # |     \ |
        # V-------A-----3
        # |       |
        # ---0    2

        res = res.tensordot(T_b, ([0, 2], [2, 1]))
        # ----
        #  ----\--T_t---0
        # |     \ |
        # V-------A-----1
        # |       |
        #  -------T_b---2
    elif dirn == (-1, 0):
        # left action
        #          --
        # 0--T_t-/--
        #    | /   |
        # 1--A-----V
        #    |     |
        # 2--T_b----
        T_t, T_b = env[site].t, env[site].b
        # 0 --T_t -- 2              1
        #     ||                   ||
        #     1               2---T_b---0

        V = V.swap_gate(axes=(3, 0))
        #      --
        #  0-/--
        # (3)  |
        #  1---V
        #  2---|
        #
        res = V.tensordot(T_t, (0, 2))
        res = res.transpose(axes=(0, 1, 3, 4, 2))
        #           __
        # 2--T_t--/--
        #    | (4)  |
        #    3 0----V
        #           |
        #      1----

        res = res.tensordot(dl_tensor, ([0, 3, 4], [3, 0, 4]))
        #           __
        # 1--T_t--/---
        #    |  /    |
        # 2--A------ V
        #    |       |
        #    3  0----

        res = res.tensordot(T_b, ([0, 3], [0, 1]))
        #          --
        # 0--T_t-/--
        #    | /   |
        # 1--A-----V
        #    |     |
        # 2--T_b----

    elif dirn == (0, -1):
        # up action
        # \  \   \/\
        #  \  \ / \ \
        #  T_l-A-T_r \
        #   \0  \1 \2 \(aux)
        #    ---V------
        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  T_l-1          1-T_r
        #  |               |
        #  0               2
        res = V.tensordot(T_r, ([2], [2]))
        res = res.swap_gate(axes=(2,3))
        #      3
        #       \ /\
        #       /\  \
        #  (aux)2 \  \
        #     1 4-T_r \
        #  0   \   \   \
        #   \   \   \   \
        #   ----V---------
        res = res.tensordot(dl_tensor, ([1, 4, 2], [2, 3, 4]))
        #       1
        #     2 \ /\
        #     \ |\  \
        #      \| \  \
        #   3--A--T_r \
        #  0   \   \   \
        #   \   \   \   \
        #   ----V---------
        res = res.tensordot(T_l, ([0, 3], [0, 1]))
        res = res.transpose((2,1,0))
        # 0  1   2
        # \  \   \/\
        #  \  \ / \ \
        #  T_l-A-T_r \
        #   \  \  \   \(aux)
        #    ---V------
    elif dirn == (0, 1):
        # down action
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A---T_r
        #   \  \    \

        T_l, T_r = env[site].l, env[site].r
        #  2                0
        #  |                |
        #  TL-1          1-TR
        #  |               |
        #  0               2
        res = V.tensordot(T_l, ([0], [2]))
        # ----V--------
        # \    \   \  \
        # \     0   1  2
        # T_l-4
        # \
        # 3
        res = res.swap_gate(axes=(1,2))
        # ----V--------
        # \    \   \  \
        # \     0   \ /
        # T_l-4     /\
        # \       2   1
        # 3

        res = res.tensordot(dl_tensor, ([0, 4, 2], [0, 1, 4]))
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A-3 0
        #   \  \
        #   1   2

        res = res.tensordot(T_r, ([0, 3], [0, 1]))
        # -----V----\
        # \  \   \ __\(aux)
        #  \  \ / \
        #  T_l-A---T_r
        #   \  \    \
        #    0   1   2
    return res


def corr(state, env, site, dirn, op1, op2, dist):
    c0 = site
    rev_dirn = (-dirn[0], -dirn[1])
    E0 = get_edge(state, env, site, rev_dirn)

    E_1 = apply_TM_TAT(state, env, c0, dirn, E0, op=op1)
    E_N =  apply_TM_TAT(state, env, c0, dirn, E0, op=None)

    corrf = torch.empty(dist+1, dtype=torch.complex128, device=state.device)

    for r in range(dist+1):
        #
        #       C--T--- [ --T-- ]^r --T---
        # E12 = T--O1-- [ --A-- ]   --O2--
        #       C--T--- [ --T-- ]   --T---
        c0 = state.nn_site(c0, dirn)
        if len(op2.get_shape()) == 3:
            E_12 = apply_TM_TAT_contract_aux(state, env, c0, dirn, E_1, op=op2)
        else:
            E_12 = apply_TM_TAT(state, env, c0, dirn, E_1, op=op2)
        E_1 = apply_TM_TAT(state, env, c0, dirn, E_1, op=None)
        E_N = apply_TM_TAT(state, env, c0, dirn, E_N, op=None)

        E_end = get_edge(state, env, c0, dirn)
        val_12 = E_12.tensordot(E_end, axes=((0,1,2), (0,1,2)))
        val_norm = E_N.tensordot(E_end, axes=((0,1,2), (0,1,2)))
        corrf[r] = val_12.to_number()/val_norm.to_number()

        # normalize by largest element of E_N
        max_elem_EN = yastn.linalg.norm(abs(E_N), p='inf')
        E_N=E_N/max_elem_EN
        E_1=E_1/max_elem_EN

    return corrf



def transfer_spec_TAT(state, env, site, dirn, tot_D, tot_chi, n=10, eigenvectors=False):
    r"""
    Compute the leading `n` eigenvalues of width-0 transfer operator of IPEPS::

            --T---------...--T--------            --\               /---
            --A(x,y)----...--A(x+lX,y)-- = \sum_i ---v_i \lambda_i v_i--
            --T---------...--T--------            --/               \---

    where `A` is a double-layer tensor. The transfer matrix is given by width-1 channel
    of the same length lX as the unit cell of iPEPS, embedded in environment of T-tensors.

    Other directions are obtained by analogous construction.
    """

    # depending on the direction, get unit-cell length
    if dirn == (1, 0) or dirn == (-1, 0):
        N = state.Nx
    elif dirn == (0, 1) or dirn == (0, -1):
        N = state.Ny
    else:
        raise ValueError("Invalid direction: " + str(dirn))

    V_shape = (tot_chi, tot_D * tot_D, tot_chi)
    def mv(v0):
        c0 = site
        V = torch.as_tensor(v0, device=state[site].device)
        V = V.view(V_shape)
        for i in range(N):
            V = apply_TM_TAT(state, env, c0, dirn, V)
            c0 = state.nn_site(c0, dirn)
        V = V.view(np.prod(V_shape))
        v = V.cpu().numpy()
        return v

    T = LinearOperator(
        (np.prod(V_shape), np.prod(V_shape)),
        matvec=mv,
        dtype="complex128" if state[site].is_complex() else "float64",
    )
    if eigenvectors:
        vals, vecs = eigs(T, k=n, v0=None, return_eigenvectors=True)
    else:
        vals = eigs(T, k=n, v0=None, return_eigenvectors=False)

    # post-process and return as torch tensor with first and second column
    # containing real and imaginary parts respectively
    # sort by abs value in ascending order, then reverse order to descending
    ind_sorted = np.argsort(np.abs(vals))[::-1]
    vals = vals[ind_sorted]
    # vals= np.copy(vals[::-1]) # descending order
    vals = (1.0 / np.abs(vals[0])) * vals
    L = torch.zeros((n, 2), dtype=torch.float64, device=state.device)
    L[:, 0] = torch.as_tensor(np.real(vals))
    L[:, 1] = torch.as_tensor(np.imag(vals))

    if eigenvectors:
        return L, torch.as_tensor(vecs[:, ind_sorted], device=state.device)
    return L


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
