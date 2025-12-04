import os, sys, json, pickle, argparse
from itertools import cycle
from functools import partial
sys.path.insert(0, os.path.abspath('..'))

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigs

import yastn.yastn as yastn
from yastn.yastn.sym import sym_Z2, sym_U1, sym_none
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tn.fpeps import product_peps, RectangularUnitcell, Peps
from yastn.yastn.tn.fpeps.envs.fixed_pt import refill_env, fp_ctmrg
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn import load_from_dict
from ipeps.integration_yastn import PepsAD, load_PepsAD


def load_env_from_dict(env, d, yastn_config):
    assert d['class'] == 'EnvCTM'
    for site in d['data']:
        for dirn in d['data'][site]:
            setattr(env[site], dirn, yastn.load_from_dict(yastn_config, d['data'][site][dirn]))

    return env

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

def Ts_to_MPO_list(env, start_site=0, dirn="x"):
    if dirn == "x":
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

        #  0   __ 1 (top)    (top) 1__   0
        #  | /                        \  |
        #  TL----2 (bot)     (bot) 2----TR
        #  |                            |
        #  3                            3

        TR_cycle = [None]*len(TRs)
        for i in range(0, env.psi.dims[0]):
            TR_middle = TRs[i % env.psi.dims[0]]
            TR_middle = TR_middle.swap_gate(axes=(2, 3))
            TR_cycle[i] = TR_middle.transpose(axes=(0, 1, 3, 2))
            #       0
            #       |
            #	    |
            #    1--TR
            #     /-|---3
            #       |
            #       2

        TL_cycle = [None]*len(TLs)
        for i in range(0, env.psi.dims[0]):
            TL_middle = TLs[i % env.psi.dims[0]]
            TL_middle = TL_middle.swap_gate(axes=(2, 3))
            TL_cycle[i] = TL_middle.transpose(axes=(0, 2, 3, 1))
            #      0
            #	   |
            #      TL---3
            #  1---|-\
            #      2

        return TR_cycle, TL_cycle

def ent_spec_cylinder_Lx(yastn_config, TL_gen, TR_gen, Lx, L_unit, charge_sector=0, num_evals=8, APBC=True):
    # Compute the entanglement spectrum of PEPS ansatz for a cut specified by the start_site
    # Lx: width of the cylinder, measure in unit-cells (finite width along x-direction)
    # start_site: the site whose left-bond is crossed by the entanglement cut

    def mv(v0, meta):
        # V: YASTN Tensor
        # TR action
        TR = next(TR_gen)
        v0 = torch.as_tensor(v0, dtype=TR.dtype, device=TR.device)
        V = yastn.decompress_from_1d(v0, meta)
        TR_start = TR.swap_gate(axes=(0, 3))
        V = TR_start.tensordot(V, ([3], [0]))

        #
        #       | \
        #   1--TR |  |-----|
        #    /-|--|--|     |
        #      2  0  |  V  |
        #    3 ------|     |
        #   ...      |     |
        #  Lx+1 -----|-----|

        for i in range(1, L_unit*Lx - 1):
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
            V = V.swap_gate(axes=(0, i+2))
            V = TR.tensordot(V, ([0, 3], [i + 1, i + 2]))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, L_unit*Lx + 2))
            V = V.transpose(order)

        if not APBC:
            # add sigma_z to leg 0:
            V = V.swap_gate(axes=(0,0))
        TR = next(TR_gen)
        TR = TR.swap_gate(axes=(2, 3))
        V = V.tensordot(TR, ([L_unit*Lx, L_unit*Lx + 1, 0], [0, 3, 2]))


        TL = next(TL_gen)
        TL_start = TL.swap_gate(axes=(0, 1))
        V = TL_start.tensordot(V, ([3], [0]))
        for i in range(1, L_unit*Lx - 1):
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
            V = TL.tensordot(V, ([0, 3], [i + 1, i + 2]))
            V = V.swap_gate(axes=(0,2))
            order = list(range(2, i + 3)) + [0, 1] + list(range(i + 3, L_unit*Lx + 2))
            V = V.transpose(order)

        if not APBC:
            # add sigma_z to leg 0:
            V = V.swap_gate(axes=(0,0))
        TL = next(TL_gen)
        TL = TL.swap_gate(axes=(1, 2))
        V = V.tensordot(TL, ([L_unit*Lx, L_unit*Lx + 1, 0], [0, 3, 2]))
        v1, _ = yastn.Tensor.compress_to_1d(V, meta)
        return v1.cpu().numpy()


    legs = [next(TR_gen).get_legs(axes=3).conj() for _ in range(L_unit*Lx)]
    v0 = yastn.rand(config=yastn_config, n=charge_sector, legs=legs)
    data, meta = yastn.Tensor.compress_to_1d(v0)

    sigma_L_sigma_R = LinearOperator(
        (len(data), len(data)),
        matvec=partial(mv, meta=meta) ,
        dtype="complex128" if next(TL_gen).is_complex() else "float64",
    )

    vals, evecs = eigs(
        sigma_L_sigma_R, k=num_evals, v0=data.cpu().numpy(), which="LM", return_eigenvectors=True, maxiter=int(10**8)
    )

    return vals, evecs, meta

def compress_T_horizontal(T, D_max):
    T1 = T.fuse_legs(axes=(1, (0, 2, 3)))
    u1, _, _ = T1.svd_with_truncation(sU=T1.s[0], D_total=D_max)

    T2 = T.fuse_legs(axes=(3, (0, 1, 2)))
    u2, _, _ = T2.svd_with_truncation(sU=T2.s[0], D_total=D_max)

    T = u1.tensordot(T, axes=(0, 1), conj=(1, 0))
    T = T.transpose(axes=(1, 0, 2, 3))
    T = u2.tensordot(T, axes=(0, 3), conj=(1, 0))
    T = T.transpose(axes=(1, 2, 3, 0))

    return T, u1, u2

def compress_T_vertical(T, D_max):
    T1 = T.fuse_legs(axes=(0, (1, 2, 3)))
    u1, _, _ = T1.svd_with_truncation(sU=T1.s[0], D_total=D_max)

    # T2 = T.fuse_legs(axes=(2, (0, 1, 3)))
    # u2, _, _ = T2.svd_with_truncation(sU=T2.s[0], D_total=D_max)

    T = u1.tensordot(T, axes=(0, 0), conj=(1, 0))
    T = u1.tensordot(T, axes=(0, 2))
    T = T.transpose(axes=(1, 2, 0, 3))

    return T

# def compress_Ts(T1, T2, D_max):
#     T1T2 = T1.tensordot(T2, axes=(3, 1))
#     T1T2 = T1T2.fuse_legs(axes=((0, 1, 2), (3, 4, 5)))
#     print(T1T2)
#     u1, s1, vh1 = T1T2.svd_with_truncation(sU=T1.s[3], policy="fullrank", D_block=D_max, D_total=D_max)
#     T1 = u1@s1.sqrt()
#     T1 = T1.unfuse_legs(axes=0)

#     T2 = s1.sqrt()@vh1
#     T2 = T2.unfuse_legs(axes=1)

#     T2T1 = T2.tensordot(T1, axes=(3, 1))
#     T2T1 = T2T1.fuse_legs(axes=((0, 1, 2), (3, 4, 5)))
#     print(T2T1)
#     u2, s2, vh2 = T2T1.svd_with_truncation(sU=T2.s[3], policy="fullrank", D_block=D_max, D_total=D_max)
#     T2 = u2@s2.sqrt()
#     T2 = T2.unfuse_legs(axes=0)

#     T1 = s2.sqrt()@vh2
#     T1 = T1.unfuse_legs(axes=1)



#     return T1, T2

def compress_Ts(T1, T2, D_max):

    T1, u1, u2 = compress_T_horizontal(T1, D_max)

    T2 = T2.tensordot(u1, axes=(3, 0))
    T2 = u2.tensordot(T2, axes=(0, 1))
    T2 = T2.transpose(axes=(1, 0, 2, 3))
    return T1, T2

def coarse_grain(mpo_list1, mpo_list2, D_phy, D_mpo, repeat=1):
    def rg(o1, o2):
        N = len(o1)
        T1, T2 = None, None
        res_list1, res_list2 = [], []
        for i in range(N):
            print(f"absorb {i:d}")
            if T1 is None:
                T1 = o1[i]
            if T2 is None:
                T2 = o2[i]
            else:
                T1 = T1.tensordot(o1[i], axes=(2, 0))
                T1 = T1.fuse_legs(axes=(0, (1, 3), 4, (2, 5)))

                T2 = T2.tensordot(o2[i], axes=(2, 0))
                T2 = T2.fuse_legs(axes=(0, (1, 3), 4, (2, 5)))
                T1, T2 = compress_Ts(T1, T2, D_max=D_phy)


            if i == N-1:
                T1, T2 = compress_Ts(T1, T2, D_max=D_phy)
                # T1, T2 = compress_T_vertical(T1, D_max=D_mpo), compress_T_vertical(T2, D_max=D_mpo)
                res_list1.append(T1)
                res_list2.append(T2)
        return res_list1, res_list2

    o1, o2 = mpo_list1, mpo_list2
    for i in range(repeat):
        o1, o2 = rg(o1, o2)

    return o1, o2

def analyze_momentum_sector_unit_L(vals, evecs, meta, unit_L, Lx, k_shift=0, APBC=True):
    # unit_L: linear length of the unit-cell; Lx: width of the cylinder
    def translation(V):
        if APBC:
            # With sigma_z on Lx-1
            for i in range(Lx):
                V = V.swap_gate(axes=(Lx-1, i))
        else:
            # Without sigma_z on Lx-1
            for i in range(Lx-1):
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
        for k in range(Lx):
            k = k+k_shift
            if (
                yastn.linalg.norm(
                    translation(V)
                    - np.exp(1j * 2 * np.pi * (k) / Lx) * V
                )/yastn.linalg.norm(V)
                <= 1e-1
            ):
                if k not in evals:
                    evals[k] = [val]
                else:
                    evals[k].append(val)
                # break
                cnt += 1
        assert cnt == 1
    return evals


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--D", type=int, default=8)
    parser.add_argument("--Dcut", type=int, default=8)
    parser.add_argument("--env_chi", type=int, default=20)
    parser.add_argument("--L", type=int, default=5)
    parser.add_argument("--state_file", type=str, default="none")
    parser.add_argument("--env_dict_file", type=str, default="none")
    parser.add_argument("--output", type=str, default="tmp")
    parser.add_argument("--mpo_dir", type=str, default="./tmp")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="float64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fermionic", action='store_true')
    parser.add_argument("--APBC", action='store_true')
    args = parser.parse_args()

    omp_cores=args.num_threads
    torch.set_num_threads(omp_cores)
    torch.set_num_interop_threads(omp_cores)

    D, env_chi = args.D, args.env_chi
    state_file, env_dict_file = args.state_file, args.env_dict_file
    print(args.device, f"APBC: {args.APBC}")

    yastn_config = yastn.make_config(
        backend=backend,
        # sym=sym_Z2,
        sym=sym_U1,
        # sym=sym_none,
        fermionic=args.fermionic,
        default_device=args.device,
        default_dtype=args.data_type,
    )


    # Converge environment
    opts_svd = {
        "D_total": env_chi,
        "tol": 1e-10,
        "eps_multiplet": 1e-8,
        "fix_signs": True,
        "truncate_multiplets": True,
    }

    start_site=0
    state = load_PepsAD(yastn_config, state_file)
    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(env_chi,))
    env = EnvCTM(state, init="eye", leg=env_leg)

    with open(env_dict_file, "rb") as handle:
        d = pickle.load(handle)
    env = load_env_from_dict(env, d,  yastn_config)
    print("Loaded Env")
    print(env[(0,0)].l)
    print(env[(1,0)].l)
    print(env[(2,0)].l)

    # HOSVD
    D_max = args.Dcut
    D_mpo = env_chi
    start = env.sites()[0]

    mpo_R, mpo_L = Ts_to_MPO_list(env, start_site=start, dirn="x")
    if not os.path.exists(args.mpo_dir):
        os.makedirs(args.mpo_dir, exist_ok=True)
    mpos_file = os.path.join(args.mpo_dir, f"mpo_lists_D_{D:d}_D_compress_{D_max:d}_chi_{env_chi:d}")

    rerun = True
    if os.path.exists(mpos_file) and not rerun:
        L_list, R_list = [], []
        with open(mpos_file, "rb") as f:
            L_list_dicts, R_list_dicts = pickle.load(f)
        for L_dict, R_dict in zip(L_list_dicts, R_list_dicts):
            L_list.append(load_from_dict(yastn_config, L_dict))
            R_list.append(load_from_dict(yastn_config, R_dict))
        print("Loaded compressed MPOs")
    else:
        L_list, R_list = coarse_grain(mpo_L, mpo_R, D_max, D_mpo, repeat=1)
        L_list_dicts = [L_list[i].save_to_dict() for i in range(len(L_list))]
        R_list_dicts = [R_list[i].save_to_dict() for i in range(len(R_list))]

        with open(mpos_file, "wb") as f:
            pickle.dump((L_list_dicts, R_list_dicts), f)


    Lx = args.L
    n, unit_L = 0, 1
    id_charge = sym_U1.zero()
    vals, evecs, meta = ent_spec_cylinder_Lx(yastn_config, cycle(L_list), cycle(R_list), Lx, unit_L, charge_sector=id_charge, num_evals=60, APBC=args.APBC)
    vals_dict = analyze_momentum_sector_unit_L(vals/vals[0], evecs, meta, unit_L, Lx, k_shift=0, APBC=args.APBC)
    with open(args.output, "wb") as f:
        pickle.dump(vals_dict, f)
