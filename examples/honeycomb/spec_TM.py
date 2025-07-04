import os
import pickle
import numpy as np
import torch
import scipy.sparse as sp
import context


import yastn.yastn as yastn
import matplotlib.pyplot as plt
from yastn.yastn.sym import sym_Z2, sym_U1
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tn.fpeps import EnvCTM
from ipeps.integration_yastn import load_PepsAD


def TM_vert(env):
    TMs = []

    for site in env.sites():
        Tl = env[site].l
        r_site = env.nn_site(site, "l")
        Tr = env[r_site].r
        res = Tl.tensordot(Tr, axes=((1, 1)))
        res = res.transpose(axes=(1, 2, 0, 3))

        walker = env.nn_site(site, "b")
        walker_r = env.nn_site(r_site, "b")
        for _ in range(env.dims[0]-1):
            Tl = env[walker].l
            Tr = env[walker_r].r
            # 0       1
            #  \     /
            #    res
            #  /    \
            # Tl -- Tr
            # |     |
            # 2     3

            tmp = Tl.tensordot(Tr, axes=(1, 1))
            tmp = tmp.transpose(axes=(1, 2, 0, 3))
            res = res.tensordot(tmp, axes=((2, 3), (0, 1)))
            walker = env.nn_site(walker, "b")
            walker_r = env.nn_site(walker_r, "b")
        res = res.fuse_legs(axes=((0, 1), (2, 3)))
        TMs.append(res)
    return TMs


def TM_horz(env):
    TMs = []
    for site in env.sites():
        Tt = env[site].t
        b_site = site
        b_site = env.nn_site(b_site, "t")
        Tb = env[b_site].b
        res = Tt.tensordot(Tb, axes=((1, 1)))
        res = res.transpose(axes=(0, 3, 1, 2))
        walker_t = env.nn_site(site, "r")
        walker_b = env.nn_site(b_site, "r")
        for _ in range(env.dims[1]-1):
            Tt = env[walker_t].t
            Tb = env[walker_b].b
            # 0---\     /--- Tt --- 2
            #       res      |
            # 1---/     \--- Tb --- 3
            res = res.tensordot(Tt, axes=(2, 0))
            res = res.tensordot(Tb, axes=((2, 3), (2, 1)))
            walker_t = env.nn_site(walker_t, "r")
            walker_b = env.nn_site(walker_b, "r")
        TMs.append(res.fuse_legs(axes=((0,1), (2, 3))))
    return TMs

def load_env_from_dict(env, d, yastn_config):
    assert d['class'] == 'EnvCTM'
    for site in d['data']:
        for dirn in d['data'][site]:
            setattr(env[site], dirn, yastn.load_from_dict(yastn_config, d['data'][site][dirn]))

    return env

def plot_eigs(ws):
    xs = np.zeros(len(ws))
    fig, ax = plt.subplots()
    ax.plot(xs, np.abs(ws)/np.max(np.abs(ws)), lw=0, marker='_', markersize=10)
    return fig, ax

if __name__ == "__main__":
    for chi in [36, 72, 108, 144, 216, 288]:
        state_file = (
            # f"./t1t2t3_V1_grad/seed_123/V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu_cont4_state.json"
            f"./t1t2t3_V1_grad/CDW_3x3/V1_1.0_V2_0.5_V3_0.5_t1_0.05_t2_0.035_t3_-0.045_phi_0_3x3_N3_D_6_chi_72_fullrank_cpu_state.json"
        )

        yastn_config = yastn.make_config(
            backend=backend,
            sym=sym_U1,
            fermionic=True,
            default_device="cpu",
            default_dtype="complex128",
        )
        D = 6
        opt_chi = 72
        omp_cores = 16
        torch.set_num_threads(omp_cores)

        # Converge environment
        opts_svd = {
            "D_total": opt_chi,
            "tol": 1e-8,
            "eps_multiplet": 1e-8,
            "fix_signs": True,
            "truncate_multiplets": True,
        }

        env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(opt_chi,))
        # env_dict_filename = "./t1t2t3_V1_grad/seed_123/es/env_dict_V1_1.0_t1_0.1_t2_0.07_t3_-0.09_3x3_N9_D_6_chi_72_fullrank_cpu"
        env_dict_filename = f"./t1t2t3_V1_grad/CDW_3x3/es/env_dict_V1_1.0_V2_0.5_V3_0.5_t1_0.05_t2_0.035_t3_-0.045_phi_0_3x3_N3_optchi_72_chi_{chi:d}_fullrank_cpu"

        state = load_PepsAD(yastn_config, state_file)
        with open(env_dict_filename, "rb") as handle:
            d = pickle.load(handle)
        env = EnvCTM(state, init="eye", leg=env_leg)
        env = load_env_from_dict(env, d,  yastn_config)
        print("Loaded env")
        print(env[(0,0)].t)

        TM_h = TM_horz(env)
        for i, TM in enumerate(TM_h):
            TM_sp = sp.csr_matrix(TM.to_dense().numpy())
            ws = sp.linalg.eigs(TM_sp, k=50, which='LM', return_eigenvectors=False)
            # print(np.abs(ws)[::-1])

            filename = f"obs/CDW_3x3_N3/spec_TM_D_6_optchi_72_chi_{chi:d}_site_{i:d}_dirn_(0,1).npy"
            with open(filename, 'wb') as f:
                np.save(f, ws)
            # fig, ax = plot_eigs(ws)
            # ax.set_ylabel(r"$|\lambda_i|$")
            # plt.savefig(f"./obs/TM_spec_site_{i:d}_dirn_(0,1)")

        TM_v = TM_vert(env)
        for i, TM in enumerate(TM_v):
            TM_sp = sp.csr_matrix(TM.to_dense().numpy())
            ws = sp.linalg.eigs(TM_sp, k=50, which='LM', return_eigenvectors=False)
            # print(np.abs(ws)[::-1])

            filename = f"obs/CDW_3x3_N3/spec_TM_D_6_optchi_72_chi_{chi:d}_site_{i:d}_dirn_(1,0).npy"
            with open(filename, 'wb') as f:
                np.save(f, ws)