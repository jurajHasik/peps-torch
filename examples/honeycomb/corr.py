import os, pickle, argparse

import context
import numpy as np
import torch

from ctm.generic.env_yastn import ctmrg
from ctm.generic_abelian.corrf_fermionic import corr
from ipeps.integration_yastn import load_PepsAD

import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs.rdm import op_order
from yastn.yastn.backend import backend_torch as backend
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_chi", type=int, default=20)
    parser.add_argument("--dist", type=int, default=50)
    parser.add_argument("--state_file", type=str, default="none")
    parser.add_argument("--env_dict_file", type=str, default="none")
    parser.add_argument("--obs_dir", type=str, default="tmp")
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--data_type", type=str, default="float64")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--fermionic", action='store_true')
    args = parser.parse_args()

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=args.fermionic,
        default_device=args.device,
        default_dtype=args.data_type,
    )
    torch.set_num_threads(args.num_threads)

    # Converge environment
    env_chi = args.env_chi
    if os.path.isfile(args.state_file):
        state = load_PepsAD(yastn_config, state_file=args.state_file).to_Peps()
    else:
        raise ValueError("Provide valid state file via --state_file")

    if os.path.isfile(args.env_dict_file):
        with open(args.env_dict_file, 'rb') as file:
            d = pickle.load(file)
        env = yastn.from_dict(d, config=yastn_config)
    else:
    # Converge environment
        opts_svd = {
            "D_total": env_chi,
            "tol": 1e-10,
            "eps_multiplet": 1e-8,
            "fix_signs": True,
            "truncate_multiplets": True,
        }
        env = EnvCTM(state, init="eye")
        def ctm_conv_check_f(env, conv_history):
            corner_tol = 1e-9
            converged, max_dsv, conv_history = ctm_conv_corner_spec(env, conv_history, corner_tol)
            print(f"CTM iter {len(conv_history)} |delta_C| {max_dsv}")
            return converged, conv_history

        env, _, _, _, _ = ctmrg(env, ctm_conv_check_f=ctm_conv_check_f, max_sweeps=1500, options_svd=opts_svd)
        d = env.to_dict(level=2)
        with open(args.env_dict_file, 'wb') as file:
            pickle.dump(d, file)


    start_site=0
    _tmp_config = {x: y for x, y in yastn_config._asdict().items() if x != "sym"}
    sf = yastn.operators.SpinfulFermions(sym=str(yastn_config.sym), **_tmp_config)
    n_A = sf.n(spin="u")  # parity-even operator, no swap gate needed
    n_B = sf.n(spin="d")
    c_A = sf.c(spin="u")
    cp_A = sf.cp(spin="u")
    c_B = sf.c(spin="d")
    cp_B = sf.cp(spin="d")
    I = sf.I()

    torch.set_printoptions(precision=10)

    ci_A, cjp_A = op_order(c_A, cp_A, ordered=True, fermionic=True)
    cip_A, cj_B = op_order(cp_A, c_B, ordered=True, fermionic=True)

    data_dir = args.obs_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    for i, site in enumerate(state.sites()):
        dist = args.dist
        corrf, _ = corr(state, env, site, (0, 1), ci_A, cjp_A, dist, connected=False)
        filename = os.path.join(data_dir, f"cA_cpA_corrf_site_{i:d}_dirn_(0,1).npy")
        with open(filename, 'wb') as f:
            np.save(f, np.arange(1, dist+1))
            np.save(f, corrf)

        corrf, _ = corr(state, env, site, (1, 0), ci_A, cjp_A, dist, connected=False)
        filename = os.path.join(data_dir, f"cA_cpA_corrf_site_{i:d}_dirn_(1,0).npy")
        with open(filename, 'wb') as f:
            np.save(f, np.arange(1, dist+1))
            np.save(f, corrf)

    for i, site in enumerate(state.sites()):
        dist = args.dist
        # nB_n_(0,1)
        corrf, op_vals = corr(state, env, site, (0, 1), n_B, n_A, dist, connected=True)
        # filename = os.path.join(data_dir, f"nB_nA_corrf_site_{i:d}_dirn_(0,1).npy")
        filename = os.path.join(data_dir, f"normalized_nB_nA_corrf_site_{i:d}_dirn_(0,1).npy")
        with open(filename, 'wb') as f:
            np.save(f, np.arange(1, dist+1))
            np.save(f, corrf/op_vals)

        corrf, op_vals = corr(state, env, site, (0, 1), n_B, n_B, dist, connected=True)
        # filename = os.path.join(data_dir, f"nB_nB_corrf_site_{i:d}_dirn_(0,1).npy")
        filename = os.path.join(data_dir, f"normalized_nB_nB_corrf_site_{i:d}_dirn_(0,1).npy")
        with open(filename, 'wb') as f:
            np.save(f, np.arange(1, dist+1))
            np.save(f, corrf/op_vals)

        # nB_n_(1, 0)
        corrf, op_vals = corr(state, env, site, (1, 0), n_B, n_A, dist, connected=True)
        filename = os.path.join(data_dir, f"normalized_nB_nA_corrf_site_{i:d}_dirn_(1,0).npy")
        o1 = env.measure_1site(n_B@n_A, site=site).item()/(env.measure_1site(n_A, site=site).item()*env.measure_1site(n_B, site=site).item())
        with open(filename, 'wb') as f:
            np.save(f, np.arange(0, dist+1))
            np.save(f, np.insert(corrf/op_vals, 0, o1))

        corrf, op_vals = corr(state, env, site, (1, 0), n_B, n_B, dist, connected=True)
        filename = os.path.join(data_dir, f"normalized_nB_nB_corrf_site_{i:d}_dirn_(1,0).npy")
        with open(filename, 'wb') as f:
            np.save(f, np.arange(1, dist+1))
            np.save(f, corrf/op_vals)

        # # nA_n_(0,1)
        # corrf, op_vals = corr(state, env, site, (0, 1), n_A, n_A, dist, connected=True)
        # filename = os.path.join(data_dir, f"nA_nA_corrf_site_{i:d}_dirn_(0,1).npy")
        # with open(filename, 'wb') as f:
        #     np.save(f, np.arange(1, dist+1))
        #     np.save(f, corrf - op_vals)

        # corrf, op_vals = corr(state, env, site, (0, 1), n_A, n_B, dist, connected=True)
        # filename = os.path.join(data_dir, f"nA_nB_corrf_site_{i:d}_dirn_(0,1).npy")
        # o1 = env.measure_1site(n_B@n_A, site=site).item() - env.measure_1site(n_A, site=site).item()*env.measure_1site(n_B, site=site).item()
        # with open(filename, 'wb') as f:
        #     np.save(f, np.arange(0, dist+1))
        #     np.save(f, np.insert(corrf - op_vals, 0, o1))

        # # nA_n_(1, 0)
        # corrf, op_vals = corr(state, env, site, (1, 0), n_A, n_A, dist, connected=True)
        # filename = os.path.join(data_dir, f"nA_nA_corrf_site_{i:d}_dirn_(1,0).npy")
        # with open(filename, 'wb') as f:
        #     np.save(f, np.arange(1, dist+1))
        #     np.save(f, corrf - op_vals)

        # corrf, op_vals = corr(state, env, site, (1, 0), n_A, n_B, dist, connected=True)
        # filename = os.path.join(data_dir, f"nA_nB_corrf_site_{i:d}_dirn_(1,0).npy")
        # with open(filename, 'wb') as f:
        #     np.save(f, np.arange(1, dist+1))
        #     np.save(f, corrf - op_vals)