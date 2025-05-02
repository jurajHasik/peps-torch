import json
import logging
import os
import time
import ast, argparse


import context
import config as cfg
import numpy as np
import torch

import yastn.yastn as yastn
from yastn.yastn.tn.fpeps import EnvCTM, RectangularUnitcell, Bond
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps.envs.rdm import *
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint, refill_env, fp_ctmrg
from ipeps.integration_yastn import PepsAD, load_PepsAD
from optim.ad_optim_lbfgs_mod import optimize_state

from models.fermion.tv_model import *


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

def main():
    # global args
    args = parser.parse_args()  # process command line arguments
    bond_dims = args.bond_dims
    D = sum([bond_dims[t] for t in bond_dims.keys()])
    seed_dir = f"FCI_U1_data_seed_{args.seed:d}"
    if not os.path.exists(seed_dir):
        os.mkdir(seed_dir)

    args, unknown_args = parser.parse_known_args(
        [
            "--opt_max_iter",
            "100",
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

    # state_file = f"FCI_U1_data_seed_{args.seed}/FCI_fp_{args.pattern}_cores_{args.omp_cores:d}_D_{D:d}_U1_chi_{args.chi:d}_V_{args.V1:.2f}_t1_{args.t1:.2f}_t2_{args.t2:.2f}_t3_{args.t3:.2f}_phi_{args.phi/np.pi:.2f}_mu_{args.mu:.3f}"
    # args, unknown_args = parser.parse_known_args(
    #     [
    #         "--out_prefix",
    #         state_file,
    #     ],
    #     namespace=args,
    # )


    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(args.omp_cores)
    cfg.configure(args)
    cfg.print_config()
    log.log(logging.INFO, "device: "+cfg.global_args.device)
    log.log(logging.INFO, f"bond_dims:{bond_dims}")
    log.log(logging.INFO, f"ctm_args.ctm_max_iter:{cfg.ctm_args.ctm_max_iter}")

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=True,
        default_device=cfg.global_args.device,
        default_dtype=cfg.global_args.dtype,
        tensordot_policy="no_fusion",
    )
    yastn_config.backend.random_seed(args.seed)

    args.t2, args.t3, args.phi= 0.7*args.t1, -0.9*args.t1, 0.35*np.pi
    model = tV_model(yastn_config, V1=args.V1, V2=args.V2, V3=args.V3, t1=args.t1, t2=args.t2, t3=args.t3, phi=args.phi, mu=args.mu, m=args.m)

    def loss_fn(state, ctm_env_in, opt_context):
        state.sync_()
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            print("Reinit")
            chi = cfg.main_args.chi
            env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
            ctm_env_in = EnvCTM(state, init=ctm_args.ctm_env_init_type, leg=env_leg)

        opts_svd = {
            "D_total": cfg.main_args.chi,
            "tol": cfg.ctm_args.projector_svd_reltol,
            "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
            "truncate_multiplets": True,
        }
        ctm_env_out, env_ts_slices, env_ts = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd={'opts_svd': opts_svd, 'corner_tol': 1e-10, 'max_sweeps': cfg.ctm_args.ctm_max_iter,
                'method': "2site", 'use_qr': False, 'svd_policy': ctm_args.fwd_svd_policy, 'D_krylov':args.chi, 'D_block': args.chi, \
                "svds_thresh":ctm_args.fwd_svds_thresh, "svds_solver":ctm_args.fwd_svds_solver}, \
            ctm_opts_fp={'svd_policy': 'fullrank'})

        refill_env(ctm_env_out, env_ts, env_ts_slices)
        ctm_log, t_ctm, t_check = FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check
        # 2) evaluate loss with converged environment
        t_loss_before = time.perf_counter()
        loss = model.energy_per_site(state, ctm_env_out)  # H= H_0 - mu * (nA + nB)
        t_loss_after = time.perf_counter()
        t_loss = t_loss_after - t_loss_before
        return (loss, ctm_env_out, *ctm_log, t_ctm, t_check, t_loss)

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
        else:
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
            print(
                ", ".join(
                    [f"{epoch}", f"{loss}"]
                    + [f"{x.item()}" for x in model.get_parameters()]
                )
            )

        obs = model.eval_obs(state, ctm_env)
        log.info(json.dumps(obs))

    if args.init_state_file is None or not os.path.exists(args.init_state_file):
        if args.pattern == '1x1':
            state = random_1x1_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3':
            state = random_3x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3_2':
            state = random_3x3_2_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '1x3':
            state = random_1x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '1x6':
            state = random_1x6_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x1':
            state = random_3x1_state_U1(bond_dims=bond_dims, config=yastn_config)
        else:
            raise ValueError(f"Unknown pattern: {args.pattern}")
    else:
        state = load_PepsAD(yastn_config, args.init_state_file)
        log.log(logging.INFO, "loaded " + args.init_state_file)
        print("loaded ", args.init_state_file)
        # state.add_noise_(noise=0.2)

    chi = cfg.main_args.chi
    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
    ctm_env_in = EnvCTM(state, init=cfg.ctm_args.ctm_env_init_type, leg=env_leg)
    optimize_state(state, ctm_env_in, loss_fn, obs_fn=obs_fn, post_proc=None)
    # opt_context = {"ctm_args": cfg.ctm_args, "opt_args": cfg.opt_args}
    # loss_fn(state, ctm_env_in, opt_context)

if __name__ == "__main__":
    main()
