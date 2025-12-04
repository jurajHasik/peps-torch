import json, pickle
import logging
import os
import time
import ast, argparse


import context
import config as cfg
import numpy as np
import torch

import yastn.yastn as yastn
from yastn.yastn.tn.fpeps import EnvCTM, Peps2Layers
from ctm.generic.env_yastn import YASTN_PROJ_METHOD
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps.envs.rdm import *
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint, fp_ctmrg
from ipeps.integration_yastn import load_PepsAD
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
parser.add_argument("--eval_loss", action='store_true')
parser.add_argument("--devices", help='cpu or (list of) cuda. Default is cpu', default=None, dest='devices', nargs="+")

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
    # D = sum([bond_dims[t] for t in bond_dims.keys()])

    args, unknown_args = parser.parse_known_args(
        [
            "--opt_max_iter",
            "500",
            "--CTMARGS_ctm_env_init_type",
            "eye",
            "--OPTARGS_no_opt_ctm_reinit",
            "--OPTARGS_no_line_search_ctm_reinit",
            "--GLOBALARGS_dtype",
            "complex128",
            # "--OPTARGS_opt_log_grad",
            # "--CTMARGS_fwd_checkpoint_move",
            # "--OPTARGS_line_search",
            # "backtracking",
            # "strong_wolfe",
        ],
        namespace=args,
    )


    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend
    torch.set_num_threads(args.omp_cores)
    torch.set_num_interop_threads(args.omp_cores)
    cfg.configure(args)
    cfg.print_config()

    # process devices
    ctm_devices= ['cpu'] if args.devices is None else args.devices

    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym_U1,
        fermionic=True,
        # default_device=cfg.global_args.device,
        default_device=args.devices[0],
        default_dtype=cfg.global_args.dtype,
        tensordot_policy="no_fusion",
    )
    yastn_config.backend.random_seed(args.seed)

    args.t2, args.t3, args.phi= 0.7*args.t1, -0.9*args.t1, 0.35*np.pi
    model = tV_model(yastn_config, V1=args.V1, V2=args.V2, V3=args.V3, t1=args.t1, t2=args.t2, t3=args.t3, phi=args.phi, mu=args.mu, m=args.m)

    def loss_fn(stateAD, ctm_env_in, opt_context):
        stateAD.sync_()
        state = stateAD.to_Peps()
        ctm_args = opt_context["ctm_args"]
        opt_args = opt_context["opt_args"]
        if ctm_args.projector_svd_method == "DEFAULT":
            ctm_args.projector_svd_method = "GESDD"

        # possibly re-initialize the environment
        if opt_args.opt_ctm_reinit:
            print("Reinit")
            chi = cfg.main_args.chi
            env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
            ctm_env_in = EnvCTM(state, init=ctm_args.ctm_env_init_type, leg=env_leg)

        opts_svd = {
            "D_total": cfg.main_args.chi,
            'D_krylov':cfg.main_args.chi,
            'D_block': cfg.main_args.chi,
            "tol": cfg.ctm_args.projector_svd_reltol,
            "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
            "truncate_multiplets": True,
        }
        ctm_env_out = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd={'opts_svd': opts_svd, 'corner_tol': cfg.ctm_args.ctm_conv_tol, 'max_sweeps': cfg.ctm_args.ctm_max_iter,
                'method': "2site", 'use_qr': False, 'svd_policy': YASTN_PROJ_METHOD[ctm_args.projector_svd_method], \
                "svds_thresh":ctm_args.fwd_svds_thresh, 'verbosity': 3}, \
            ctm_opts_fp={'svd_policy': 'fullrank'}, fwd_devices=ctm_devices)
        d = ctm_env_out.to_dict()
        with open(args.out_prefix + "_ctm_env_dict", "wb") as f:
            pickle.dump(d, f)
        ctm_log, t_ctm, t_check = FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check
        # 2) evaluate loss with converged environment
        t_loss_before = time.perf_counter()
        loss = model.energy_per_site(state, ctm_env_out, eval_checkpoint="nonreentrant")  # H= H_0 - mu * (nA + nB)
        t_loss_after = time.perf_counter()
        t_loss = t_loss_after - t_loss_before
        return (loss, ctm_env_out, *ctm_log, t_ctm, t_check, t_loss)

    @torch.no_grad()
    def post_proc(stateAD, env, opt_context):
        stateAD.sync_()
        stateAD.normalize_()

    @torch.no_grad()
    def obs_fn(stateAD, ctm_env, opt_context):
        stateAD.sync_()
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

        obs = model.eval_obs(stateAD, ctm_env)
        log.info(json.dumps(obs))

    if args.instate is None or not os.path.exists(args.instate):
        if args.pattern == '1x1':
            stateAD = random_1x1_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3':
            stateAD = random_3x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3_2':
            stateAD = random_3x3_2_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x3_9':
            stateAD = random_3x3_9_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '1x3':
            stateAD = random_1x3_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '1x6':
            stateAD = random_1x6_state_U1(bond_dims=bond_dims, config=yastn_config)
        elif args.pattern == '3x1':
            stateAD = random_3x1_state_U1(bond_dims=bond_dims, config=yastn_config)
        else:
            raise ValueError(f"Unknown pattern: {args.pattern}")
    else:
        stateAD = load_PepsAD(yastn_config, args.instate)
        stateAD.normalize_()
        log.log(logging.INFO, "loaded " + args.instate)
        print("loaded ", args.instate)
        stateAD.add_noise_(args.instate_noise)

    chi = cfg.main_args.chi
    env_leg = yastn.Leg(yastn_config, s=1, t=(0,), D=(chi,))
    ctm_env_in = EnvCTM(stateAD.to_Peps(), init=cfg.ctm_args.ctm_env_init_type, leg=env_leg)

    # load env from the previous file, if it exists
    def load_env_from_dict(env, d, yastn_config):
        assert d['class'] == 'EnvCTM'
        for site in d['data']:
            for dirn in d['data'][site]:
                setattr(env[site], dirn, yastn.load_from_dict(yastn_config, d['data'][site][dirn]))

        return env

    suffix = "_state.json"
    if args.instate is not None:
        in_env_dict_file = args.instate[:-len(suffix)] + "_ctm_env_dict"

        rerun=False
        if os.path.exists(in_env_dict_file) and not rerun:
            print(in_env_dict_file)
            with open(in_env_dict_file, "rb") as f:
                d = pickle.load(f)
            ctm_env_in = yastn.from_dict(d)
            ctm_env_in.psi = Peps2Layers(bra=stateAD.to_Peps())


    if args.eval_loss:
        opt_context = {"ctm_args": cfg.ctm_args, "opt_args": cfg.opt_args}
        loss_fn(stateAD, ctm_env_in, opt_context)
    else:
        optimize_state(stateAD, ctm_env_in, loss_fn, obs_fn=obs_fn, post_proc=post_proc)

if __name__ == "__main__":
    main()
