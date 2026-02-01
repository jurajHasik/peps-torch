import context
import os, time, json
import torch
import numpy as np
import config as cfg


import unittest
import logging

import yastn.yastn as yastn
from yastn.yastn.sym import sym_U1
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint,fp_ctmrg
from ipeps.integration_yastn import load_PepsAD
from models.fermion.tv_model import *
from examples.honeycomb.CI import parse_dict
from ctm.generic.env_yastn import YASTN_PROJ_METHOD
from optim.ad_optim_lbfgs_mod import optimize_state

logging.basicConfig(filename=f"{__file__}.log", filemode='w', level=logging.INFO)

parser= cfg.get_args_parser()

parser.add_argument(
    "--V1", type=float, default=0.0, help="Nearest-neighbor interaction"
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
parser.add_argument("--bond_dims", type=parse_dict, help="dict of bond dimensions keyed on charge sectors  (e.g., \"{'charge1': 'D1', 'charge2': D2}\")")
parser.add_argument("--devices", help='cpu or (list of) cuda. Default is cpu', default=None, dest='devices', nargs="+")
parser.add_argument(
    "--yast_backend",
    type=str,
    default="torch",
    help="YAST backend",
    choices=["torch", "torch_cpp"],
)
args, unknown_args = parser.parse_known_args()

import re
import ast
from typing import Any, Dict, Tuple, Union

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

def parse_energy_and_nab(text: str) -> Tuple[int, float, Union[float, Dict[str, float]], Union[float, Dict[str, float]]]:
    """
    Parse the last reported (epoch, energy) and the last nA/nB dictionary from program output.

    Returns:
        (epoch, energy, nA, nB)

    Where nA/nB are:
        - floats if exactly one nA_* and one nB_* key are found
        - otherwise dicts {key: value}
    """
    last_epoch = None
    last_energy = None
    last_nA: Dict[str, float] = {}
    last_nB: Dict[str, float] = {}

    energy_re = re.compile(rf"epoch\s*,\s*energy\s*,\s*(\d+)\s*,\s*({_FLOAT})")

    for line in text.splitlines():
        # 1) epoch, energy, <epoch>, <energy>
        m = energy_re.search(line)
        if m:
            last_epoch = int(m.group(1))
            last_energy = float(m.group(2))
            continue

        # 2) dict line with nA_/nB_
        s = line.strip()
        if s.startswith("{") and ("nA_" in s) and ("nB_" in s):
            try:
                d: Dict[str, Any] = ast.literal_eval(s)  # safe for literals
                nA = {k: float(v) for k, v in d.items() if isinstance(k, str) and k.startswith("nA_")}
                nB = {k: float(v) for k, v in d.items() if isinstance(k, str) and k.startswith("nB_")}
                if nA and nB:
                    last_nA, last_nB = nA, nB
            except Exception:
                # ignore malformed dict lines
                pass

    if last_epoch is None or last_energy is None:
        raise AssertionError("Did not find any line like: 'epoch, energy, <epoch>, <energy>'")

    if not last_nA or not last_nB:
        raise AssertionError("Did not find any dict line containing both 'nA_' and 'nB_'")

    # If there is only one site reported, return scalars for convenience
    if len(last_nA) == 1 and len(last_nB) == 1:
        return last_epoch, last_energy, next(iter(last_nA.values())), next(iter(last_nB.values()))

    return last_epoch, last_energy, last_nA, last_nB


def main():
    global args
    bond_dims = args.bond_dims
    args, unknown_args = parser.parse_known_args(
        [
            "--CTMARGS_ctm_env_init_type",
            "eye",
            "--OPTARGS_no_opt_ctm_reinit",
            "--OPTARGS_no_line_search_ctm_reinit",
            "--GLOBALARGS_dtype",
            "complex128",
        ],
        namespace=args,
    )

    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend
    cfg.configure(args)
    cfg.print_config()
    ctm_devices= ['cpu'] if args.devices is None else args.devices

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
            "tol": cfg.ctm_args.projector_svd_reltol,
            "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
            "truncate_multiplets": True,
        }
        ctm_env_out = fp_ctmrg(ctm_env_in, \
            ctm_opts_fwd={'opts_svd': opts_svd, 'corner_tol': cfg.ctm_args.ctm_conv_tol, 'max_sweeps': cfg.ctm_args.ctm_max_iter,
                'method': "2site", 'use_qr': False, 'svd_policy': YASTN_PROJ_METHOD[ctm_args.projector_svd_method], \
                "svds_thresh":ctm_args.fwd_svds_thresh, 'verbosity':1}, \
            ctm_opts_fp={'svd_policy': 'fullrank'}, fwd_devices=ctm_devices)

        ctm_log, t_ctm, t_check = FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check
        # 2) evaluate loss with converged environment
        t_loss_before = time.perf_counter()
        loss = model.energy_per_site(state, ctm_env_out, eval_checkpoint="nonreentrant")  # H= H_0 - mu * (nA + nB)
        t_loss_after = time.perf_counter()
        t_loss = t_loss_after - t_loss_before
        log.info(f"energy: {loss}")
        return (loss, ctm_env_out, *ctm_log, t_ctm, t_check, t_loss)

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
                    ["epoch, energy", f"{epoch}", f"{loss}"]
                    + [f"{x.item()}" for x in model.get_parameters()]
                )
            )

        obs = model.eval_obs(stateAD.to_Peps(), ctm_env)
        # log.info(json.dumps(obs))
        print(obs)

    if args.instate is None or not os.path.exists(args.instate):
        if args.pattern == '1x1':
            stateAD = random_1x1_state_U1(bond_dims=bond_dims, config=yastn_config)
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
    optimize_state(stateAD, ctm_env_in, loss_fn, obs_fn=obs_fn)

if __name__ == "__main__":
    main()


class TestOptim_CI_state(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run-opt_u1_CI"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../test-input/abelian/CI_D3_1x1_U1_state.json"
        args.t1=1.0
        args.t2, args.t3, args.phi= 0.7*args.t1, -0.9*args.t1, 0.35*np.pi
        args.bond_dims = {-1:1,0:1,1:1}
        args.chi=36
        args.instate_noise=0.3
        args.seed=123
        args.out_prefix=self.OUT_PRFX
        args.opt_max_iter=3
        args.CTMARGS_ctm_conv_tol=1e-10
        args.CTMARGS_ctm_max_iter=800
        args.GLOBALARGS_dtype= "complex128"

    def test_basic_opt_noisy_CI(self):
        from io import StringIO
        from unittest.mock import patch
        from cmath import isclose

        with patch('sys.stdout', new = StringIO()) as tmp_out:
            main()
        tmp_out.seek(0)

        out = tmp_out.getvalue()
        print(out)
        epoch, energy, nA, nB = parse_energy_and_nab(out)

        # compare with the reference
        ref_energy = -2.6116462661745645
        ref_nA = 0.5092230390029766
        ref_nB = 0.49077769168530994

        assert isclose(energy,ref_energy, rel_tol=self.tol, abs_tol=self.tol)
        assert isclose(nA,ref_nA, rel_tol=self.tol, abs_tol=self.tol)
        assert isclose(nB,ref_nB, rel_tol=self.tol, abs_tol=self.tol)


    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)