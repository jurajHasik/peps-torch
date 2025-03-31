import argparse
import json
import logging
import os
import time
import unittest


import context
import config as cfg
import numpy as np
import torch

import yastn.yastn as yastn
from yastn.yastn.tn.fpeps import EnvCTM
from yastn.yastn.tn.fpeps.envs._env_ctm import ctm_conv_corner_spec
from yastn.yastn.tn.fpeps.envs.fixed_pt import FixedPoint, env_raw_data, refill_env
from yastn.yastn.sym import sym_Z2, sym_U1

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
parser.add_argument("--ansatz", type=str, default="1x1", choices=["1x1","2x1","3x3",], help="ansatz type")
parser.add_argument("--sym", type=str, default="Z2", choices=["Z2","U1"], help="symmetry type")
parser.add_argument(
    "--yast_backend",
    type=str,
    default="torch",
    help="YAST backend",
    choices=["torch", "torch_cpp"],
)

args = parser.parse_args()  # process command line arguments

def main():
    args.CTMARGS_ctm_env_init_type = "eye"
    args.omp_cores = 4
    cfg.configure(args)
    cfg.print_config()

    if args.yast_backend == "torch":
        from yastn.yastn.backend import backend_torch as backend
    elif args.yast_backend=='torch_cpp':
        from yastn.yastn.backend import backend_torch_cpp as backend

    sym= sym_Z2
    if args.sym=="U1": sym_U1
    yastn_config = yastn.make_config(
        backend=backend,
        sym=sym,
        fermionic=True,
        default_device=cfg.global_args.device,
        default_dtype=cfg.global_args.dtype,
    )
    torch.set_num_threads(args.omp_cores)
    yastn_config.backend.random_seed(args.seed)
    model = tV_model(yastn_config, V1=args.V1, V2=args.V2, V3=args.V3, t1=args.t1, t2=args.t2, phi=args.phi, mu=args.mu, m=args.m)

    @torch.no_grad()
    def ctm_conv_check(env, history, corner_tol):
        converged, max_dsv, history = ctm_conv_corner_spec(env, history, corner_tol)
        print("max_dsv:", max_dsv)
        log.log(logging.INFO, f"CTM iter {len(history)} |delta_C| {max_dsv}")
        return converged, history


    def get_converged_env(env, method='2site', max_sweeps=100, iterator_step=1, opts_svd=None, corner_tol=1e-8):
        t_ctm, t_check = 0.0, 0.0
        t_ctm_prev = time.perf_counter()
        converged, conv_history = False, []

        prev_rdms = None
        for sweep in range(max_sweeps):
            env.update_(
                opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=True,
            )
            t_ctm_after = time.perf_counter()
            t_ctm += t_ctm_after - t_ctm_prev
            t_ctm_prev = t_ctm_after

            converged, conv_history = ctm_conv_check(env, conv_history, corner_tol)
            if converged:
                break
        env.update_(
            opts_svd=opts_svd, method=method, use_qr=False, checkpoint_move=True,
        )

        return env, converged, conv_history, t_ctm, t_check


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

        # 1) compute environment by CTMRG
        # ------fixed_point---------
        opts_svd = {
            "D_total": cfg.main_args.chi,
            "tol": cfg.ctm_args.projector_svd_reltol,
            "eps_multiplet": cfg.ctm_args.projector_eps_multiplet,
            "truncate_multiplets": True,
        }
        state_params = state.get_parameters()
        env_params, slices = env_raw_data(ctm_env_in)
        env_out_data = FixedPoint.apply(env_params, slices, yastn_config, ctm_env_in, opts_svd, cfg.main_args.chi, 1e-10, ctm_args, *state_params)
        ctm_env_out, ctm_log, t_ctm, t_check = FixedPoint.ctm_env_out, FixedPoint.ctm_log, FixedPoint.t_ctm, FixedPoint.t_check
        refill_env(ctm_env_out, env_out_data, FixedPoint.slices)

        # ------direct CTMRG---------
        # ctm_env_out, converged, *ctm_log, t_ctm, t_check = get_converged_env(
        #     ctm_env_in,
        #     max_sweeps=ctm_args.ctm_max_iter,
        #     iterator_step=1,
        #     opts_svd={
        #         "D_total": cfg.main_args.chi, "tol": ctm_args.projector_svd_reltol,
        #             "eps_multiplet": ctm_args.projector_eps_multiplet,
        #         },
        #     corner_tol=ctm_args.projector_svd_reltol
        # )

        # 2) evaluate loss with converged environment
        loss = model.energy_per_site(state, ctm_env_out)  # H= H_0 + mu * (nA + nB)
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
            print("LS " + ", ".join([f"{epoch}", f"{loss}"]+[f"{x.item()}" for x in model.get_parameters()]))
        else:
            epoch = len(opt_context["loss_history"]["loss"])
            loss = opt_context["loss_history"]["loss"][-1]
            print(", ".join([f"{epoch}", f"{loss}"]+[f"{x.item()}" for x in model.get_parameters()]))

        model.eval_obs(state, ctm_env)

    # choose initial state or load it from file / checkpoint
    if not args.instate and args.sym=="Z2":
        if args.ansatz in ["1x1",]:
            state= random_1x1_state(config=yastn_config, bond_dim=(args.bond_dim, args.bond_dim))
        elif args.ansatz in ["3x3",]:
            state= random_3x3_state(config=yastn_config, bond_dim=(args.bond_dim, args.bond_dim))
        else:
            raise ValueError("Missing ansatz specification --ansatz "\
                +str(args.ansatz)+" is not supported")
    elif args.instate!=None:
        state= load_PepsAD(yastn_config, args.instate)
        # TODO: extending bond dimensions
        state.add_noise(args.instate_noise)
    elif args.opt_resume is not None:
        state= state.load_checkpoint(yastn_config, args.opt_resume)

    conv_env = None
    print("\n\nepoch, loss,")
    optimize_state(state, conv_env, loss_fn, obs_fn=obs_fn, post_proc=post_proc)



class Test_1x1_CDW(unittest.TestCase):
    r"""
    Test case for CDW with 1x1 ansatz using Z2 symmetry and D_even,D_odd= 1,1.
    """
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_opt_1x1_CDW"

    def setUp(self):
        args.ansatz, args.sym= "1x1", "Z2"
        args.V1, args.V2, args.V3, args.t1, args.t2, args.t3, args.phi= 1.4, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0
        args.mu= args.V1*1.5
        args.m = 0
        args.bond_dim=1
        args.chi=20
        args.seed=120
        args.opt_max_iter= 35
        args.instate_noise=0
        # args.CTMARGS_ctm_env_init_type= "eye"
        # args.GLOBALARGS_dtype= "complex128"

    def test_opt_1x1_CDW(self):
        import builtins
        from unittest.mock import patch
        from io import StringIO

        args.out_prefix=self.OUT_PRFX
        # args.instate= args.out_prefix[len("RESULT_"):]+"_instate.json"

        # i) run optimization
        tmp_out= StringIO()
        original_print = builtins.print
        def passthrough_print(*args, **kwargs):
            original_print(*args, **kwargs)
            kwargs.update(file=tmp_out)
            original_print(*args, **kwargs)

        with patch('builtins.print', new=passthrough_print) as tmp_print:
            main()

        # parse FINAL observables

        obs_opt_lines=[]
        final_obs=None
        OPT_OBS= OPT_OBS_DONE= False
        tmp_out.seek(0)
        l= tmp_out.readline()
        while l:
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="":
                OPT_OBS_DONE= True
                OPT_OBS=False
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>1:
                obs_opt_lines.append(l)
            if "epoch, loss," in l and not OPT_OBS_DONE:
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert len(obs_opt_lines)>0
        print(obs_opt_lines)
        # compare the line of observables with lowest energy from optimization (i)
        # TODO and final observables evaluated from best state stored in *_state.json output file
        best_e_line_index= np.argmin([ float(l.split(',')[1]) for l in obs_opt_lines ])
        opt_line_last= [complex(x) for x in obs_opt_lines[best_e_line_index].split(",")]
        for val0,val1 in zip(opt_line_last, [35,-2.9280089] ):
            assert np.isclose(val0,val1, rtol=self.tol, atol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        out_prefix=self.OUT_PRFX
        for f in [out_prefix+"_checkpoint.p",out_prefix+"_state.json",out_prefix+".log"]:
            if os.path.isfile(f): os.remove(f)

if __name__ == "__main__":
    unittest.main()
