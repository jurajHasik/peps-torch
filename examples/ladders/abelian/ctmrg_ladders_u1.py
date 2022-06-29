import os
import context
import torch
import numpy as np
import argparse
import config as cfg
import examples.abelian.settings_full_torch as settings_full
import examples.abelian.settings_U1_torch as settings_U1
from ipeps.ipeps_abelian import *
from ctm.generic_abelian.env_abelian import *
import ctm.generic_abelian.ctmrg as ctmrg
from models.abelian import coupledLadders
from models import coupledLadders as coupledLadders_dense
import ctm.generic.ctmrg as ctmrg_dense
from ctm.generic import transferops
import json
import unittest
import logging
log = logging.getLogger(__name__)

# parse command line args and build necessary configuration objects
parser= cfg.get_args_parser()
# additional model-dependent arguments
parser.add_argument("--alpha", type=float, default=0., help="inter-ladder coupling")
parser.add_argument("--bz_stag", type=float, default=0., help="staggered magnetic field")
parser.add_argument("--top_n", type=int, default=2, help="number of leading eigenvalues"+
    "of transfer operator to compute")
args, unknown_args = parser.parse_known_args()

def main():
    cfg.configure(args)
    cfg.print_config()
    settings= settings_U1
    # override default device specified in settings
    settings.default_device= settings_full.default_device= cfg.global_args.device
    # override default dtype
    settings.default_dtype= settings_full.default_dtype= cfg.global_args.dtype
    settings.backend.set_num_threads(args.omp_cores)
    settings.backend.random_seed(args.seed)

    model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha)

    # initialize an ipeps
    # 1) define lattice-tiling function, that maps arbitrary vertex of square lattice
    # coord into one of coordinates within unit-cell of iPEPS ansatz
    if args.instate!=None:
        state= read_ipeps(args.instate, settings)
        state= state.add_noise(args.instate_noise)
    else:
        raise ValueError("Missing trial state: --instate=None and --ipeps_init_type= "\
            +str(args.ipeps_init_type)+" is not supported")

    if not state.dtype==model.dtype:
        cfg.global_args.dtype= state.dtype
        print(f"dtype of initial state {state.dtype} and model {model.dtype} do not match.")
        print(f"Setting default dtype to {cfg.global_args.dtype} and reinitializing "\
        +" the model")
        model= coupledLadders.COUPLEDLADDERS_NOSYM(settings_full,alpha=args.alpha,\
            Bz_val=args.bz_stag)

    print(state)

    # 2) define convergence criterion for ctmrg
    @torch.no_grad()
    def ctmrg_conv_energy(state, env, history, ctm_args=cfg.ctm_args):
        if not history:
            history=[]
        e_curr = model.energy_2x1_1x2(state, env).item()
        history.append(e_curr)
        obs_values, obs_labels = model.eval_obs(state, env)
        # obs_values, obs_labels= ["None"], [None]
        print(", ".join([f"{len(history)}",f"{e_curr}"]+[f"{v}" for v in obs_values]))


        if (len(history) > 1 and abs(history[-1]-history[-2]) < ctm_args.ctm_conv_tol)\
            or len(history) >= ctm_args.ctm_max_iter:
            return True, history
        return False, history

    ctm_env= ENV_ABELIAN(args.chi, state=state, init=True)

    # 3) evaluate observables for initial environment
    loss= model.energy_2x1_1x2(state, ctm_env).to_number()
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    print(", ".join(["epoch","energy"]+obs_labels))
    print(", ".join([f"{-1}",f"{loss}"]+[f"{v}" for v in obs_values]))

    # 4) execute ctmrg
    ctm_env, *ctm_log = ctmrg.run(state, ctm_env, conv_check=ctmrg_conv_energy)

    # 5) compute final observables and timings
    loss= model.energy_2x1_1x2(state, ctm_env).to_number()
    obs_values, obs_labels= model.eval_obs(state,ctm_env)
    history, t_ctm, t_obs= ctm_log
    print("\n")
    print(", ".join(["epoch","energy"]+obs_labels))
    print("FINAL "+", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))
    print(f"TIMINGS ctm: {t_ctm} conv_check: {t_obs}")

    # environment diagnostics
    for c_loc,c_ten in ctm_env.C.items(): 
        u,s,v= c_ten.svd(([0],[1]))
        print(f"\n\nspectrum C[{c_loc}]")
        for charges in s.get_blocks_charge():
            print(charges)
            sector= s[charges]
            for i in range(len(sector)):
                print(f"{i} {sector[i]}")

    # convert to dense env and compute transfer operator spectrum
    state_dense= state.to_dense()
    ctm_env_dense= ctm_env.to_dense(state)

    # CORRECTNESS check 
    #
    # for c_loc,c_ten in ctm_env_dense.C.items(): 
    #     u,s,v= torch.svd(c_ten, compute_uv=False)
    #     print(f"\n\nspectrum C[{c_loc}]")
    #     for i in range(args.chi):
    #         print(f"{i} {s[i]}")
    #
    # model_dense= coupledLadders_dense.COUPLEDLADDERS(alpha=args.alpha)
    # loss= model_dense.energy_2x1_1x2(state_dense, ctm_env_dense)
    # obs_values, obs_labels= model_dense.eval_obs(state_dense,ctm_env_dense)
    # print(", ".join(["energy"]+obs_labels))
    # print(", ".join([f"{loss}"]+[f"{v}" for v in obs_values]))

    site_dir_list=[((0,0), (1,0)),((0,0), (0,1)), ((1,1), (1,0)), ((1,1), (0,1))]
    for sdp in site_dir_list:
        print(f"\n\nspectrum(T)[{sdp[0]},{sdp[1]}]")
        l= transferops.get_Top_spec(args.top_n, *sdp, state_dense, ctm_env_dense)
        for i in range(l.size()[0]):
            print(f"{i} {l[i,0]} {l[i,1]}")

if __name__=='__main__':
    if len(unknown_args)>0:
        print("args not recognized: "+str(unknown_args))
        raise Exception("Unknown command line arguments")
    main()


class TestCtmrg_plain_VBS(unittest.TestCase):
    tol= 1.0e-6
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    OUT_PRFX = "RESULT_test_run_ctmrg_u1_VBS"

    def setUp(self):
        args.instate=self.DIR_PATH+"/../../../test-input/abelian/VBS_2x2_ABCD.in"
        args.alpha=0.2
        args.chi=16
        args.out_prefix=self.OUT_PRFX

    def test_run_ctmrg_u1_VBS(self):
        from io import StringIO
        from unittest.mock import patch 
        from cmath import isclose

        with patch('sys.stdout', new = StringIO()) as tmp_out: 
            main()
        tmp_out.seek(0)

        # parse FINAL observables
        final_obs=None
        final_opt_line=None
        OPT_OBS= OPT_OBS_DONE= False
        l= tmp_out.readline()
        while l:
            print(l,end="")
            if OPT_OBS and not OPT_OBS_DONE and l.rstrip()=="": OPT_OBS_DONE= True
            if OPT_OBS and not OPT_OBS_DONE and len(l.split(','))>2:
                final_opt_line= l
            if "epoch, energy," in l and not OPT_OBS_DONE: 
                OPT_OBS= True
            if "FINAL" in l:
                final_obs= l.rstrip()
                break
            l= tmp_out.readline()
        assert final_obs
        assert final_opt_line

        # compare with the reference
        ref_data="""
        -0.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        -0.75, -0.75, 0.0, 0.0
        """

        # compare final observables from final state against expected reference 
        # drop first token, corresponding to iteration step
        fobs_tokens= [complex(x) for x in final_obs[len("FINAL"):].split(",")]
        ref_tokens= [complex(x) for x in ref_data.split(",")]
        for val,ref_val in zip(fobs_tokens, ref_tokens):
            assert isclose(val,ref_val, rel_tol=self.tol, abs_tol=self.tol)

    def tearDown(self):
        args.opt_resume=None
        args.instate=None
        for f in [self.OUT_PRFX+"_state.json",self.OUT_PRFX+"_checkpoint.p",self.OUT_PRFX+".log"]:
            if os.path.isfile(f): os.remove(f)